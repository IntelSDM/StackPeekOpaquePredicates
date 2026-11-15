#include "StackPeekOpaquePass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> StackPeekOpaqueEnable("obfstkpeek", llvm::cl::init(true), llvm::cl::desc("Stack-peek opaque predicates pass."));

static constexpr unsigned TrueCountMin = 1;
static constexpr unsigned TrueCountMax = 4;
static constexpr unsigned FalseCountMin = 2;
static constexpr unsigned FalseCountMax = 5;
static constexpr unsigned ChannelSize = TrueCountMax + FalseCountMax;
static constexpr unsigned DonorCloneCap = 6;
static constexpr unsigned DecoyMinTailOps = 2;
static constexpr unsigned DecoyMaxTailOps = 6;
static constexpr unsigned DecoyStoreChancePct = 10;

static constexpr unsigned RspFalseCountMin = 3;
static constexpr unsigned RspFalseCountMax = 6;

// RspOffsetMax stays at 0x1000 to keep the read inside the first stack page below RSP where Windows x64 stack commit guarantees that a first guard page hit commits silently and execution continues
static constexpr int32_t RspOffsetMin = 0x10;
static constexpr int32_t RspOffsetMax = 0x1000;

// each entry is an impossible u64 that real stack data has ~2^-64 probability of matching so the hash check against it is always false at runtime
static constexpr uint64_t EmojiImpossible[] = {
    0xF09F9880F09F9880ULL,
    0xF09F9280F09F9280ULL,
    0xF09F8E83F09F8E83ULL,
    0xF09FA496F09FA496ULL,
    0xF09F91BEF09F91BEULL,
    0xF09F94A5F09F94A5ULL,
    0xF09F92A5F09F92A5ULL,
    0xF09F9880F09F9280ULL,
    0xF09F8E83F09FA496ULL,
    0xF09F91BEF09F94A5ULL,
    0xF09F9280F09F92A5ULL,
    0xF09FA496F09F9880ULL,
};
static constexpr unsigned EmojiImpossibleCount = sizeof(EmojiImpossible) / sizeof(EmojiImpossible[0]);

static bool HasAnnotationTag(Function& f, const char* tag)
{
    Module* m = f.getParent();
    GlobalVariable* ga = m->getGlobalVariable("llvm.global.annotations");
    if (!ga || !ga->hasInitializer()) return false;
    auto* ca = dyn_cast<ConstantArray>(ga->getOperand(0));
    if (!ca) return false;
    for (unsigned i = 0; i < ca->getNumOperands(); ++i)
    {
        auto* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs || cs->getNumOperands() < 2) continue;
        if (cs->getOperand(0)->stripPointerCasts() != &f) continue;
        auto* annogv = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!annogv || !annogv->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataSequential>(annogv->getInitializer());
        if (!cds || !cds->isString()) continue;
        if (strcmp(cds->getAsString().str().c_str(), tag) == 0) return true;
    }
    return false;
}

static bool HasObfuscateAnnotation(Function& f)   { return HasAnnotationTag(f, "obfus"); }
static bool HasNoObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "noobfus"); }
static bool HasNoStackPeekAnnotation(Function& f) { return HasAnnotationTag(f, "nostkpeek"); }

static std::mt19937_64 Rng;

static bool FunctionAlreadyProcessed(Function& f)
{
    BasicBlock* bb = &f.getEntryBlock();
    Instruction* term = bb->getTerminator();
    return term && term->getMetadata("stkpeek.done") != nullptr;
}

static void MarkFunctionProcessed(Function& f)
{
    BasicBlock* bb = &f.getEntryBlock();
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("stkpeek.done", MDNode::get(bb->getContext(), {}));
}

static bool IsDecoyBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    return term && term->getMetadata("stkpeek.decoy") != nullptr;
}

static void MarkDecoyBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("stkpeek.decoy", MDNode::get(bb->getContext(), {}));
}

static void TagEscalateConstMBA(Value* v)
{
    auto* inst = dyn_cast<Instruction>(v);
    if (!inst) return;
    LLVMContext& c = inst->getContext();
    Metadata* ops[1] = { MDString::get(c, "poly_const") };
    inst->setMetadata("mbagen.escalate", MDNode::get(c, ops));
}

// empty body inline asm with "=r,0" takes zero final-binary instructions but makes LLVM treat its input as an opaque runtime value so constant folding cannot see through the identity
static Value* InsertFoldBarrier(IRBuilder<>& b, Value* v)
{
    Type* ty = v->getType();
    if (!ty->isIntegerTy()) return v;
    FunctionType* asmty = FunctionType::get(ty, { ty }, false);
    InlineAsm* ia = InlineAsm::get(asmty, "", "=r,0", true);
    return b.CreateCall(ia, { v }, "");
}

static GlobalVariable* GetOrCreateStackPeekSink(Module& m)
{
    const char* name = ".stkpeek.sink";
    if (auto* existing = m.getGlobalVariable(name)) return existing;
    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);
    auto* gv = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    gv->setAlignment(Align(8));
    return gv;
}

static Value* EmitSinkLoad(IRBuilder<>& b, Module& m, IntegerType* i64, const char* name)
{
    GlobalVariable* sink = GetOrCreateStackPeekSink(m);
    LoadInst* ld = b.CreateLoad(i64, sink, true, name);
    ld->setAlignment(Align(8));
    return ld;
}

struct HashConsts
{
    uint64_t K0;
    uint64_t K1;
    uint64_t K2;
    uint64_t K3;
    uint64_t Kmix;
};

static HashConsts RollHashConsts(std::mt19937_64& rng)
{
    HashConsts h;
    h.K0 = ((uint64_t)rng()) | 1ull;
    h.K1 = ((uint64_t)rng()) | 1ull;
    h.K2 = ((uint64_t)rng()) | 1ull;
    h.K3 = ((uint64_t)rng()) | 1ull;
    h.Kmix = (uint64_t)rng();
    return h;
}

// both sides of each hash eq check run EmitHashIR at runtime with the same HashConsts so hash(v)==hash(c) iff v==c; the FoldBarrier wraps on both inputs are what stop LLVM from recognising the bijection and collapsing the compare to v==c
static Value* EmitHashIR(IRBuilder<>& b, Value* xi64, IntegerType* i64, const HashConsts& hc)
{
    Value* r = InsertFoldBarrier(b, xi64);

    Value* s1 = b.CreateLShr(r, ConstantInt::get(i64, 30), "");
    r = b.CreateXor(r, s1, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K0), "");
    Value* s2 = b.CreateLShr(r, ConstantInt::get(i64, 27), "");
    r = b.CreateXor(r, s2, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K1), "");
    Value* s3 = b.CreateLShr(r, ConstantInt::get(i64, 31), "");
    r = b.CreateXor(r, s3, "");

    Value* kmix = InsertFoldBarrier(b, ConstantInt::get(i64, hc.Kmix));
    r = b.CreateXor(r, kmix, "");

    Value* s4 = b.CreateLShr(r, ConstantInt::get(i64, 29), "");
    r = b.CreateXor(r, s4, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K2), "");
    Value* s5 = b.CreateLShr(r, ConstantInt::get(i64, 25), "");
    r = b.CreateXor(r, s5, "");
    r = b.CreateMul(r, ConstantInt::get(i64, hc.K3), "");
    Value* s6 = b.CreateLShr(r, ConstantInt::get(i64, 32), "");
    r = b.CreateXor(r, s6, "");

    return r;
}

struct PolluteSlot
{
    AllocaInst* Base;
    int FieldIdx;
    Type* BaseTy;
    Type* RawTy;
    bool IsPtr;
    IntegerType* IntTy;
};

static Value* MaterializeSlotPtr(IRBuilder<>& b, const PolluteSlot& s)
{
    if (s.FieldIdx < 0) return s.Base;
    LLVMContext& c = b.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    return b.CreateInBoundsGEP(s.BaseTy, s.Base, { ConstantInt::get(i32, 0), ConstantInt::get(i32, s.FieldIdx) }, "");
}

static Value* EmitSlotLoad(IRBuilder<>& b, const PolluteSlot& s)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    LoadInst* ld = b.CreateLoad(s.RawTy, ptr, "");
    if (!s.IsPtr) return ld;
    return b.CreatePtrToInt(ld, s.IntTy, "");
}

static void EmitSlotStore(IRBuilder<>& b, const PolluteSlot& s, Value* intval, bool volatilestore)
{
    Value* ptr = MaterializeSlotPtr(b, s);
    Value* v = intval;
    if (s.IsPtr) v = b.CreateIntToPtr(intval, s.RawTy, "");
    StoreInst* st = b.CreateStore(v, ptr);
    if (volatilestore) st->setVolatile(true);
}

static void CollectPolluteSlots(Function& f, IntegerType* i32, IntegerType* i64, SmallVectorImpl<PolluteSlot>& out32, SmallVectorImpl<PolluteSlot>& out64)
{
    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        Type* ty = a->getAllocatedType();

        if (ty->isIntegerTy(32))
        {
            out32.push_back({a, -1, ty, ty, false, i32});
            continue;
        }
        if (ty->isIntegerTy(64))
        {
            out64.push_back({a, -1, ty, ty, false, i64});
            continue;
        }
        if (ty->isPointerTy())
        {
            out64.push_back({a, -1, ty, ty, true, i64});
            continue;
        }
        if (auto* st = dyn_cast<StructType>(ty))
        {
            for (unsigned fi = 0; fi < st->getNumElements(); ++fi)
            {
                Type* ft = st->getElementType(fi);
                if (ft->isIntegerTy(32)) out32.push_back({a, (int)fi, ty, ft, false, i32});
                else if (ft->isIntegerTy(64)) out64.push_back({a, (int)fi, ty, ft, false, i64});
                else if (ft->isPointerTy()) out64.push_back({a, (int)fi, ty, ft, true, i64});
            }
        }
    }
}

static bool CanCloneWithDominance(const Instruction& i, BasicBlock* switchsrc, DominatorTree& dt, const ValueToValueMapTy& vmap)
{
    for (const Use& u : i.operands())
    {
        Value* op = u.get();
        if (!op) continue;
        if (isa<Constant>(op)) continue;
        if (isa<Argument>(op)) continue;
        if (isa<BasicBlock>(op)) continue;
        if (isa<MetadataAsValue>(op)) continue;
        if (vmap.count(op)) continue;
        if (auto* oi = dyn_cast<Instruction>(op))
        {
            if (dt.dominates(oi, switchsrc->getTerminator())) continue;
            return false;
        }
        return false;
    }
    return true;
}

static unsigned DiamondCloneDonorInto(BasicBlock* donor, BasicBlock* decoy, BasicBlock* switchsrc, DominatorTree& dt, std::mt19937_64& rng, unsigned maxclone)
{
    if (maxclone == 0) return 0;
    if (!donor || !decoy) return 0;

    ValueToValueMapTy vmap;
    unsigned cloned = 0;
    const uint64_t mutnonce = rng();

    for (Instruction& i : *donor)
    {
        if (cloned >= maxclone) break;
        if (i.isTerminator()) break;

        if (isa<PHINode>(&i)) continue;
        if (isa<AllocaInst>(&i)) continue;
        if (isa<CallBase>(&i)) continue;
        if (isa<LandingPadInst>(&i)) continue;
        if (isa<CatchPadInst>(&i)) continue;
        if (isa<CleanupPadInst>(&i)) continue;
        if (i.isAtomic()) continue;
        if (isa<FenceInst>(&i)) continue;
        if (i.mayHaveSideEffects()) continue;

        if (!CanCloneWithDominance(i, switchsrc, dt, vmap)) continue;

        Instruction* clone = i.clone();
        clone->setName("");
        clone->insertInto(decoy, decoy->end());
        RemapInstruction(clone, vmap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

        clone->setMetadata("mbagen.escalate", nullptr);

        if (auto* bo = dyn_cast<BinaryOperator>(clone))
        {
            if ((rng() & 1u) == 0)
            {
                Instruction::BinaryOps newop = bo->getOpcode();
                switch (bo->getOpcode())
                {
                    case Instruction::Add: newop = Instruction::Sub; break;
                    case Instruction::Sub: newop = Instruction::Add; break;
                    case Instruction::And: newop = Instruction::Or; break;
                    case Instruction::Or: newop = Instruction::And; break;
                    case Instruction::Xor: newop = Instruction::Or; break;
                    case Instruction::Mul: newop = Instruction::Add; break;
                    default: break;
                }
                if (newop != bo->getOpcode())
                {
                    IRBuilder<> bob(bo);
                    Value* nb = bob.CreateBinOp(newop, bo->getOperand(0), bo->getOperand(1));
                    bo->replaceAllUsesWith(nb);
                    vmap[&i] = nb;
                    bo->eraseFromParent();
                    ++cloned;
                    continue;
                }
            }
            for (unsigned op = 0; op < bo->getNumOperands(); ++op)
            {
                if (auto* ci = dyn_cast<ConstantInt>(bo->getOperand(op)))
                {
                    if ((rng() % 100u) < 50u)
                    {
                        bo->setOperand(op, ConstantInt::get(ci->getType(), ci->getZExtValue() ^ mutnonce));
                    }
                }
            }
        }

        vmap[&i] = clone;
        ++cloned;
    }

    return cloned;
}

// the empty BB terminator null check must precede isEHPad because BasicBlock::isEHPad null derefs when the BB has no instructions
static void CollectDonors(Function& f, BasicBlock* dispatchhead, const SmallPtrSetImpl<BasicBlock*>& excludes, SmallVectorImpl<BasicBlock*>& out)
{
    for (BasicBlock& bb : f)
    {
        if (&bb == dispatchhead) continue;
        if (!bb.getTerminator()) continue;
        if (bb.isEHPad()) continue;
        if (IsDecoyBlock(&bb)) continue;
        if (excludes.count(&bb)) continue;
        out.push_back(&bb);
    }
}

static void FillDecoyBlock(Function& f, BasicBlock* decoy, BasicBlock* dispatchsrc, BasicBlock* fallthru, ArrayRef<BasicBlock*> donors, DominatorTree& dt, std::mt19937_64& rng, const SmallVectorImpl<PolluteSlot>& slots32, const SmallVectorImpl<PolluteSlot>& slots64, unsigned donorclonecap)
{
    LLVMContext& c = f.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);

    unsigned diamondcloned = 0;
    if (!donors.empty())
    {
        BasicBlock* donor = donors[rng() % donors.size()];
        diamondcloned = DiamondCloneDonorInto(donor, decoy, dispatchsrc, dt, rng, donorclonecap);
    }

    IRBuilder<> fb(decoy);
    fb.SetInsertPoint(decoy);

    const bool have32 = !slots32.empty();
    const bool have64 = !slots64.empty();

    if (!have32 && !have64)
    {
        (void)EmitSinkLoad(fb, *f.getParent(), i64, "");
        fb.CreateBr(fallthru);
        MarkDecoyBlock(decoy);
        return;
    }

    const unsigned opspan = DecoyMaxTailOps - DecoyMinTailOps + 1u;
    unsigned targetops = DecoyMinTailOps + (unsigned)(rng() % opspan);
    if (diamondcloned >= targetops) targetops = 1;
    else targetops -= diamondcloned;

    Value* acc = nullptr;
    IntegerType* accty = nullptr;

    auto pickslot = [&](IntegerType* ity) -> const PolluteSlot&
    {
        auto& slots = (ity == i64) ? slots64 : slots32;
        return slots[rng() % slots.size()];
    };

    for (unsigned n = 0; n < targetops; ++n)
    {
        IntegerType* ity;
        if (have32 && have64) ity = (rng() & 1) ? i64 : i32;
        else ity = have64 ? i64 : i32;

        if (acc && accty != ity) acc = nullptr;

        if (!acc)
        {
            acc = EmitSlotLoad(fb, pickslot(ity));
            accty = ity;
            continue;
        }

        Value* rhs = EmitSlotLoad(fb, pickslot(ity));
        switch (rng() % 6)
        {
            case 0: acc = fb.CreateXor(acc, rhs, ""); break;
            case 1: acc = fb.CreateAdd(acc, rhs, ""); break;
            case 2: acc = fb.CreateSub(acc, rhs, ""); break;
            case 3: acc = fb.CreateAnd(acc, rhs, ""); break;
            case 4: acc = fb.CreateOr(acc, rhs, ""); break;
            default: acc = fb.CreateMul(acc, rhs, ""); break;
        }

        if ((rng() % 100u) < DecoyStoreChancePct)
        {
            EmitSlotStore(fb, pickslot(ity), acc, false);
        }
    }

    if (acc)
    {
        EmitSlotStore(fb, pickslot(accty), acc, true);
    }

    fb.CreateBr(fallthru);
    MarkDecoyBlock(decoy);
}

//  the terminator null check must precede isEHPad because BasicBlock::isEHPad null derefs on empty BBs
static bool IsEHOrFuncletInterior(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    if (bb->isEHPad()) return true;
    if (isa<CatchReturnInst>(term) || isa<CleanupReturnInst>(term) || isa<ResumeInst>(term)) return true;
    for (Instruction& i : *bb)
    {
        if (auto* cb = dyn_cast<CallBase>(&i))
        {
            if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) return true;
        }
    }
    return false;
}

// head sweep starts from the first instruction this pass emitted rather than head->begin() because head's pre existing allocas legitimately host ConstMBA on their user sites and tagging them would suppress legit amplification
static void ApplyBlanketSkipTag(LLVMContext& c, ArrayRef<BasicBlock*> allemittedbbs, Instruction* firstemittedinhead, BasicBlock* headbb)
{
    Metadata* skipops[1] = { MDString::get(c, "stkpeek_skip") };
    MDNode* skipnode = MDNode::get(c, skipops);

    auto blanket = [&](BasicBlock* target, Instruction* from)
    {
        BasicBlock::iterator it = from ? BasicBlock::iterator(from) : target->begin();
        for (; it != target->end(); ++it)
        {
            MDNode* existing = it->getMetadata("mbagen.escalate");
            if (existing && existing->getNumOperands() > 0)
            {
                if (auto* ms = dyn_cast<MDString>(existing->getOperand(0).get()))
                {
                    if (ms->getString() == "poly_const") continue;
                }
            }
            it->setMetadata("mbagen.escalate", skipnode);
        }
    };

    blanket(headbb, firstemittedinhead);
    for (BasicBlock* b : allemittedbbs)
    {
        if (b == headbb) continue;
        blanket(b, nullptr);
    }
}

struct ConstSlot
{
    AllocaInst* Alloca;
    uint64_t Value;
    Type* ElemTy;
    StoreInst* Store;
};

static bool AllocaUsersAreDirectLoadStore(AllocaInst* a, StoreInst*& outstore)
{
    outstore = nullptr;
    unsigned storecount = 0;
    for (User* u : a->users())
    {
        if (auto* ld = dyn_cast<LoadInst>(u))
        {
            if (ld->getPointerOperand() != a) return false;
            continue;
        }
        if (auto* st = dyn_cast<StoreInst>(u))
        {
            if (st->getValueOperand() == a) return false;
            if (st->getPointerOperand() != a) return false;
            outstore = st;
            ++storecount;
            continue;
        }
        return false;
    }
    return storecount == 1 && outstore != nullptr;
}

static void CollectConstSlots(Function& f, SmallVectorImpl<ConstSlot>& out)
{
    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        Type* ty = a->getAllocatedType();
        if (!ty->isIntegerTy(32) && !ty->isIntegerTy(64)) continue;

        StoreInst* thestore = nullptr;
        if (!AllocaUsersAreDirectLoadStore(a, thestore)) continue;

        auto* ci = dyn_cast<ConstantInt>(thestore->getValueOperand());
        if (!ci) continue;

        out.push_back({a, ci->getZExtValue(), ty, thestore});
    }
}

static void CollectNonConstSlots(Function& f, const SmallVectorImpl<ConstSlot>& constslots, SmallVectorImpl<AllocaInst*>& out)
{
    SmallPtrSet<AllocaInst*, 16> constset;
    for (const auto& cs : constslots) constset.insert(cs.Alloca);

    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        if (constset.count(a)) continue;
        Type* ty = a->getAllocatedType();
        if (!ty->isIntegerTy(32) && !ty->isIntegerTy(64)) continue;

        bool ok = true;
        for (User* u : a->users())
        {
            if (auto* ld = dyn_cast<LoadInst>(u))
            {
                if (ld->getPointerOperand() != a)
                {
                    ok = false;
                    break;
                }
            }
            else if (auto* st = dyn_cast<StoreInst>(u))
            {
                if (st->getValueOperand() == a)
                {
                    ok = false;
                    break;
                }
                if (st->getPointerOperand() != a)
                {
                    ok = false;
                    break;
                }
            }
            else
            {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        out.push_back(a);
    }
}

// any none call use address takes the function; InvokeInst's normal dest is not a splittable return site; musttail calls are themselves terminators so inserting before them is undefined; funclet interior callers break token threading
static CallInst* FindSingleSafeCaller(Function& g)
{
    CallInst* thecall = nullptr;
    unsigned count = 0;

    if (g.use_empty()) return nullptr;

    for (Use& u : g.uses())
    {
        User* user = u.getUser();
        auto* cb = dyn_cast<CallBase>(user);
        if (!cb) return nullptr;
        if (!cb->isCallee(&u)) return nullptr;
        if (isa<InvokeInst>(cb)) return nullptr;
        auto* ci = dyn_cast<CallInst>(cb);
        if (!ci) return nullptr;
        Function* caller = ci->getFunction();
        if (!caller) return nullptr;
        if (caller == &g) return nullptr;
        if (IsEHOrFuncletInterior(ci->getParent())) return nullptr;
        if (ci->isMustTailCall()) return nullptr;
        if (!ci->getNextNode()) return nullptr;

        ++count;
        if (count > 1) return nullptr;
        thecall = ci;
    }
    return (count == 1) ? thecall : nullptr;
}

static GlobalVariable* CreateChannel(Module& m, const std::string& basename)
{
    LLVMContext& c = m.getContext();
    PointerType* ptrty = PointerType::getUnqual(c);
    ArrayType* arrty = ArrayType::get(ptrty, ChannelSize);

    std::string name = ".stkpeek.ch." + basename + "." + std::to_string(Rng() & 0xFFFFFFFFull);

    auto* gv = new GlobalVariable(m, arrty, false, GlobalValue::InternalLinkage, ConstantAggregateZero::get(arrty), name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    gv->setAlignment(Align(8));
    return gv;
}

static void EmitPlantingAtCallsite(CallInst* ci, GlobalVariable* channel, ArrayRef<Value*> ptrs)
{
    LLVMContext& c = ci->getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    ArrayType* arrty = cast<ArrayType>(channel->getValueType());

    IRBuilder<> b(ci);
    for (unsigned i = 0; i < ptrs.size() && i < ChannelSize; ++i)
    {
        Value* slot = b.CreateInBoundsGEP(arrty, channel, { ConstantInt::get(i32, 0), ConstantInt::get(i32, i) }, "");
        StoreInst* st = b.CreateStore(ptrs[i], slot);
        st->setVolatile(true);
        st->setAlignment(Align(8));
    }
}

struct CheckSpec
{
    enum KindE { True, False, RspFalse };
    KindE Kind;
    unsigned ChannelIdx;
    Type* LoadTy;
    uint64_t ExpectedValue;
    int32_t RspOffset;
};

static Instruction* PickInsertionPointInBB(BasicBlock* bb, std::mt19937_64& rng)
{
    Instruction* first = bb->getFirstNonPHIOrDbgOrLifetime();
    if (!first) return nullptr;
    if (bb->size() < 2) return nullptr;

    if (bb == &bb->getParent()->getEntryBlock())
    {
        Instruction* lastalloca = nullptr;
        for (Instruction& i : *bb)
            if (isa<AllocaInst>(&i)) lastalloca = &i;
        if (lastalloca)
        {
            Instruction* after = lastalloca->getNextNode();
            if (!after) return nullptr;
            BasicBlock::iterator seek(first);
            while (seek != bb->end() && &*seek != after) ++seek;
            if (seek != bb->end()) first = &*seek;
            else first = after;
        }
    }

    unsigned splittable = 0;
    for (auto it = BasicBlock::iterator(first); it != bb->end(); ++it)
    {
        if (&*it == bb->getTerminator()) break;
        ++splittable;
    }
    if (splittable == 0) return nullptr;

    const unsigned pick = (unsigned)(rng() % splittable);
    BasicBlock::iterator it(first);
    for (unsigned i = 0; i < pick; ++i) ++it;
    if (it == bb->end()) return nullptr;
    if (&*it == bb->getTerminator()) return nullptr;
    if (isa<PHINode>(&*it)) return nullptr;
    return &*it;
}

// explicit Intel dialect because Windows x64 targeted LLVM defaults to Intel and AT&T syntax with %%rsp does not get preprocessed through so the assembler sees literal %%rsp and rejects it
static Value* EmitGetRSP(IRBuilder<>& b, IntegerType* i64)
{
    FunctionType* asmty = FunctionType::get(i64, {}, false);
    InlineAsm* ia = InlineAsm::get(asmty, "mov $0, rsp", "=r", true, false, InlineAsm::AD_Intel);
    return b.CreateCall(ia, {}, "");
}

static bool EmitSingleCheckAtBB(Function& g, BasicBlock* bb, GlobalVariable* channel, const CheckSpec& spec, const SmallPtrSetImpl<BasicBlock*>& ourbbs)
{
    if (IsDecoyBlock(bb)) return false;
    if (IsEHOrFuncletInterior(bb)) return false;
    if (ourbbs.count(bb)) return false;

    Instruction* splitat = PickInsertionPointInBB(bb, Rng);
    if (!splitat) return false;

    LLVMContext& c = g.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);
    PointerType* ptrty = PointerType::getUnqual(c);
    ArrayType* arrty = (channel != nullptr) ? cast<ArrayType>(channel->getValueType()) : nullptr;
    if (spec.Kind != CheckSpec::RspFalse && !channel) return false;

    BasicBlock* realbb = bb->splitBasicBlock(splitat, "Real");

    // the split inserts an unconditional br that we erase so bb has no terminator between erase and the new CondBr; no walker API may touch bb until the terminator is in place
    Instruction* splitbr = bb->getTerminator();
    Instruction* lastexisting = splitbr->getPrevNode();
    splitbr->eraseFromParent();

    IRBuilder<> builder(bb);

    Value* v = nullptr;
    if (spec.Kind == CheckSpec::RspFalse)
    {
        Value* rsp = EmitGetRSP(builder, i64);
        Value* off = ConstantInt::get(i64, (uint64_t)spec.RspOffset);
        Value* addrint = builder.CreateSub(rsp, off, "");
        Value* addr = builder.CreateIntToPtr(addrint, ptrty, "");
        LoadInst* ld = builder.CreateLoad(i64, addr, true, "");
        // alignment 1 because the stack slot at an arbitrary negative offset is not guaranteed 8 byte aligned; x86/x64 handles misalignment with minor cost and the compare never fires on real stack data so the hash path absorbs any unaligned load latency
        ld->setAlignment(Align(1));
        v = ld;
    }
    else
    {
        Value* chgep = builder.CreateInBoundsGEP(arrty, channel, { ConstantInt::get(i32, 0), ConstantInt::get(i32, spec.ChannelIdx) }, "");

        LoadInst* ptrld = builder.CreateLoad(ptrty, chgep, true, "");
        ptrld->setAlignment(Align(8));

        if (spec.LoadTy->isIntegerTy(32))
        {
            LoadInst* ld = builder.CreateLoad(i32, ptrld, "");
            ld->setAlignment(Align(4));
            v = builder.CreateZExt(ld, i64, "");
        }
        else
        {
            LoadInst* ld = builder.CreateLoad(i64, ptrld, "");
            ld->setAlignment(Align(8));
            v = ld;
        }
    }

    HashConsts hc = RollHashConsts(Rng);

    Value* hv = EmitHashIR(builder, v, i64, hc);
    Value* expected = ConstantInt::get(i64, spec.LoadTy->isIntegerTy(32) ? (spec.ExpectedValue & 0xFFFFFFFFULL) : spec.ExpectedValue);
    Value* he = EmitHashIR(builder, expected, i64, hc);

    const uint64_t nonce = Rng();
    const uint64_t esck = Rng();
    Value* noncev = InsertFoldBarrier(builder, ConstantInt::get(i64, nonce));
    Value* lhs = builder.CreateXor(hv, noncev, "");
    lhs = builder.CreateXor(lhs, ConstantInt::get(i64, esck), "");
    Value* rhs = builder.CreateXor(he, noncev, "");
    rhs = builder.CreateXor(rhs, ConstantInt::get(i64, esck), "");
    TagEscalateConstMBA(rhs);

    Value* m = builder.CreateICmpEQ(lhs, rhs, "");

    BasicBlock* decoy = BasicBlock::Create(c, "Decoy", &g);

    if (spec.Kind == CheckSpec::True) builder.CreateCondBr(m, realbb, decoy);
    else builder.CreateCondBr(m, decoy, realbb);

    SmallPtrSet<BasicBlock*, 32> excludes;
    excludes.insert(ourbbs.begin(), ourbbs.end());
    excludes.insert(bb);
    excludes.insert(realbb);
    excludes.insert(decoy);
    SmallVector<BasicBlock*, 32> donors;
    CollectDonors(g, bb, excludes, donors);

    SmallVector<PolluteSlot, 32> slots32, slots64;
    CollectPolluteSlots(g, i32, i64, slots32, slots64);

    DominatorTree dt;
    dt.recalculate(g);

    FillDecoyBlock(g, decoy, bb, realbb, donors, dt, Rng, slots32, slots64, DonorCloneCap);

    SmallVector<BasicBlock*, 4> allemitted{bb, decoy};
    Instruction* firstemitted = lastexisting ? lastexisting->getNextNode() : &bb->front();
    ApplyBlanketSkipTag(c, allemitted, firstemitted, bb);

    return true;
}

static bool EmitChecksAcrossBody(Function& g, GlobalVariable* channel, ArrayRef<CheckSpec> checks)
{
    if (checks.empty()) return false;
    if (FunctionAlreadyProcessed(g)) return false;
    if (g.isDeclaration() || g.empty()) return false;

    SmallVector<BasicBlock*, 64> candidates;
    for (BasicBlock& bb : g)
    {
        if (!bb.getTerminator()) continue;
        if (bb.isEHPad()) continue;
        if (IsEHOrFuncletInterior(&bb)) continue;
        if (IsDecoyBlock(&bb)) continue;
        if (bb.size() < 2) continue;
        candidates.push_back(&bb);
    }
    if (candidates.empty()) return false;

    const unsigned n = std::min<unsigned>((unsigned)checks.size(), (unsigned)candidates.size());
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned j = i + (unsigned)(Rng() % ((unsigned)candidates.size() - i));
        std::swap(candidates[i], candidates[j]);
    }

    SmallPtrSet<BasicBlock*, 32> ourbbs;

    unsigned emitted = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        BasicBlock* bb = candidates[i];
        if (ourbbs.count(bb)) continue;

        if (!EmitSingleCheckAtBB(g, bb, channel, checks[i], ourbbs)) continue;
        ++emitted;

        auto* br = cast<BranchInst>(bb->getTerminator());
        for (unsigned s = 0; s < br->getNumSuccessors(); ++s)
            ourbbs.insert(br->getSuccessor(s));
        ourbbs.insert(bb);
    }

    if (emitted == 0) return false;
    MarkFunctionProcessed(g);
    return true;
}

static bool CanModifyFunction(Function& g)
{
    if (g.isDeclaration()) return false;
    if (g.empty()) return false;
    if (g.isIntrinsic()) return false;
    if (g.hasFnAttribute(Attribute::AlwaysInline)) return false;
    if (g.hasFnAttribute(Attribute::Naked)) return false;
    if (HasNoObfuscateAnnotation(g)) return false;
    if (HasNoStackPeekAnnotation(g)) return false;
    return true;
}

static bool ProcessCallee(Function& g, Module& m)
{
    if (!CanModifyFunction(g)) return false;
    if (FunctionAlreadyProcessed(g)) return false;

    CallInst* thecall = FindSingleSafeCaller(g);
    Function* caller = thecall ? thecall->getFunction() : nullptr;

    SmallVector<ConstSlot, 16> constslots;
    SmallVector<AllocaInst*, 16> nonconstslots;
    if (caller)
    {
        CollectConstSlots(*caller, constslots);
        CollectNonConstSlots(*caller, constslots, nonconstslots);
    }

    auto pick = [](auto& pool, unsigned want, std::mt19937_64& rng)
    {
        unsigned have = (unsigned)pool.size();
        unsigned n = std::min<unsigned>(want, have);
        for (unsigned i = 0; i < n; ++i)
        {
            unsigned j = i + (unsigned)(rng() % (have - i));
            std::swap(pool[i], pool[j]);
        }
        pool.resize(n);
    };

    const unsigned trueavail = (unsigned)constslots.size();
    const unsigned falseavail = (unsigned)nonconstslots.size();

    const unsigned truespan = TrueCountMax - TrueCountMin + 1u;
    unsigned truewant = TrueCountMin + (unsigned)(Rng() % truespan);
    if (truewant > trueavail) truewant = trueavail;

    const unsigned falsespan = FalseCountMax - FalseCountMin + 1u;
    unsigned falsewant = FalseCountMin + (unsigned)(Rng() % falsespan);
    if (falsewant > falseavail) falsewant = falseavail;

    pick(constslots, truewant, Rng);
    pick(nonconstslots, falsewant, Rng);

    SmallVector<Value*, 16> plantptrs;
    SmallVector<CheckSpec, 16> checks;

    for (unsigned i = 0; i < constslots.size(); ++i)
    {
        unsigned chidx = (unsigned)plantptrs.size();
        plantptrs.push_back(constslots[i].Alloca);
        CheckSpec s;
        s.Kind = CheckSpec::True;
        s.ChannelIdx = chidx;
        s.LoadTy = constslots[i].ElemTy;
        s.ExpectedValue = constslots[i].Value;
        s.RspOffset = 0;
        checks.push_back(s);
    }
    for (unsigned i = 0; i < nonconstslots.size(); ++i)
    {
        unsigned chidx = (unsigned)plantptrs.size();
        plantptrs.push_back(nonconstslots[i]);
        CheckSpec s;
        s.Kind = CheckSpec::False;
        s.ChannelIdx = chidx;
        s.LoadTy = nonconstslots[i]->getAllocatedType();
        s.ExpectedValue = EmojiImpossible[Rng() % EmojiImpossibleCount];
        s.RspOffset = 0;
        checks.push_back(s);
    }

    LLVMContext& c = g.getContext();
    IntegerType* i64ty = Type::getInt64Ty(c);
    const unsigned rspspan = RspFalseCountMax - RspFalseCountMin + 1u;
    const unsigned rspfalsewant = RspFalseCountMin + (unsigned)(Rng() % rspspan);
    const int32_t rspstepcount = (RspOffsetMax - RspOffsetMin) / 8 + 1;
    for (unsigned i = 0; i < rspfalsewant; ++i)
    {
        CheckSpec s;
        s.Kind = CheckSpec::RspFalse;
        s.ChannelIdx = 0;
        s.LoadTy = i64ty;
        s.ExpectedValue = EmojiImpossible[Rng() % EmojiImpossibleCount];
        s.RspOffset = RspOffsetMin + (int32_t)((Rng() % (uint64_t)rspstepcount) * 8);
        checks.push_back(s);
    }

    if (checks.empty()) return false;

    // shuffle check order while ChannelIdx stays bound to each spec so the caller's plant layout is unchanged; only the emission order inside the callee gets scrambled so an attacker cannot group checks as all trues first or all RspFalses last
    for (unsigned i = 0; i < checks.size(); ++i)
    {
        unsigned j = i + (unsigned)(Rng() % (checks.size() - i));
        std::swap(checks[i], checks[j]);
    }

    GlobalVariable* channel = nullptr;
    if (!plantptrs.empty() && thecall && caller)
    {
        channel = CreateChannel(m, caller->getName().str() + "." + g.getName().str());
        EmitPlantingAtCallsite(thecall, channel, plantptrs);
    }

    if (!EmitChecksAcrossBody(g, channel, checks)) return false;

    return true;
}

bool RunStackPeekOpaque(Module& m)
{
    if (!StackPeekOpaqueEnable) return false;

    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    seed ^= 0x53544B50454B5F5FULL;
    Rng.seed(seed);

    SmallVector<Function*, 64> fns;
    for (Function& f : m) fns.push_back(&f);

    bool mutated = false;
    for (Function* f : fns)
    {
        if (ProcessCallee(*f, m)) mutated = true;
    }

    return mutated;
}

PreservedAnalyses StackPeekOpaquePass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunStackPeekOpaque(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}