#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct StackPeekOpaquePass : llvm::PassInfoMixin<StackPeekOpaquePass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunStackPeekOpaque(llvm::Module& m);

extern llvm::cl::opt<bool> StackPeekOpaqueEnable;