# Stack Peek Opaque Predicates

Built for LLVM 19.1 - Windows x64 <br> <br>

This yet again is another stack negative usage. Instead of using it as a way to destroy optimisers, we use the runtime nature of it to write always false opaques. <br>
This pairs well with the [opaque exceptions](https://github.com/IntelSDM/OpaqueExceptionObfuscation) as you can't just optimise away the usage of stack negative storage, as it is both actually used, but also used in multiple different conditional patterns. <br>
In hindsight it would be best to find known constants too and load them into the stack negative slots to then also have them to compare against. <br>

<img width="940" height="267" alt="image" src="https://github.com/user-attachments/assets/c6da09eb-14c4-4bff-a386-28cd26788913" /> <br>
<img width="1017" height="201" alt="image" src="https://github.com/user-attachments/assets/9b3085b8-f49c-47f9-b447-ebd28ca46214" /> <br>
<img width="1104" height="499" alt="image" src="https://github.com/user-attachments/assets/aee53ce8-b836-4eeb-b4b5-e4479453cd5c" /> <br>


## Annotations
```c
__attribute__((annotate("obfus")))
``` 
* Marks your function for the pass

## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll
