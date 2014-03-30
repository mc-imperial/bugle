#include "bugle/Preprocessing/RemovePrototypePass.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace bugle;

bool RemovePrototypePass::runOnModule(llvm::Module &M) {
  bool change = false;

  for (auto i = M.begin(), e = M.end(); i != e;) {
    // The iterator needs to be incremented before we remove F, otherwise
    // it will point to an invalid value afterwards.
    Function *F = i++;
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      change = true;
    }
  }

  return change;
}

char RemovePrototypePass::ID = 0;
