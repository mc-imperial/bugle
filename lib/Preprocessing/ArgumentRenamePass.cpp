#include "bugle/Preprocessing/ArgumentRenamePass.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace bugle;

bool ArgumentRenamePass::runOnFunction(llvm::Function &F) {
  for (auto &A : F.args()) {
    A.setName("arg." + A.getName());
  }

  return true;
}

char ArgumentRenamePass::ID = 0;
