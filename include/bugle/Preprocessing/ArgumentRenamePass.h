#ifndef BUGLE_PREPROCESSING_ARGUMENTRENAMEPASS_H
#define BUGLE_PREPROCESSING_ARGUMENTRENAMEPASS_H

#include "llvm/Pass.h"

namespace bugle {

// Compiler pass to prefix each function argument with "arg.". This ensures
// no number-only function arguments occur during the translation to Boogie.
// The number-only arguments 0 and 1 would cause problems down the line as they
// are translated into variables ending in $0 and $1, which the endings also
// added during dualisation in GPUVerifyVCGen.

class ArgumentRenamePass : public llvm::FunctionPass {
public:
  static char ID;

  ArgumentRenamePass() : FunctionPass(ID) {}

  llvm::StringRef getPassName() const override {
    return "Rename function arguments";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnFunction(llvm::Function &F) override;
};
}

#endif
