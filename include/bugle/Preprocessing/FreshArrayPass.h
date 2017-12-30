#ifndef BUGLE_PREPROCESSING_FRESHARRAYPASS_H
#define BUGLE_PREPROCESSING_FRESHARRAYPASS_H

#include "llvm/Pass.h"

namespace bugle {

class FreshArrayPass : public llvm::ModulePass {
public:
  static char ID;

  FreshArrayPass() : ModulePass(ID) {}

  llvm::StringRef getPassName() const override {
    return "Transform fresh array calls into assignments";
  }

  bool runOnModule(llvm::Module &M) override;
};
}

#endif
