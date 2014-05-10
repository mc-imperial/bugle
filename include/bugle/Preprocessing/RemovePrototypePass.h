#ifndef BUGLE_PREPROCESSING_REMOVEPROTOTYPEPASS_H
#define BUGLE_PREPROCESSING_REMOVEPROTOTYPEPASS_H

#include "llvm/Pass.h"

namespace bugle {

class RemovePrototypePass : public llvm::ModulePass {
public:
  static char ID;

  RemovePrototypePass() : ModulePass(ID) {}

  virtual const char *getPassName() const {
    return "Remove function prototypes after inlining";
  }

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const {}

  virtual bool runOnModule(llvm::Module &M);
};
}

#endif
