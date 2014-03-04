#ifndef BUGLE_PREPROCESSING_CYCLEDETECTPASS_H
#define BUGLE_PREPROCESSING_CYCLEDETECTPASS_H

#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"

namespace bugle {

class CycleDetectPass : public llvm::ModulePass {
public:
  static char ID;

  CycleDetectPass() : ModulePass(ID) {
    initializeCallGraphWrapperPassPass(*llvm::PassRegistry::getPassRegistry());
  }

  virtual const char *getPassName() const {
    return "CallGraph cycle detection";
  }

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<llvm::CallGraphWrapperPass>();
  }

  virtual bool runOnModule(llvm::Module &M);
};

}

#endif
