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

  llvm::StringRef getPassName() const override {
    return "CallGraph cycle detection";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<llvm::CallGraphWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;
};
}

#endif
