#ifndef BUGLE_PREPROCESSING_CYCLEDETECTPASS_H
#define BUGLE_PREPROCESSING_CYCLEDETECTPASS_H

#include "llvm/Pass.h"
#include "llvm/Analysis/CallGraph.h"

namespace bugle {

class CycleDetectPass : public llvm::ModulePass {
public:
  static char ID;

  CycleDetectPass() : ModulePass(ID) {
#if LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR > 4)
    initializeCallGraphWrapperPassPass(*llvm::PassRegistry::getPassRegistry());
#elif LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR == 4
    initializeCallGraphPass(*llvm::PassRegistry::getPassRegistry());
#else
    initializeCallGraphAnalysisGroup(*llvm::PassRegistry::getPassRegistry());
#endif
  }

  virtual const char *getPassName() const {
    return "CallGraph cycle detection";
  }

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.setPreservesAll();
#if LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR > 4)
    AU.addRequired<llvm::CallGraphWrapperPass>();
#else
    AU.addRequired<llvm::CallGraph>();
#endif
  }

  virtual bool runOnModule(llvm::Module &M);
};

}

#endif
