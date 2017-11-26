#ifndef BUGLE_PREPROCESSING_STRUCTSIMPLIFICATIONPASS_H
#define BUGLE_PREPROCESSING_STRUCTSIMPLIFICATIONPASS_H

#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

namespace bugle {

class StructSimplificationPass : public llvm::FunctionPass {
private:
  bool isGetElementPtrAllocaChain(llvm::Value *V);
  llvm::Value *getExtractValueChain(llvm::Value *V, llvm::LoadInst *LI);
  bool simplifyLoads(llvm::Function &F);

public:
  static char ID;

  StructSimplificationPass() : FunctionPass(ID) {}

  llvm::StringRef getPassName() const override {
    return "Simplify the handling of structs after argument promotion";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(llvm::Function &F) override;
};
}

#endif
