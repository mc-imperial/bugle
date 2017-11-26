#ifndef BUGLE_PREPROCESSING_STRUCTSIMPLIFICATIONPASS_H
#define BUGLE_PREPROCESSING_STRUCTSIMPLIFICATIONPASS_H

#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

namespace llvm {
template <typename T, unsigned N> class SmallVector;
}

namespace bugle {

class StructSimplificationPass : public llvm::FunctionPass {
private:
  bool isGetElementPtrAllocaChain(llvm::Value *V);
  llvm::AllocaInst *getAllocaAndIndexes(llvm::Value *V,
                                        llvm::SmallVector<unsigned, 32> &Idxs);

  void simplifySingleLoad(llvm::LoadInst *LI);
  bool simplifyLoads(llvm::Function &F);

  void simplifySingleStore(llvm::StoreInst *SI);
  bool simplifyStores(llvm::Function &F);

public:
  static char ID;

  StructSimplificationPass() : FunctionPass(ID) {}

  llvm::StringRef getPassName() const override {
    return "Simplify the handling of structs";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(llvm::Function &F) override;
};
}

#endif
