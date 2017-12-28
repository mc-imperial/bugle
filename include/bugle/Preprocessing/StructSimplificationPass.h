#ifndef BUGLE_PREPROCESSING_STRUCTSIMPLIFICATIONPASS_H
#define BUGLE_PREPROCESSING_STRUCTSIMPLIFICATIONPASS_H

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"

namespace llvm {
template <typename T, unsigned N> class SmallVector;
}

namespace bugle {

class StructSimplificationPass : public llvm::FunctionPass {
private:
  llvm::DataLayout DL;

  bool isGetElementPtrBitCastAllocaChain(llvm::Value *V);
  llvm::AllocaInst *getAllocaAndIndexes(llvm::Value *V,
                                        llvm::SmallVectorImpl<unsigned> &Idxs);

  void simplifySingleLoad(llvm::LoadInst *LI);
  bool simplifyLoads(llvm::Function &F);

  void simplifySingleStore(llvm::StoreInst *SI);
  bool simplifyStores(llvm::Function &F);

  bool isBitCastAllocaOfSize(llvm::Value *MaybeAlloca, llvm::Value *Size);

  bool isAllocaMemCpyPair(llvm::Value *MaybeAlloca, llvm::Value *MaybeOther,
                          llvm::Value *Size);
  void simplifySingleMemcpy(llvm::MemCpyInst *MemCpy);
  bool simplifyMemcpys(llvm::Function &F);

  bool isAllocaMemsetOfZero(llvm::Value *MaybeAlloca, llvm::Value *MaybeZero,
                            llvm::Value *Size);
  void simplifySingleMemset(llvm::MemSetInst *MemSet);
  bool simplifyMemsets(llvm::Function &F);

public:
  static char ID;

  StructSimplificationPass(llvm::Module *M) : FunctionPass(ID), DL(M) {}

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
