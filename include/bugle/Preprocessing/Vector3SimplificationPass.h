#ifndef BUGLE_PREPROCESSING_VECTOR3SIMPLIFICATIONPASS_H
#define BUGLE_PREPROCESSING_VECTOR3SIMPLIFICATIONPASS_H

#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

namespace bugle {

class Vector3SimplificationPass : public llvm::FunctionPass {
private:
  bool isVec4StoreOfVec3(llvm::StoreInst *SI);
  void replaceVec4ByVec3Store(llvm::Function &F, llvm::StoreInst *SI);
  bool simplifyStores(llvm::Function &F);

  bool isVec4LoadOfVec3(llvm::LoadInst *LI);
  void replaceVec4ByVec3Load(llvm::Function &F, llvm::LoadInst *LI);
  bool simplifyLoads(llvm::Function &F);

  bool isVec3ShufflePair(llvm::Function &F, llvm::ShuffleVectorInst *SVI);
  void eraseShufflePair(llvm::ShuffleVectorInst *SVI);
  bool simplifyShufflePairs(llvm::Function &F);

public:
  static char ID;

  Vector3SimplificationPass() : FunctionPass(ID) {}

  llvm::StringRef getPassName() const override {
    return "Simplify loads and stores of vectors of size 3";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(llvm::Function &F) override;
};
}

#endif
