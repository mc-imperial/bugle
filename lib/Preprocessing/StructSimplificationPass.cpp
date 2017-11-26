#include "bugle/Preprocessing/StructSimplificationPass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace bugle;

bool StructSimplificationPass::isGetElementPtrAllocaChain(llvm::Value *V) {
  // Check that the Value V starts a chain of GetElementPtrInsts leading to
  // an AllocaInst of a struct, where all indexes of the GetElementPtrInsts
  // are constant. When starting from a load, the latter allows us to turn
  // such a chain into a chain of ExtractValueInsts.

  if (auto *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    // Check that the first index is constant zero.
    auto *CI = dyn_cast<ConstantInt>(GEPI->getOperand(1));
    if (CI == nullptr)
      return false;

    if (!CI->isZero())
      return false;

    // Check that all other indexes are constant, as required by extract value.
    for (unsigned i = 1; i < GEPI->getNumIndices(); ++i) {
      if (!isa<ConstantInt>(GEPI->getOperand(i + 1)))
        return false;
    }

    // Recurse.
    return isGetElementPtrAllocaChain(GEPI->getPointerOperand());
  } else if (auto *AI = dyn_cast<AllocaInst>(V)) {
    return AI->getAllocatedType()->isStructTy() && !AI->isArrayAllocation();
  } else {
    return false;
  }
}

llvm::Value *
StructSimplificationPass::getExtractValueChain(llvm::Value *V,
                                               llvm::LoadInst *LI) {
  // Turn a GetElementPtrInst chain starting from the load LI, and as found by
  // isGetElementPtrAllocaChain, into a load followed by a number of
  // ExtractElemInsts.

  if (auto *AI = dyn_cast<AllocaInst>(V)) {
    auto *NewLI = new LoadInst(V, "", LI);
    NewLI->setDebugLoc(LI->getDebugLoc());
    return NewLI;
  } else if (auto *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    // Recurse.
    auto *NewV = getExtractValueChain(GEPI->getPointerOperand(), LI);

    // Get indexes. Index 0 is not used, as we are using a concrete value and
    // not a pointer in the ExtractValueInst to be created.
    assert(cast<ConstantInt>(GEPI->getOperand(1))->isZero());
    SmallVector<unsigned, 32> Idxs;
    for (unsigned i = 1; i < GEPI->getNumIndices(); ++i) {
      auto *CI = cast<ConstantInt>(GEPI->getOperand(i + 1));
      Idxs.push_back(CI->getZExtValue());
    }

    auto *EV = ExtractValueInst::Create(NewV, Idxs, "", LI);
    EV->setDebugLoc(GEPI->getDebugLoc());
    return EV;
  } else {
    llvm_unreachable("Unexpected value");
  }
}

bool StructSimplificationPass::simplifyLoads(llvm::Function &F) {
  llvm::SmallPtrSet<LoadInst *, 32> LIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (isGetElementPtrAllocaChain(LI->getPointerOperand()))
          LIs.insert(LI);

  for (auto *LI : LIs) {
    auto *V = getExtractValueChain(LI->getPointerOperand(), LI);
    LI->replaceAllUsesWith(V);

    auto *GEPI = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
    LI->eraseFromParent();
    while (GEPI != nullptr && GEPI->getNumUses() == 0) {
      auto *NextGEPI = GEPI->getPointerOperand();
      GEPI->eraseFromParent();
      GEPI = dyn_cast<GetElementPtrInst>(NextGEPI);
    }
  }

  return LIs.size() != 0;
}

bool StructSimplificationPass::runOnFunction(llvm::Function &F) {
  bool simplified = false;

  simplified |= simplifyLoads(F);

  return simplified;
}

char StructSimplificationPass::ID = 0;
