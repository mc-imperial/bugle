#include "bugle/Preprocessing/StructSimplificationPass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace bugle;

bool StructSimplificationPass::isGetElementPtrAllocaChain(llvm::Value *V) {
  // Check that the Value V starts a chain of GetElementPtrInsts leading to
  // an AllocaInst of a struct, where all indexes of the GetElementPtrInsts
  // are constant. When starting from a load/store, the latter allows us to turn
  // such a chain into a chain of ExtractValueInsts/InsertValueInsts.

  if (auto *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    // Check that the first index is constant zero.
    auto *CI = dyn_cast<ConstantInt>(GEPI->getOperand(1));
    if (CI == nullptr)
      return false;

    if (!CI->isZero())
      return false;

    // Check that all other indexes are constant.
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

llvm::AllocaInst *StructSimplificationPass::getAllocaAndIndexes(
    llvm::Value *V, llvm::SmallVector<unsigned, 32> &Idxs) {
  if (auto *AI = dyn_cast<AllocaInst>(V)) {
    return AI;
  } else if (auto *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    // Recurse.
    auto *AI = getAllocaAndIndexes(GEPI->getPointerOperand(), Idxs);

    // Get indexes. Index 0 is not used, as we are using a concrete value and
    // not a pointer in the ExtractValueInst to be created.
    assert(cast<ConstantInt>(GEPI->getOperand(1))->isZero());
    for (unsigned i = 1; i < GEPI->getNumIndices(); ++i) {
      auto *CI = cast<ConstantInt>(GEPI->getOperand(i + 1));
      Idxs.push_back(CI->getZExtValue());
    }

    return AI;
  } else {
    llvm_unreachable("Unexpected value");
  }
}

void StructSimplificationPass::simplifySingleLoad(llvm::LoadInst *LI) {
  // Turn a GetElementPtrInst chain starting from LI, and as found by
  // isGetElementPtrAllocaChain, into a load followed by a single
  // ExtractElemInst.

  SmallVector<unsigned, 32> Idxs;
  auto *GEPI = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  auto *AI = getAllocaAndIndexes(GEPI, Idxs);

  auto *NewLI = new LoadInst(AI, "", LI);
  NewLI->setDebugLoc(LI->getDebugLoc());

  auto *EV = ExtractValueInst::Create(NewLI, Idxs, "", LI);
  EV->setDebugLoc(GEPI->getDebugLoc());

  LI->replaceAllUsesWith(EV);

  LI->eraseFromParent();
  while (GEPI != nullptr && GEPI->getNumUses() == 0) {
    auto *NextGEPI = GEPI->getPointerOperand();
    GEPI->eraseFromParent();
    GEPI = dyn_cast<GetElementPtrInst>(NextGEPI);
  }
}

bool StructSimplificationPass::simplifyLoads(llvm::Function &F) {
  llvm::SmallPtrSet<LoadInst *, 32> LIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (isa<GetElementPtrInst>(LI->getPointerOperand()) &&
            isGetElementPtrAllocaChain(LI->getPointerOperand()))
          LIs.insert(LI);

  for (auto *LI : LIs)
    simplifySingleLoad(LI);

  return LIs.size() != 0;
}

void StructSimplificationPass::simplifySingleStore(llvm::StoreInst *SI) {
  // Turn a GetElementPtrInst chain starting from SI, and as found by
  // isGetElementPtrAllocaChain, into a load followed by a single
  // InsertElemInst, and a store.

  SmallVector<unsigned, 32> Idxs;
  auto *GEPI = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
  auto *AI = getAllocaAndIndexes(GEPI, Idxs);

  auto *NewLI = new LoadInst(AI, "", SI);

  auto *IV =
      InsertValueInst::Create(NewLI, SI->getValueOperand(), Idxs, "", SI);
  IV->setDebugLoc(GEPI->getDebugLoc());

  auto *NewSI = new StoreInst(IV, AI, SI);
  NewSI->setDebugLoc(SI->getDebugLoc());

  SI->eraseFromParent();
  while (GEPI != nullptr && GEPI->getNumUses() == 0) {
    auto *NextGEPI = GEPI->getPointerOperand();
    GEPI->eraseFromParent();
    GEPI = dyn_cast<GetElementPtrInst>(NextGEPI);
  }
}

bool StructSimplificationPass::simplifyStores(llvm::Function &F) {
  llvm::SmallPtrSet<StoreInst *, 32> SIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *SI = dyn_cast<StoreInst>(&I))
        if (isa<GetElementPtrInst>(SI->getPointerOperand()) &&
            isGetElementPtrAllocaChain(SI->getPointerOperand()))
          SIs.insert(SI);

  for (auto *SI : SIs)
    simplifySingleStore(SI);

  return SIs.size() != 0;
}

bool StructSimplificationPass::runOnFunction(llvm::Function &F) {
  bool simplified = false;

  simplified |= simplifyLoads(F);
  simplified |= simplifyStores(F);

  return simplified;
}

char StructSimplificationPass::ID = 0;
