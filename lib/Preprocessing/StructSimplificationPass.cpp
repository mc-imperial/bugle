#include "bugle/Preprocessing/StructSimplificationPass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace bugle;

bool StructSimplificationPass::isGetElementPtrBitCastAllocaChain(
    llvm::Value *V) {
  // Check that the Value V starts a chain of GetElementPtrInsts/BitCastInts
  // leading to an AllocaInst of a struct, where (a) all indexes of the
  // GetElementPtrInsts are constant and (b) all BitCastInts take a pointer to a
  // struct and cast the pointer to a pointer of the type of what is the nested
  // first element of the struct. When starting from a load/store, the latter
  // allows us to turn such a chain into a ExtractValueInst/InsertValueInst.

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
    return isGetElementPtrBitCastAllocaChain(GEPI->getPointerOperand());
  } else if (auto *BCI = dyn_cast<BitCastInst>(V)) {
    auto *OperandType = BCI->getOperand(0)->getType()->getPointerElementType();
    auto *ResultType = BCI->getType()->getPointerElementType();

    do {
      if (!OperandType->isStructTy())
        return false;

      OperandType = OperandType->getStructElementType(0);
    } while(OperandType != ResultType);

    // Recurse.
    return isGetElementPtrBitCastAllocaChain(BCI->getOperand(0));
  } else if (auto *AI = dyn_cast<AllocaInst>(V)) {
    return AI->getAllocatedType()->isStructTy() && !AI->isArrayAllocation();
  } else {
    return false;
  }
}

llvm::AllocaInst *StructSimplificationPass::getAllocaAndIndexes(
    llvm::Value *V, llvm::SmallVectorImpl<unsigned> &Idxs) {
  if (auto *AI = dyn_cast<AllocaInst>(V)) {
    return AI;
  } else if (auto *BCI = dyn_cast<BitCastInst>(V)) {
    // Recurse.
    auto *AI = getAllocaAndIndexes(BCI->getOperand(0), Idxs);

    auto *OperandType = BCI->getOperand(0)->getType()->getPointerElementType();
    auto *ResultType = BCI->getType()->getPointerElementType();

    do {
      Idxs.push_back(0);
      OperandType = OperandType->getStructElementType(0);
    } while(OperandType != ResultType);

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
  // Turn a GetElementPtrInst/BitCastInst chain starting from LI, and as found
  // by isGetElementPtrBitCastAllocaChain, into a load followed by a single
  // ExtractElemInst.

  SmallVector<unsigned, 32> Idxs;
  auto *I = dyn_cast<Instruction>(LI->getPointerOperand());
  auto *AI = getAllocaAndIndexes(I, Idxs);

  if (Idxs.size() == 0)
    return;

  auto *NewLI = new LoadInst(AI, "", LI);
  NewLI->setDebugLoc(LI->getDebugLoc());

  auto *EV = ExtractValueInst::Create(NewLI, Idxs, "", LI);
  EV->setDebugLoc(I->getDebugLoc());

  LI->replaceAllUsesWith(EV);

  LI->eraseFromParent();
  while (I != nullptr && I->getNumUses() == 0) {
    auto *NextI = I->getOperand(0);
    I->eraseFromParent();
    I = dyn_cast<Instruction>(NextI);
  }
}

bool StructSimplificationPass::simplifyLoads(llvm::Function &F) {
  llvm::SmallPtrSet<LoadInst *, 32> LIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (isGetElementPtrBitCastAllocaChain(LI->getPointerOperand()))
          LIs.insert(LI);

  for (auto *LI : LIs)
    simplifySingleLoad(LI);

  return LIs.size() != 0;
}

void StructSimplificationPass::simplifySingleStore(llvm::StoreInst *SI) {
  // Turn a GetElementPtrInst/BitCastInst chain starting from SI, and as found
  // by isGetElementPtrBitCastAllocaChain, into a load followed by a single
  // InsertElemInst, and a store.

  SmallVector<unsigned, 32> Idxs;
  auto *I = dyn_cast<Instruction>(SI->getPointerOperand());
  auto *AI = getAllocaAndIndexes(I, Idxs);

  if (Idxs.size() == 0)
    return;

  auto *NewLI = new LoadInst(AI, "", SI);

  auto *IV =
      InsertValueInst::Create(NewLI, SI->getValueOperand(), Idxs, "", SI);
  IV->setDebugLoc(I->getDebugLoc());

  auto *NewSI = new StoreInst(IV, AI, SI);
  NewSI->setDebugLoc(SI->getDebugLoc());

  SI->eraseFromParent();
  while (I != nullptr && I->getNumUses() == 0) {
    auto *NextI = I->getOperand(0);
    I->eraseFromParent();
    I = dyn_cast<Instruction>(NextI);
  }
}

bool StructSimplificationPass::simplifyStores(llvm::Function &F) {
  llvm::SmallPtrSet<StoreInst *, 32> SIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *SI = dyn_cast<StoreInst>(&I))
        if (isGetElementPtrBitCastAllocaChain(SI->getPointerOperand()))
          SIs.insert(SI);

  for (auto *SI : SIs)
    simplifySingleStore(SI);

  return SIs.size() != 0;
}

bool StructSimplificationPass::isBitCastAllocaOfSize(llvm::Value *MaybeAlloca,
                                                     llvm::Value *Size) {
  // Check whether the MaybeAlloca parameter is a bitcast of the result of an
  // alloca of a struct whose size is Size.

  if (MaybeAlloca->getType()->getPointerElementType() !=
      Type::getInt8Ty(MaybeAlloca->getContext()))
    return false;

  // Check the MaybeAlloca parameter.
  auto *AllocaBCI = dyn_cast<BitCastInst>(MaybeAlloca);
  if (AllocaBCI == nullptr)
    return false;

  auto *AI = dyn_cast<AllocaInst>(AllocaBCI->getOperand(0));
  if (AI == nullptr)
    return false;

  if (!AI->getAllocatedType()->isStructTy() || AI->isArrayAllocation())
    return false;

  auto *CSize = dyn_cast<ConstantInt>(Size);
  if (!CSize)
    return false;

  return DL.getTypeAllocSize(AI->getAllocatedType()) == CSize->getZExtValue();
}

bool StructSimplificationPass::isAllocaMemCpyPair(llvm::Value *MaybeAlloca,
                                                  llvm::Value *MaybeOther,
                                                  llvm::Value *Size) {
  // Check whether the MaybeAlloca parameter is a bitcast of the result of an
  // alloca of a struct whose size is Size, and check whether the MaybeOther
  // parameter is a pointer to a struct of the same type as the allocated one
  // after application of either (a) a bitcast, or (b) a getelementptr to the
  // first element of the struct whose type is i8, and either of these possibly
  // followed by an address space cast.

  if (!isBitCastAllocaOfSize(MaybeAlloca, Size))
    return false;

  auto *AllocaBCI = cast<BitCastInst>(MaybeAlloca);
  auto *AI = cast<AllocaInst>(AllocaBCI->getOperand(0));

  if (MaybeOther->getType()->getPointerElementType() !=
      Type::getInt8Ty(MaybeOther->getContext()))
    return false;

  // Check the MaybeOther parameter. The parameter can either be an instruction
  // or a constant, which require different checks.
  if (isa<Instruction>(MaybeOther)) {
    Value *Other = isa<AddrSpaceCastInst>(MaybeOther)
                       ? cast<AddrSpaceCastInst>(MaybeOther)->getOperand(0)
                       : MaybeOther;
    if (auto *GEPI = dyn_cast<GetElementPtrInst>(Other)) {
      for (unsigned i = 1; i < GEPI->getNumOperands(); ++i) {
        auto *Op = dyn_cast<ConstantInt>(GEPI->getOperand(i));

        if (Op == nullptr || !Op->isZero())
          return false;
      }

      Other = GEPI->getPointerOperand();
    } else if (auto *BCI = dyn_cast<BitCastInst>(Other)) {
      Other = BCI->getOperand(0);
    }

    return AI->getAllocatedType() == Other->getType()->getPointerElementType();
  } else if (auto *CE = dyn_cast<ConstantExpr>(MaybeOther)) {
    Value *Other =
        CE->getOpcode() == Instruction::AddrSpaceCast ? CE->getOperand(0) : CE;

    if (auto *OtherCE = dyn_cast<ConstantExpr>(Other)) {
      if (OtherCE->getOpcode() == Instruction::GetElementPtr) {
        for (unsigned i = 1; i < OtherCE->getNumOperands(); ++i) {
          auto *Op = dyn_cast<ConstantInt>(OtherCE->getOperand(i));

          if (Op == nullptr || !Op->isZero())
            return false;
        }

        Other = OtherCE->getOperand(0);
      } else if (OtherCE->getOpcode() == Instruction::BitCast) {
        Other = OtherCE->getOperand(0);
      }
    }

    return AI->getAllocatedType() == Other->getType()->getPointerElementType();
  } else {
    return false;
  }
}

void StructSimplificationPass::simplifySingleMemcpy(llvm::MemCpyInst *MemCpy) {
  // Simplify a memcpy, whose arguments satisfy the conditions of the test
  // performed by isAllocaMemCpyPair, by replacing the memcpy by a sequence
  // consisting of a load and a store.

  bool DestIsAlloca = isAllocaMemCpyPair(
      MemCpy->getOperand(0), MemCpy->getOperand(1), MemCpy->getOperand(2));

  auto *AllocaBCI = DestIsAlloca ? cast<BitCastInst>(MemCpy->getOperand(0))
                                 : cast<BitCastInst>(MemCpy->getOperand(1));
  auto *AI = cast<AllocaInst>(AllocaBCI->getOperand(0));

  auto *Other = DestIsAlloca ? MemCpy->getOperand(1) : MemCpy->getOperand(0);
  Value *OtherOp = Other;

  if (isa<Instruction>(OtherOp)) {
    OtherOp = isa<AddrSpaceCastInst>(OtherOp)
                  ? cast<AddrSpaceCastInst>(OtherOp)->getOperand(0)
                  : OtherOp;

    if (auto *GEPI = dyn_cast<GetElementPtrInst>(OtherOp)) {
      OtherOp = GEPI->getPointerOperand();
    } else if (auto *BCI = dyn_cast<BitCastInst>(OtherOp)) {
      OtherOp = BCI->getOperand(0);
    }
  } else if (auto *CE = dyn_cast<ConstantExpr>(OtherOp)) {
    OtherOp = CE->getOpcode() == Instruction::AddrSpaceCast
                  ? cast<ConstantExpr>(CE->getOperand(0))
                  : CE;

    if (auto *OtherCE = dyn_cast<ConstantExpr>(OtherOp)) {
      if (OtherCE->getOpcode() == Instruction::GetElementPtr ||
          OtherCE->getOpcode() == Instruction::BitCast) {
        OtherOp = OtherCE->getOperand(0);
      }
    }
  }

  auto *NewLI = DestIsAlloca ? new LoadInst(OtherOp, "", MemCpy)
                             : new LoadInst(AI, "", MemCpy);
  NewLI->setDebugLoc(MemCpy->getDebugLoc());

  auto *NewSI = DestIsAlloca ? new StoreInst(NewLI, AI, MemCpy)
                             : new StoreInst(NewLI, OtherOp, MemCpy);
  NewSI->setDebugLoc(MemCpy->getDebugLoc());

  MemCpy->eraseFromParent();
  if (isa<Instruction>(Other) && Other->getNumUses() == 0) {
    auto *ASCI = dyn_cast<AddrSpaceCastInst>(Other);
    if (ASCI != nullptr) {
      Other = ASCI->getOperand(0);
      ASCI->eraseFromParent();
    }
    cast<Instruction>(Other)->eraseFromParent();
  }

  if (AllocaBCI->getNumUses() == 0)
    AllocaBCI->eraseFromParent();
}

bool StructSimplificationPass::simplifyMemcpys(llvm::Function &F) {
  llvm::SmallPtrSet<MemCpyInst *, 32> MemCpys;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *MemCpy = dyn_cast<MemCpyInst>(&I))
        if (isAllocaMemCpyPair(MemCpy->getOperand(0), MemCpy->getOperand(1),
                               MemCpy->getOperand(2)) ||
            isAllocaMemCpyPair(MemCpy->getOperand(1), MemCpy->getOperand(0),
                               MemCpy->getOperand(2)))
          MemCpys.insert(MemCpy);

  for (auto *MemCpy : MemCpys)
    simplifySingleMemcpy(MemCpy);

  return MemCpys.size() != 0;
}

bool StructSimplificationPass::isAllocaMemsetOfZero(llvm::Value *MaybeAlloca,
                                                    llvm::Value *MaybeZero,
                                                    llvm::Value *Size) {
  // Check whether the MaybeAlloca parameter is a bitcast of the result of an
  // alloca of a struct whose size is Size, and check whether the MaybeZero
  // parameter is a constant zero value.

  if (!isBitCastAllocaOfSize(MaybeAlloca, Size))
    return false;

  auto *CI = dyn_cast<ConstantInt>(MaybeZero);
  return CI != nullptr && CI->getZExtValue() == 0;
}

void StructSimplificationPass::simplifySingleMemset(llvm::MemSetInst *MemSet) {
  // Simplify a memset, whose arguments satisfy the conditions of the test
  // performed by isAllocaMemsetOfZero, by replacing the memset by a store of
  // a zero value.

  auto *AllocaBCI = cast<BitCastInst>(MemSet->getOperand(0));
  auto *AI = cast<AllocaInst>(AllocaBCI->getOperand(0));

  auto *Zero = ConstantAggregateZero::get(AI->getAllocatedType());
  auto *NewSI = new StoreInst(Zero, AI, MemSet);
  NewSI->setDebugLoc(MemSet->getDebugLoc());

  MemSet->eraseFromParent();
  if (AllocaBCI->getNumUses() == 0)
    AllocaBCI->eraseFromParent();
}

bool StructSimplificationPass::simplifyMemsets(llvm::Function &F) {
  llvm::SmallPtrSet<MemSetInst *, 32> MemSets;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *MemSet = dyn_cast<MemSetInst>(&I))
        if (isAllocaMemsetOfZero(MemSet->getOperand(0), MemSet->getOperand(1),
                                 MemSet->getOperand(2)))
          MemSets.insert(MemSet);

  for (auto *MemSet : MemSets)
    simplifySingleMemset(MemSet);

  return MemSets.size() != 0;
}

bool StructSimplificationPass::runOnFunction(llvm::Function &F) {
  bool simplified = false;

  simplified |= simplifyLoads(F);
  simplified |= simplifyStores(F);
  simplified |= simplifyMemcpys(F);
  simplified |= simplifyMemsets(F);

  return simplified;
}

char StructSimplificationPass::ID = 0;
