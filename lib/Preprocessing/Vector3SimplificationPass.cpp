#include "bugle/Preprocessing/Vector3SimplificationPass.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace bugle;

bool Vector3SimplificationPass::isVec4StoreOfVec3(llvm::StoreInst *SI) {
  // Look for an instruction sequence of the following form:
  //   %bc = bitcast <3 x t>* %ptr to <4 x t>*
  //   store <4 x t> %val, <4 x t>* %bc

  auto *VTy = SI->getValueOperand()->getType();
  if (!VTy->isVectorTy() || VTy->getVectorNumElements() != 4)
    return false;

  auto *BCI = dyn_cast<BitCastInst>(SI->getPointerOperand());
  if (BCI == nullptr)
    return false;

  auto *PtrTy = cast<PointerType>(BCI->getSrcTy());
  auto *ElemTy = PtrTy->getElementType();

  return ElemTy->isVectorTy() && ElemTy->getVectorNumElements() == 3;
}

void Vector3SimplificationPass::replaceVec4ByVec3Store(llvm::Function &F,
                                                       llvm::StoreInst *SI) {
  // Create an instruction sequence of the following form:
  //   %shuffle = shufflevector <4 x t> %val, <4 x t> undef,
  //                            <3 x i32> <i32 0, i32 1, i32 2>
  //   store <3 x t> %shuffle, <3 x t>* %ptr

  uint32_t MaskVal[] = {0, 1, 2};
  auto *VOp = SI->getValueOperand();
  auto *SVI = new ShuffleVectorInst(
      VOp, UndefValue::get(VOp->getType()),
      ConstantDataVector::get(F.getContext(), MaskVal), "", SI);

  auto *BCI = cast<BitCastInst>(SI->getPointerOperand());
  auto *NewSI = new StoreInst(SVI, BCI->getOperand(0), SI);
  NewSI->setDebugLoc(SI->getDebugLoc());

  // Erase the old store and associated bitcast.
  SI->eraseFromParent();
  if (BCI->getNumUses() == 0)
    BCI->eraseFromParent();
}

bool Vector3SimplificationPass::simplifyStores(llvm::Function &F) {
  llvm::SmallPtrSet<StoreInst *, 32> SIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *SI = dyn_cast<StoreInst>(&I))
        if (isVec4StoreOfVec3(SI))
          SIs.insert(SI);

  for (auto *SI : SIs)
    replaceVec4ByVec3Store(F, SI);

  return SIs.size() != 0;
}

bool Vector3SimplificationPass::isVec4LoadOfVec3(llvm::LoadInst *LI) {
  // Look for an instruction sequence of the following form:
  //   %bc = bitcast <3 x t>* %ptr to <4 x t>*
  //   %val = load <4 x t>, <4 x t>* %bc

  auto *LITy = LI->getType();
  if (!LITy->isVectorTy() || LITy->getVectorNumElements() != 4)
    return false;

  auto *BCI = dyn_cast<BitCastInst>(LI->getPointerOperand());
  if (BCI == nullptr)
    return false;

  auto *ElemTy = cast<PointerType>(BCI->getSrcTy())->getElementType();

  return ElemTy->isVectorTy() && ElemTy->getVectorNumElements() == 3;
}

void Vector3SimplificationPass::replaceVec4ByVec3Load(llvm::Function &F,
                                                      llvm::LoadInst *LI) {
  // Create an instruction sequence of the following form:
  //   %val3 = load <3 x t>, <3 x t>* %ptr
  //   %val  = shufflevector <3 x t> %val3, <3 x t> undef,
  //                         <4 x i32> <i32 0, i32 1, i32 2, i32 3>

  auto *BCI = cast<BitCastInst>(LI->getPointerOperand());
  auto *NewLI = new LoadInst(BCI->getOperand(0), "", LI);
  NewLI->setDebugLoc(LI->getDebugLoc());

  uint32_t MaskVal[] = {0, 1, 2, 3};
  auto *SVI = new ShuffleVectorInst(
      NewLI, UndefValue::get(NewLI->getType()),
      ConstantDataVector::get(F.getContext(), MaskVal), "", LI);

  // Replace the uses of the old load.
  LI->replaceAllUsesWith(SVI);

  // Erase the old load and associated bitcast.
  LI->eraseFromParent();
  if (BCI->getNumUses() == 0)
    BCI->eraseFromParent();
}

bool Vector3SimplificationPass::simplifyLoads(llvm::Function &F) {
  llvm::SmallPtrSet<LoadInst *, 32> LIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (isVec4LoadOfVec3(LI))
          LIs.insert(LI);

  for (auto *LI : LIs)
    replaceVec4ByVec3Load(F, LI);

  return LIs.size() != 0;
}

bool Vector3SimplificationPass::isVec3ShufflePair(
    llvm::Function &F, llvm::ShuffleVectorInst *SVI) {
  // Look for an instruction sequence of the following form:
  //   %val4 = shufflevector <3 x t> %val, <3 x t> undef,
  //                          <4 x i32> <i32 0, i32 1, i32 2, ?>
  //   %val3 = shufflevector <4 x t> %val4, <4 x t> undef,
  //                         <3 x i32> <i32 0, i32 1, i32 2>
  // Half of such a sequence is generally created by the load and store
  // simplifications, while the other half is usually already present to
  // support the loads and stores of vectors of size 4.

  if (SVI->getType()->getVectorNumElements() != 3)
    return false;

  if (SVI->getOperand(0)->getType()->getVectorNumElements() != 4)
    return false;

  if (!isa<UndefValue>(SVI->getOperand(1)))
    return false;

  uint32_t MaskVal[] = {0, 1, 2};
  if (ConstantDataVector::get(F.getContext(), MaskVal) != SVI->getMask())
    return false;

  auto *NestedSVI = dyn_cast<ShuffleVectorInst>(SVI->getOperand(0));
  if (NestedSVI == nullptr)
    return false;

  assert(NestedSVI->getType()->getVectorNumElements() == 4);

  if (NestedSVI->getOperand(0)->getType()->getVectorNumElements() != 3)
    return false;

  if (!isa<UndefValue>(NestedSVI->getOperand(1)))
    return false;

  if (auto *Mask = dyn_cast<ConstantVector>(NestedSVI->getMask())) {
    for (unsigned i = 0; i < 3; ++i) {
      if (Mask->getOperand(i) !=
          ConstantInt::get(Mask->getType()->getVectorElementType(), i))
        return false;
    }
    return true;
  } else {
    uint32_t MaskVal[] = {0, 1, 2, 3};
    return ConstantDataVector::get(F.getContext(), MaskVal) ==
           NestedSVI->getMask();
  }
}

void Vector3SimplificationPass::eraseShufflePair(llvm::ShuffleVectorInst *SVI) {
  auto *NestedSVI = cast<ShuffleVectorInst>(SVI->getOperand(0));

  // Replace all usages of SVI with the 0th operand of the nested shuffle.
  SVI->replaceAllUsesWith(NestedSVI->getOperand(0));

  if (SVI->getNumUses() == 0)
    SVI->eraseFromParent();
  if (NestedSVI->getNumUses() == 0)
    NestedSVI->eraseFromParent();
}

bool Vector3SimplificationPass::simplifyShufflePairs(llvm::Function &F) {
  llvm::SmallPtrSet<ShuffleVectorInst *, 32> SVIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *SVI = dyn_cast<ShuffleVectorInst>(&I))
        if (isVec3ShufflePair(F, SVI))
          SVIs.insert(SVI);

  for (auto *SVI : SVIs)
    eraseShufflePair(SVI);

  return SVIs.size() != 0;
}

bool Vector3SimplificationPass::runOnFunction(llvm::Function &F) {
  bool simplified = false;

  simplified |= simplifyStores(F);
  simplified |= simplifyLoads(F);
  simplified |= simplifyShufflePairs(F);

  return simplified;
}

char Vector3SimplificationPass::ID = 0;
