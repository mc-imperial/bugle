#include "bugle/Preprocessing/FreshArrayPass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace bugle;

bool FreshArrayPass::runOnModule(llvm::Module &M) {
  llvm::Function *FreshArrayFunction = nullptr;

  for (auto &F : M) {
    if (TranslateFunction::isRequiresFreshArrayFunction(F.getName())) {
      FreshArrayFunction = &F;
      break;
    }
  }

  if (FreshArrayFunction == nullptr)
    return false;

  llvm::SmallVector<CallInst *, 32> FreshArrayCalls;

  for (auto *U : FreshArrayFunction->users()) {
    if (auto *CI = dyn_cast<CallInst>(U))
      FreshArrayCalls.push_back(CI);
  }

  // Replace each call to __requires_fresh_array by a store to its pointer
  // argument, where the value stored is produced by a fresh function named
  // __requires_fresh_array.n for some integer n.
  unsigned FunctionCount = 0;
  for (auto *CI : FreshArrayCalls) {
    auto *PtrToStoreTo = CI->getOperand(0);

    auto *BCI = dyn_cast<BitCastInst>(PtrToStoreTo);
    if (BCI != nullptr) {
      PtrToStoreTo = BCI->getOperand(0);
    }

    // Append a number to the name of the newly created function, as we may
    // create multiple instances with different types.
    std::string FName;
    llvm::raw_string_ostream FNameS(FName);
    FNameS << FreshArrayFunction->getName() << '.' << FunctionCount++;

    auto *NewFreshArrayFunctionTy = FunctionType::get(
        PtrToStoreTo->getType()->getPointerElementType(), {}, false);
    auto *NewFreshArrayFunction = llvm::Function::Create(
        NewFreshArrayFunctionTy, FreshArrayFunction->getLinkage(), FNameS.str(),
        &M);

    auto *ValToStore = CallInst::Create(NewFreshArrayFunction, {}, "", CI);
    ValToStore->setDebugLoc(CI->getDebugLoc());

    auto *NewSI = new StoreInst(ValToStore, PtrToStoreTo, CI);
    NewSI->setDebugLoc(CI->getDebugLoc());
    CI->eraseFromParent();

    if (BCI != nullptr && BCI->getNumUses() == 0) {
      BCI->eraseFromParent();
    }
  }

  return FreshArrayCalls.size() != 0;
}

char FreshArrayPass::ID = 0;
