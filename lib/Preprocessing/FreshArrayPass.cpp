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

  unsigned i = 0;
  for (auto *CI : FreshArrayCalls) {
    auto *StorePtr = CI->getOperand(0);

    auto *BCI = dyn_cast<BitCastInst>(StorePtr);
    if (BCI != nullptr) {
      StorePtr = BCI->getOperand(0);
    }

    // Append a number to the name of the newly created function, as we may
    // create multiple instances with different types.
    std::string FName;
    llvm::raw_string_ostream FNameS(FName);
    FNameS << FreshArrayFunction->getName() << '.' << i++;

    auto *NewFreshArrayFunctionTy = FunctionType::get(
        StorePtr->getType()->getPointerElementType(), {}, false);
    auto *NewFreshArrayFunction = llvm::Function::Create(
        NewFreshArrayFunctionTy, FreshArrayFunction->getLinkage(), FNameS.str(),
        &M);

    auto *StoreVal = CallInst::Create(NewFreshArrayFunction, {}, "", CI);
    StoreVal->setDebugLoc(CI->getDebugLoc());

    auto *NewSI = new StoreInst(StoreVal, StorePtr, CI);
    NewSI->setDebugLoc(CI->getDebugLoc());
    CI->eraseFromParent();

    if (BCI != nullptr && BCI->getNumUses() == 0) {
      BCI->eraseFromParent();
    }
  }

  return FreshArrayCalls.size() != 0;
}

char FreshArrayPass::ID = 0;
