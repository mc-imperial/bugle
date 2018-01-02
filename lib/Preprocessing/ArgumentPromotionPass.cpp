#include "bugle/Preprocessing/ArgumentPromotionPass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/Pass.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include <set>
#include <vector>

using namespace llvm;
using namespace bugle;

bool ArgumentPromotionPass::needsPromotion(llvm::Function *F) {
  // Only apply promotion to normal functions.
  if (TranslateFunction::isNormalFunction(SL, F))
    for (auto &Arg : F->args())
      if (Arg.hasByValAttr())
        return true;

  return false;
}

bool ArgumentPromotionPass::canPromote(llvm::Function *F) {
  for (auto *User : F->users()) {
    CallSite CS(User);
    if (!CS.getInstruction())
      return false;
  }

  return true;
}

bool ArgumentPromotionPass::usesFunctionPointers(llvm::Function *F) {
  for (auto &BB : *F)
    for (auto &I : BB) {
      auto CI = dyn_cast<CallInst>(&I);

      if (!CI)
        continue;

      if (!CI->getCalledFunction())
        return true;
    }

  return false;
}

llvm::Function *ArgumentPromotionPass::createNewFunction(llvm::Function *F) {
  auto *FTy = F->getFunctionType();
  const auto &FAL = F->getAttributes();
  std::vector<llvm::Type *> NewArgs;
  std::vector<AttributeSet> NewAttributes;

  // Promote call by value arguments; copy other arguments including attributes
  unsigned ArgNo = 0;
  for (auto i = F->arg_begin(), e = F->arg_end(); i != e; ++i, ++ArgNo) {
    if (i->hasByValAttr()) {
      NewArgs.push_back(cast<PointerType>(i->getType())->getElementType());
      NewAttributes.push_back(AttributeSet());
    } else {
      NewArgs.push_back(i->getType());
      NewAttributes.push_back(FAL.getParamAttributes(ArgNo));
    }
  }

  auto *RetTy = FTy->getReturnType();
  auto *NFty = FunctionType::get(RetTy, NewArgs, FTy->isVarArg());
  auto *NF = llvm::Function::Create(NFty, F->getLinkage(), F->getName());
  NF->copyAttributesFrom(F);
  NF->setAttributes(AttributeList::get(F->getContext(), FAL.getFnAttributes(),
                                       FAL.getRetAttributes(), NewAttributes));

  return NF;
}

void ArgumentPromotionPass::updateCallSite(CallSite *CS, llvm::Function *F,
                                           llvm::Function *NF) {
  auto *CI = CS->getInstruction();
  const auto &CAL = CS->getAttributes();
  std::vector<Value *> NewArgs;
  std::vector<AttributeSet> NewAttributes;

  if (!dyn_cast<CallInst>(CI)) {
    ErrorReporter::reportImplementationLimitation(
        "Only call instructions supported as call sites");
  }

  // Create load instruction for each promoted argument and keep track of the
  // attributes from every other argument
  unsigned ArgNo = 0;
  for (auto i = CS->arg_begin(), e = CS->arg_end(); i != e; ++i, ++ArgNo) {
    if (CS->isByValArgument(ArgNo)) {
      NewArgs.push_back(new LoadInst(*i, (*i)->getName() + ".val", CI));
      NewAttributes.push_back(AttributeSet());
    } else {
      NewArgs.push_back(*i);
      NewAttributes.push_back(CAL.getParamAttributes(ArgNo));
    }
  }

  auto *NC = CallInst::Create(NF, NewArgs, "", CI);
  NC->setCallingConv(CS->getCallingConv());
  NC->setAttributes(AttributeList::get(F->getContext(), CAL.getFnAttributes(),
                                       CAL.getRetAttributes(), NewAttributes));
  NC->setDebugLoc(CI->getDebugLoc());
  // We do not copy the tail call attribute; we do not need it and it would
  // need to be removed by spliceBody in case any alloca created in spliceBody
  // is used in a subsequent call

  if (!CI->use_empty()) {
    CI->replaceAllUsesWith(NC);
    NC->takeName(CI);
  }
  CI->eraseFromParent();
}

void ArgumentPromotionPass::replaceMetaData(llvm::Function *F,
                                            llvm::Function *NF, MDNode *MD,
                                            std::set<llvm::MDNode *> &doneMD) {
  if (doneMD.find(MD) != doneMD.end())
    return;

  doneMD.insert(MD);

  for (unsigned i = 0, e = MD->getNumOperands(); i < e; ++i) {
    if (!MD->getOperand(i))
      continue;
    else if (MD->getOperand(i) == ValueAsMetadata::get(F))
      MD->replaceOperandWith(i, ValueAsMetadata::get(NF));
    else if (auto MDV = dyn_cast<MDNode>(MD->getOperand(i)))
      replaceMetaData(F, NF, MDV, doneMD);
  }
}

void ArgumentPromotionPass::spliceBody(llvm::Function *F, llvm::Function *NF) {
  DIBuilder DIB(*M);
  const DataLayout &DL = F->getParent()->getDataLayout();

  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  for (auto i = F->arg_begin(), e = F->arg_end(), ni = NF->arg_begin(); i != e;
       ++i, ++ni) {
    Value *AI = &*ni;
    AI->takeName(&*i);

    // The splice function does not update the relevant debug intrinsics. Update
    // these manually.
    llvm::SmallPtrSet<DbgValueInst *, 32> DVIs;
    llvm::SmallPtrSet<DbgDeclareInst *, 32> DDIs;
    for (auto &BB : *NF) {
      for (auto &I : BB) {
        if (auto *DVI = dyn_cast<DbgValueInst>(&I))
          if (DVI->getValue() == &*i)
            DVIs.insert(DVI);
        if (auto *DDI = dyn_cast<DbgDeclareInst>(&I))
          if (DDI->getAddress() == &*i)
            DDIs.insert(DDI);
      }
    }

    for (auto *DVI : DVIs) {
      DIB.insertDbgValueIntrinsic(AI, DVI->getVariable(), DVI->getExpression(),
                                  DVI->getDebugLoc(), DVI);
      DVI->eraseFromParent();
    }

    for (auto *DDI : DDIs) {
      DIB.insertDbgValueIntrinsic(AI, DDI->getVariable(), DDI->getExpression(),
                                  DDI->getDebugLoc(), DDI);
      DDI->eraseFromParent();
    }

    // Copy the value of the function argument into locally allocated space
    if (i->hasByValAttr()) {
      auto *InsertPoint = &*NF->begin()->begin();
      auto *ArgTy = cast<PointerType>(i->getType())->getElementType();
      AI = new AllocaInst(ArgTy, DL.getAllocaAddrSpace(),
                          ni->getName() + ".val", InsertPoint);
      new StoreInst(&*ni, AI, InsertPoint);
    }

    i->replaceAllUsesWith(AI);
  }
}

void ArgumentPromotionPass::promote(llvm::Function *F) {
  auto *NF = createNewFunction(F);

  // Update debug information to point to new function
  NF->setSubprogram(F->getSubprogram());
  F->setSubprogram(nullptr);

  // Replace the function in the remaining (non-debug) meta-data
  std::set<MDNode *> doneMD;
  for (auto &NMD : M->getNamedMDList()) {
    for (auto *MD : NMD.operands()) {
      replaceMetaData(F, NF, MD, doneMD);
    }
  }

  // Insert new function and take F's name
  F->getParent()->getFunctionList().insert(F->getIterator(), NF);
  NF->takeName(F);

  while (!F->use_empty()) {
    CallSite CS(F->user_back());
    updateCallSite(&CS, F, NF);
  }

  if (!F->isDeclaration()) {
    spliceBody(F, NF);
  }

  // The remains of F will be removed by a dead-code elimination pass
}

bool ArgumentPromotionPass::runOnModule(llvm::Module &M) {
  bool promoted = false;
  this->M = &M;

  for (auto &F : M) {
    if (usesFunctionPointers(&F)) {
      return false;
    }
  }

  for (auto &F : M) {
    if (needsPromotion(&F) && canPromote(&F)) {
      promote(&F);
      promoted = true;
    }
  }

  return promoted;
}

char ArgumentPromotionPass::ID = 0;
