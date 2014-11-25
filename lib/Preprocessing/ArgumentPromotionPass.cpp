#include "bugle/Preprocessing/ArgumentPromotionPass.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/Pass.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <vector>

using namespace llvm;
using namespace bugle;

bool ArgumentPromotionPass::needsPromotion(llvm::Function *F) {
  // Only apply promotion to normal functions.
  if (TranslateFunction::isNormalFunction(SL, F))
    for (auto i = F->arg_begin(), e = F->arg_end(); i != e; ++i)
      if (i->hasByValAttr())
        return true;

  return false;
}

llvm::Function *ArgumentPromotionPass::createNewFunction(llvm::Function *F) {
  FunctionType *FTy = F->getFunctionType();
  const AttributeSet &FAS = F->getAttributes();
  LLVMContext &FC = F->getContext();
  std::vector<llvm::Type *> NewArgs;
  std::vector<AttributeSet> NewAttributes;

  if (FAS.hasAttributes(AttributeSet::ReturnIndex)) {
    AttributeSet RA = AttributeSet::get(FC, FAS.getRetAttributes());
    NewAttributes.push_back(RA);
  }

  // Promote call by value arguments; copy other arguments including attributes
  unsigned ArgIdx = 1;
  for (auto i = F->arg_begin(), e = F->arg_end(); i != e; ++i, ++ArgIdx) {
    if (i->hasByValAttr())
      NewArgs.push_back(cast<PointerType>(i->getType())->getElementType());
    else {
      NewArgs.push_back(i->getType());
      AttributeSet attrs = FAS.getParamAttributes(ArgIdx);
      if (attrs.hasAttributes(ArgIdx)) {
        AttrBuilder B(attrs, ArgIdx);
        NewAttributes.push_back(AttributeSet::get(FC, NewArgs.size(), B));
      }
    }
  }

  if (FAS.hasAttributes(AttributeSet::FunctionIndex)) {
    LLVMContext &FTyC = FTy->getContext();
    NewAttributes.push_back(AttributeSet::get(FTyC, FAS.getFnAttributes()));
  }

  llvm::Type *RetTy = FTy->getReturnType();
  FunctionType *NFty = FunctionType::get(RetTy, NewArgs, FTy->isVarArg());
  llvm::Function *NF =
      llvm::Function::Create(NFty, F->getLinkage(), F->getName());
  NF->copyAttributesFrom(F);
  NF->setAttributes(AttributeSet::get(FC, NewAttributes));

  return NF;
}

void ArgumentPromotionPass::updateCallSite(CallSite *CS, llvm::Function *F,
                                           llvm::Function *NF) {
  Instruction *CI = CS->getInstruction();
  const AttributeSet &CAS = CS->getAttributes();
  LLVMContext &FC = F->getContext();
  std::vector<Value *> NewArgs;
  std::vector<AttributeSet> NewAttributes;

  if (!dyn_cast<CallInst>(CI)) {
    ErrorReporter::reportImplementationLimitation(
        "Only call instructions supported as call sites");
  }

  if (CAS.hasAttributes(AttributeSet::ReturnIndex))
    NewAttributes.push_back(AttributeSet::get(FC, CAS.getRetAttributes()));

  // Create load instruction for each promoted argument and keep track of the
  // attributes from every other argument
  unsigned ArgIdx = 0;
  auto AI = CS->arg_begin();
  for (auto i = F->arg_begin(), e = F->arg_end(); i != e; ++i, ++AI, ++ArgIdx) {
    if (i->hasByValAttr())
      NewArgs.push_back(new LoadInst(*AI, (*AI)->getName() + ".val", CI));
    else {
      NewArgs.push_back(*AI);
      if (CAS.hasAttributes(ArgIdx)) {
        AttrBuilder B(CAS, ArgIdx);
        NewAttributes.push_back(AttributeSet::get(FC, NewArgs.size(), B));
      }
    }
  }
  // Handle varargs although these should no occur in GPU code
  for (; AI != CS->arg_end(); ++AI, ++ArgIdx) {
    NewArgs.push_back(*AI);
    if (CAS.hasAttributes(ArgIdx)) {
      AttrBuilder B(CAS, ArgIdx);
      NewAttributes.push_back(AttributeSet::get(FC, NewArgs.size(), B));
    }
  }

  // Handle function attributes
  if (CAS.hasAttributes(AttributeSet::FunctionIndex))
    NewAttributes.push_back(
        AttributeSet::get(CI->getContext(), CAS.getFnAttributes()));

  CallInst *NC = CallInst::Create(NF, NewArgs, "", CI);
  NC->setCallingConv(CS->getCallingConv());
  NC->setAttributes(AttributeSet::get(NC->getContext(), NewAttributes));
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

void ArgumentPromotionPass::spliceBody(llvm::Function *F, llvm::Function *NF) {
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  for (auto i = F->arg_begin(), e = F->arg_end(), ni = NF->arg_begin(); i != e;
       ++i, ++ni) {
    Value *AI = ni;
    ni->takeName(i);

    // Copy the value of the function argument into locally allocated space
    if (i->hasByValAttr()) {
      Instruction *InsertPoint = NF->begin()->begin();
      llvm::Type *ArgTy = cast<PointerType>(i->getType())->getElementType();
      AI = new AllocaInst(ArgTy, ni->getName() + ".val", InsertPoint);
      new StoreInst(ni, AI, InsertPoint);
    }

    i->replaceAllUsesWith(AI);
  }
}

void ArgumentPromotionPass::promote(llvm::Function *F) {
  llvm::Function *NF = createNewFunction(F);

  // Update debug information to point to new function
  auto DI = FunctionDIs.find(F);
  if (DI != FunctionDIs.end()) {
    DISubprogram SP = DI->second;
    SP.replaceFunction(NF);
    FunctionDIs.erase(DI);
    FunctionDIs[NF] = SP;
  }

  // Replace the function in the remaining (non-debug) meta-data
  std::set<MDNode *> doneMD;
  const auto &NMDL = M->getNamedMDList();
  for (auto i = NMDL.begin(), e = NMDL.end(); i != e; ++i) {
    for (unsigned j = 0, k = i->getNumOperands(); j != k; ++j) {
      MDNode *MD = i->getOperand(j);
      replaceMetaData(F, NF, MD, doneMD);
    }
  }

  // Insert new function and take F's name
  F->getParent()->getFunctionList().insert(F, NF);
  NF->takeName(F);

  while (!F->use_empty()) {
    CallSite CS(F->user_back());
    updateCallSite(&CS, F, NF);
  }

  spliceBody(F, NF);

  // The remains of F will be removed by a dead-code elimination pass
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
    else if (MD->getOperand(i) == F)
      MD->replaceOperandWith(i, NF);
    else if (auto MDV = dyn_cast<MDNode>(MD->getOperand(i)))
      replaceMetaData(F, NF, MDV, doneMD);
  }
}

bool ArgumentPromotionPass::runOnModule(llvm::Module &M) {
  bool promoted = false;
  this->M = &M;
  FunctionDIs = makeSubprogramMap(M); // Needed to updated debug information

  for (auto i = M.begin(), e = M.end(); i != e; ++i) {
    if (needsPromotion(i)) {
      promote(i);
      promoted = true;
    }
  }

  return promoted;
}

char ArgumentPromotionPass::ID = 0;
