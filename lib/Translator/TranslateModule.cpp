#include "bugle/Translator/TranslateModule.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/Module.h"
#include "llvm/Constant.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"

using namespace llvm;
using namespace bugle;

ref<Expr> TranslateModule::translateConstant(Constant *C) {
  ref<Expr> &E = ConstantMap[C];
  if (E.isNull())
    E = doTranslateConstant(C);
  E->isDerivedFromConstant = true;
  return E;
}

ref<Expr> TranslateModule::doTranslateConstant(Constant *C) {
  if (auto CI = dyn_cast<ConstantInt>(C))
    return BVConstExpr::create(CI->getValue());
  if (auto CF = dyn_cast<ConstantFP>(C))
    return BVToFloatExpr::create(
             BVConstExpr::create(CF->getValueAPF().bitcastToAPInt()));
  if (auto CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      ref<Expr> Op = translateConstant(CE->getOperand(0));
      return translateGEP(Op, klee::gep_type_begin(CE), klee::gep_type_end(CE),
                          [&](Value *V) {
                            return translateConstant(cast<Constant>(V));
                          });
    }
    default:
      assert(0 && "Unhandled ConstantExpr");
    }
  }
  if (auto GV = dyn_cast<GlobalVariable>(C)) {
    GlobalArray *GA = BM->addGlobal(GV->getName());
    return PointerExpr::create(GlobalArrayRefExpr::create(GA),
                            BVConstExpr::createZero(TD.getPointerSizeInBits()));
  }
  if (auto UV = dyn_cast<UndefValue>(C)) {
    ref<Expr> CE = BVConstExpr::createZero(TD.getTypeSizeInBits(UV->getType()));
    if (UV->getType()->isFloatingPointTy())
      CE = BVToFloatExpr::create(CE);
    return CE;
  }
  if (auto CDV = dyn_cast<ConstantDataVector>(C)) {
    std::vector<ref<Expr>> Elems;
    for (unsigned i = 0; i != CDV->getNumElements(); ++i) {
      if (CDV->getElementType()->isFloatingPointTy())
        Elems.push_back(
          BVConstExpr::create(CDV->getElementAsAPFloat(i).bitcastToAPInt()));
      else
        Elems.push_back(BVConstExpr::create(CDV->getElementByteSize()*8,
                                            CDV->getElementAsInteger(i)));
    }
    return Expr::createBVConcatN(Elems);
  }
  if (auto CV = dyn_cast<ConstantVector>(C)) {
    std::vector<ref<Expr>> Elems;
    std::transform(CV->op_begin(), CV->op_end(), std::back_inserter(Elems),
                   [&](Use &U) -> ref<Expr> {
      ref<Expr> E = translateConstant(cast<Constant>(U.get()));
      if (U.get()->getType()->isFloatingPointTy())
        E = FloatToBVExpr::create(E);
      return E;
    });
    return Expr::createBVConcatN(Elems);
  }
  if (auto CAZ = dyn_cast<ConstantAggregateZero>(C)) {
    ref<Expr> CE = BVConstExpr::createZero(TD.getTypeSizeInBits(CAZ->getType()));
    if (CAZ->getType()->isFloatingPointTy())
      CE = BVToFloatExpr::create(CE);
    return CE;
  }
  assert(0 && "Unhandled constant");
  return 0;
}

bugle::Type TranslateModule::translateType(llvm::Type *T) {
  Type::Kind K;
  if (T->isFloatingPointTy())
    K = Type::Float;
  else if (T->isPointerTy())
    K = Type::Pointer;
  else
    K = Type::BV;

  return Type(K, TD.getTypeSizeInBits(T));
}

ref<Expr> TranslateModule::translateGEP(ref<Expr> Ptr,
                                        klee::gep_type_iterator begin,
                                        klee::gep_type_iterator end,
                                      std::function<ref<Expr>(Value *)> xlate) {
  ref<Expr> PtrArr = ArrayIdExpr::create(Ptr),
            PtrOfs = ArrayOffsetExpr::create(Ptr);
  for (auto i = begin; i != end; ++i) {
    if (StructType *st = dyn_cast<StructType>(*i)) {
      const StructLayout *sl = TD.getStructLayout(st);
      const ConstantInt *ci = cast<ConstantInt>(i.getOperand());
      uint64_t addend = sl->getElementOffset((unsigned) ci->getZExtValue());
      PtrOfs = BVAddExpr::create(PtrOfs,
                 BVConstExpr::create(BM->getPointerWidth(), addend));
    } else {
      const SequentialType *set = cast<SequentialType>(*i);
      uint64_t elementSize = 
        TD.getTypeStoreSize(set->getElementType());
      Value *operand = i.getOperand();
      ref<Expr> index = xlate(operand);
      index = BVSExtExpr::create(BM->getPointerWidth(), index);
      ref<Expr> addend =
        BVMulExpr::create(index,
          BVConstExpr::create(BM->getPointerWidth(), elementSize));
      PtrOfs = BVAddExpr::create(PtrOfs, addend);
    }
  }
  return PointerExpr::create(PtrArr, PtrOfs);
}

void TranslateModule::addGPUEntryPoint(StringRef Name) {
  GPUEntryPoints.insert(Name);
}

void TranslateModule::translate() {
   BM->setPointerWidth(TD.getPointerSizeInBits());

  for (auto i = M->begin(), e = M->end(); i != e; ++i) {
    auto BF = FunctionMap[&*i] = BM->addFunction(i->getName());

    auto RT = i->getFunctionType()->getReturnType();
    if (!RT->isVoidTy())
      BF->addReturn(translateType(RT), "ret");
  }

  for (auto i = M->begin(), e = M->end(); i != e; ++i) {
    TranslateFunction TF(this, FunctionMap[&*i], &*i);
    TF.isGPUEntryPoint =
      (GPUEntryPoints.find(i->getName()) != GPUEntryPoints.end());
    TF.translate();
  }
}
