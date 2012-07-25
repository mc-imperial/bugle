#include "bugle/Translator/TranslateModule.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/Module.h"
#include "bugle/Stmt.h"
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

void TranslateModule::translateGlobalInit(GlobalArray *GA, unsigned Offset,
                                          Constant *Init) {
  if (auto CS = dyn_cast<ConstantStruct>(Init)) {
    auto SL = TD.getStructLayout(CS->getType());
    for (unsigned i = 0; i < CS->getNumOperands(); ++i)
      translateGlobalInit(GA, Offset + SL->getElementOffset(i),
                          CS->getOperand(i));
  } else if (auto CA = dyn_cast<ConstantArray>(Init)) {
    uint64_t ElemSize = TD.getTypeStoreSize(CA->getType()->getElementType());
    for (unsigned i = 0; i < CA->getNumOperands(); ++i)
      translateGlobalInit(GA, Offset + i*ElemSize, CA->getOperand(i));
  } else {
    ref<Expr> Const = translateConstant(Init);
    unsigned InitByteWidth = Const->getType().width/8;
    if (GA->getRangeType() == Const->getType() && Offset % InitByteWidth == 0) {
      BM->addGlobalInit(GA, Offset/InitByteWidth, Const);
    } else if (GA->getRangeType() == Type(Type::BV, 8)) {
      if (Init->getType()->isFloatingPointTy())
        Const = FloatToBVExpr::create(Const);
      if (Init->getType()->isPointerTy())
        Const = PtrToBVExpr::create(Const);

      for (unsigned i = 0; i < Const->getType().width/8; ++i)
        BM->addGlobalInit(GA, Offset+i, BVExtractExpr::create(Const, i*8, 8));
    } else {
      NeedAdditionalByteArrayModels = true;
      ModelAsByteArray.insert(GlobalValueMap[GA]);
    }
  }
}

void TranslateModule::addGlobalArrayAttribs(GlobalArray *GA, PointerType *PT) {
  if (SL == SL_OpenCL) {
    switch (PT->getAddressSpace()) {
      case 1: GA->addAttribute("global");       break;
      case 3: GA->addAttribute("group_shared"); break;
      default: ;
    }
  }
}

GlobalArray *TranslateModule::translateGlobalVariable(GlobalVariable *GV) {
  GlobalArray *GA = addGlobalArray(GV);
  if (GV->hasInitializer() &&
      // OpenCL __local variables have bogus initialisers.
      !(SL == SL_OpenCL && GV->getType()->getAddressSpace() == 3))
    translateGlobalInit(GA, 0, GV->getInitializer());
  return GA;
}

ref<Expr> TranslateModule::translateUndef(bugle::Type t) {
  ref<Expr> E = BVConstExpr::createZero(t.width);
  if (t.kind == Type::Float)
    return BVToFloatExpr::create(E);
  else if (t.kind == Type::Pointer)
    return BVToPtrExpr::create(E);
  else
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
    case Instruction::BitCast:
      return translateBitCast(CE->getOperand(0)->getType(), CE->getType(),
                              translateConstant(CE->getOperand(0)));
    default:
      assert(0 && "Unhandled ConstantExpr");
    }
  }
  if (auto GV = dyn_cast<GlobalVariable>(C)) {
    GlobalArray *GA = translateGlobalVariable(GV);
    return PointerExpr::create(GlobalArrayRefExpr::create(GA),
                            BVConstExpr::createZero(TD.getPointerSizeInBits()));
  }
  if (auto UV = dyn_cast<UndefValue>(C)) {
    return translateUndef(translateType(UV->getType()));
  }
  if (auto CDS = dyn_cast<ConstantDataSequential>(C)) {
    std::vector<ref<Expr>> Elems;
    for (unsigned i = 0; i != CDS->getNumElements(); ++i) {
      if (CDS->getElementType()->isFloatingPointTy())
        Elems.push_back(
          BVConstExpr::create(CDS->getElementAsAPFloat(i).bitcastToAPInt()));
      else
        Elems.push_back(BVConstExpr::create(CDS->getElementByteSize()*8,
                                            CDS->getElementAsInteger(i)));
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
  if (isa<ConstantPointerNull>(C)) {
    return PointerExpr::create(NullArrayRefExpr::create(), 
                            BVConstExpr::createZero(TD.getPointerSizeInBits()));
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

bugle::Type TranslateModule::translateArrayRangeType(llvm::Type *T) {
  if (auto AT = dyn_cast<ArrayType>(T))
    return translateArrayRangeType(AT->getElementType());
  if (auto VT = dyn_cast<VectorType>(T))
    return translateArrayRangeType(VT->getElementType());
  if (isa<StructType>(T))
    return Type(Type::BV, 8);

  return translateType(T);
}

bugle::GlobalArray *TranslateModule::addGlobalArray(llvm::Value *V) {
  bugle::Type T(Type::BV, 8);
  auto PT = cast<PointerType>(V->getType());
  if (!(ModelAllAsByteArray ||
        ModelAsByteArray.find(V) != ModelAsByteArray.end()))
    T = translateArrayRangeType(PT->getElementType());
  auto GA = BM->addGlobal(V->getName(), T);
  addGlobalArrayAttribs(GA, PT);
  GlobalValueMap[GA] = V;
  return GA;
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

ref<Expr> TranslateModule::translateBitCast(llvm::Type *SrcTy,
                                            llvm::Type *DestTy,
                                            ref<Expr> Op) {
  if (SrcTy->isFloatingPointTy() && !DestTy->isFloatingPointTy()) {
    return FloatToBVExpr::create(Op);
  } else if (!SrcTy->isFloatingPointTy() && DestTy->isFloatingPointTy()) {
    return BVToFloatExpr::create(Op);
  } else {
    return Op;
  }
}

void TranslateModule::addGPUEntryPoint(StringRef Name) {
  GPUEntryPoints.insert(Name);
}

static bool isAxiomFunction(StringRef Name) {
  return Name.startswith("__axiom");
}

void TranslateModule::translate() {

  do {
    NeedAdditionalByteArrayModels = false;

    delete BM;
    BM = new bugle::Module;

    FunctionMap.clear();
    ConstantMap.clear();
    GlobalValueMap.clear();

    BM->setPointerWidth(TD.getPointerSizeInBits());

    for (auto i = M->begin(), e = M->end(); i != e; ++i) {
      if (i->isIntrinsic() || isAxiomFunction(i->getName()) ||
          TranslateFunction::isSpecialFunction(SL, i->getName()))
        continue;

      auto BF = FunctionMap[&*i] = BM->addFunction(i->getName());

      auto RT = i->getFunctionType()->getReturnType();
      if (!RT->isVoidTy())
        BF->addReturn(translateType(RT), "ret");
    }

    for (auto i = M->begin(), e = M->end(); i != e; ++i) {
      if (i->isIntrinsic())
        continue;

      if (isAxiomFunction(i->getName())) {
        bugle::Function F("");
        Type RT = translateType(i->getFunctionType()->getReturnType());
        Var *RV = F.addReturn(RT, "ret");
        TranslateFunction TF(this, &F, &*i);
        TF.translate();
        assert(F.begin()+1 == F.end() && "Expected one basic block");
        bugle::BasicBlock *BB = *F.begin();
        VarAssignStmt *S = cast<VarAssignStmt>(*(BB->end()-2));
        assert(S->getVars()[0] == RV);
        BM->addAxiom(Expr::createNeZero(S->getValues()[0]));
      } else if (!TranslateFunction::isSpecialFunction(SL, i->getName())) {
        TranslateFunction TF(this, FunctionMap[&*i], &*i);
        TF.isGPUEntryPoint =
          (GPUEntryPoints.find(i->getName()) != GPUEntryPoints.end());
        TF.translate();
      }
    }
  } while (NeedAdditionalByteArrayModels);
}
