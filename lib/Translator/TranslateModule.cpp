#include "bugle/Translator/TranslateModule.h"
#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/Module.h"
#include "bugle/Stmt.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace bugle;

static cl::opt<bool> ModelBVAsByteArray(
    "model-bv-as-byte-array", cl::Hidden, cl::init(false),
    cl::desc("Model each array composed of bit vector elements as an array of "
             "bit vectors of size 8"));

ref<Expr> TranslateModule::translateConstant(Constant *C) {
  ref<Expr> &E = ConstantMap[C];
  if (E.isNull())
    E = doTranslateConstant(C);
  E->preventEvalStmt = true;
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
      translateGlobalInit(GA, Offset + i * ElemSize, CA->getOperand(i));
  } else {
    ref<Expr> Const = translateConstant(Init);
    unsigned InitByteWidth = Const->getType().width / 8;
    Type GATy = GA->getRangeType();
    if (GATy == Const->getType() && Offset % InitByteWidth == 0) {
      BM->addGlobalInit(GA, Offset / InitByteWidth, Const);
    } else if (GATy.isKind(Type::BV) && Offset % (GATy.width / 8) == 0 &&
               InitByteWidth % (GATy.width / 8) == 0) {
      llvm::Type *InitTy = Init->getType();
      if (InitTy->isPointerTy()) {
        if (InitTy->getPointerElementType()->isFunctionTy())
          Const = FuncPtrToBVExpr::create(Const);
        else
          Const = PtrToBVExpr::create(Const);
      }

      unsigned GAWidth = GATy.width;
      for (unsigned i = 0; i < Const->getType().width / GAWidth; ++i) {
        BM->addGlobalInit(GA, Offset + i,
                          BVExtractExpr::create(Const, i * GAWidth, GAWidth));
      }
    } else {
      NeedAdditionalByteArrayModels = true;
      ModelAsByteArray.insert(GlobalValueMap[GA]);
    }
  }
}

void TranslateModule::addGlobalArrayAttribs(GlobalArray *GA, PointerType *PT) {
  // If we have a pointer in CUDA constant address space, only the pointer
  // is constant, unless used as a pointer, the memory pointed to will be
  // cudaMalloc'ed and hence be in device memory.
  if (SL == SL_CUDA && PT->getElementType()->isPointerTy() &&
      PT->getAddressSpace() == AddressSpaces.constant)
    GA->addAttribute("global");
  else if (SL == SL_OpenCL || SL == SL_CUDA) {
    if (PT->getAddressSpace() == AddressSpaces.global)
      GA->addAttribute("global");
    else if (PT->getAddressSpace() == AddressSpaces.group_shared)
      GA->addAttribute("group_shared");
    else if (PT->getAddressSpace() == AddressSpaces.constant)
      GA->addAttribute("constant");
  }
}

ref<Expr> TranslateModule::translateCUDABuiltinGlobal(std::string Prefix,
                                                      GlobalVariable *GV) {
  Type ty = translateArrayRangeType(GV->getType()->getElementType());
  ref<Expr> Arr[3] = {SpecialVarRefExpr::create(ty, Prefix + "_x"),
                      SpecialVarRefExpr::create(ty, Prefix + "_y"),
                      SpecialVarRefExpr::create(ty, Prefix + "_z")};
  return ConstantArrayRefExpr::create(Arr);
}

bool TranslateModule::hasInitializer(GlobalVariable *GV) {
  if (!GV->hasInitializer())
    return false;

  // OpenCL __local and CUDA __shared__ variables have bogus initializers
  if ((SL == SL_OpenCL || SL == SL_CUDA) &&
      GV->getType()->getAddressSpace() == AddressSpaces.group_shared)
    return false;

  // CUDA __constant__ and __device__ variables have initializers that may
  // have been overwritten by the host program
  if (SL == SL_CUDA &&
      (GV->getType()->getAddressSpace() == AddressSpaces.constant ||
       GV->getType()->getAddressSpace() == AddressSpaces.global))
    return false;

  return true;
}

ref<Expr> TranslateModule::translateGlobalVariable(GlobalVariable *GV) {
  if (SL == SL_CUDA) {
    if (GV->getName() == "gridDim")
      return translateCUDABuiltinGlobal("num_groups", GV);
    if (GV->getName() == "blockIdx")
      return translateCUDABuiltinGlobal("group_id", GV);
    if (GV->getName() == "blockDim")
      return translateCUDABuiltinGlobal("group_size", GV);
    if (GV->getName() == "threadIdx")
      return translateCUDABuiltinGlobal("local_id", GV);
  }

  GlobalArray *GA = getGlobalArray(GV);
  if (hasInitializer(GV))
    translateGlobalInit(GA, 0, GV->getInitializer());
  return GlobalArrayRefExpr::create(GA);
}

ref<Expr> TranslateModule::translateArbitrary(bugle::Type t) {
  ref<Expr> E = BVConstExpr::createZero(t.width);
  if (t.isKind(Type::Pointer))
    return BVToPtrExpr::create(E);
  else if (t.isKind(Type::FunctionPointer))
    return BVToFuncPtrExpr::create(E);
  else
    return E;
}

ref<Expr> TranslateModule::doTranslateConstant(Constant *C) {
  if (auto CI = dyn_cast<ConstantInt>(C))
    return BVConstExpr::create(CI->getValue());
  if (auto CF = dyn_cast<ConstantFP>(C))
    return BVConstExpr::create(CF->getValueAPF().bitcastToAPInt());
  if (auto CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      ref<Expr> Op = translateConstant(CE->getOperand(0));
      return translateGEP(
          Op, klee::gep_type_begin(CE), klee::gep_type_end(CE),
          [&](Value *V) { return translateConstant(cast<Constant>(V)); });
    }
    case Instruction::BitCast:
      return translateBitCast(CE->getOperand(0)->getType(), CE->getType(),
                              translateConstant(CE->getOperand(0)));
    case Instruction::AddrSpaceCast:
      return translateConstant(CE->getOperand(0));
    case Instruction::Mul: {
      ref<Expr> LHS = translateConstant(CE->getOperand(0)),
                RHS = translateConstant(CE->getOperand(1));
      return BVMulExpr::create(LHS, RHS);
    }
    case Instruction::PtrToInt: {
      ref<Expr> Op = translateConstant(CE->getOperand(0));
      Type OpTy = Op->getType();
      assert(OpTy.isKind(Type::Pointer) || OpTy.isKind(Type::FunctionPointer));
      if (OpTy.isKind(Type::FunctionPointer))
        return FuncPtrToBVExpr::create(Op);
      else
        return PtrToBVExpr::create(Op);
    }
    case Instruction::IntToPtr: {
      ref<Expr> Op = translateConstant(CE->getOperand(0));
      assert(CE->getType()->isPointerTy());
      if (CE->getType()->getPointerElementType()->isFunctionTy())
        return BVToFuncPtrExpr::create(Op);
      else
        return BVToPtrExpr::create(Op);
    }
    case Instruction::ICmp: {
      ref<Expr> LHS = translateConstant(CE->getOperand(0)),
                RHS = translateConstant(CE->getOperand(1));
      switch (CE->getPredicate()) {
      case ICmpInst::ICMP_EQ:
        return BoolToBVExpr::create(EqExpr::create(LHS, RHS));
      case ICmpInst::ICMP_NE:
        return BoolToBVExpr::create(NeExpr::create(LHS, RHS));
      default:
        std::string msg = "Unhandled icmp expression";
        ErrorReporter::reportImplementationLimitation(msg);
      }
    }
    case Instruction::ZExt: {
      llvm::IntegerType *IntTy = cast<IntegerType>(CE->getType());
      ref<Expr> Op = translateConstant(CE->getOperand(0));
      return BVZExtExpr::create(IntTy->getBitWidth(), Op);
    }
    default:
      std::string name = CE->getOpcodeName();
      std::string msg = "Unhandled constant expression '" + name + "'";
      ErrorReporter::reportImplementationLimitation(msg);
    }
  }
  if (auto GV = dyn_cast<GlobalVariable>(C)) {
    ref<Expr> Arr = translateGlobalVariable(GV);
    return PointerExpr::create(
        Arr, BVConstExpr::createZero(TD.getPointerSizeInBits()));
  }
  if (auto F = dyn_cast<llvm::Function>(C)) {
    auto FI = FunctionMap.find(F);
    if (FI == FunctionMap.end()) {
      std::string DN = getOriginalGlobalArrayName(F);
      std::string msg = "Unsupported function pointer '" + DN + "'";
      ErrorReporter::reportImplementationLimitation(msg);
    }
    std::string name = FI->second->getName();
    return FunctionPointerExpr::create(name, TD.getPointerSizeInBits());
  }
  if (auto UV = dyn_cast<UndefValue>(C)) {
    return translateArbitrary(translateType(UV->getType()));
  }
  if (auto CDS = dyn_cast<ConstantDataSequential>(C)) {
    std::vector<ref<Expr>> Elems;
    for (unsigned i = 0; i != CDS->getNumElements(); ++i) {
      if (CDS->getElementType()->isFloatingPointTy())
        Elems.push_back(
            BVConstExpr::create(CDS->getElementAsAPFloat(i).bitcastToAPInt()));
      else
        Elems.push_back(BVConstExpr::create(CDS->getElementByteSize() * 8,
                                            CDS->getElementAsInteger(i)));
    }
    return Expr::createBVConcatN(Elems);
  }
  if (auto CV = dyn_cast<ConstantVector>(C)) {
    std::vector<ref<Expr>> Elems;
    std::transform(
        CV->op_begin(), CV->op_end(), std::back_inserter(Elems),
        [&](Use &U) { return translateConstant(cast<Constant>(U.get())); });
    return Expr::createBVConcatN(Elems);
  }
  if (auto CAZ = dyn_cast<ConstantAggregateZero>(C)) {
    return BVConstExpr::createZero(TD.getTypeSizeInBits(CAZ->getType()));
  }
  if (isa<ConstantPointerNull>(C)) {
    if (C->getType()->getPointerElementType()->isFunctionTy())
      return NullFunctionPointerExpr::create(TD.getPointerSizeInBits());
    else
      return PointerExpr::create(
          NullArrayRefExpr::create(),
          BVConstExpr::createZero(TD.getPointerSizeInBits()));
  }
  ErrorReporter::reportImplementationLimitation("Unhandled constant");
}

bugle::Type TranslateModule::translateType(llvm::Type *T) {
  Type::Kind K;
  if (T->isPointerTy()) {
    llvm::Type *ElTy = T->getPointerElementType();
    if (ElTy->isFunctionTy())
      K = Type::FunctionPointer;
    else
      K = Type::Pointer;
  } else
    K = Type::BV;

  return Type(K, TD.getTypeSizeInBits(T));
}

bugle::Type TranslateModule::translateArrayRangeType(llvm::Type *T) {
  if (auto AT = dyn_cast<ArrayType>(T))
    return translateArrayRangeType(AT->getElementType());
  if (auto VT = dyn_cast<VectorType>(T))
    return translateArrayRangeType(VT->getElementType());
  if (auto ST = dyn_cast<StructType>(T)) {
    auto i = ST->element_begin(), e = ST->element_end();
    if (i == e)
      return Type(Type::BV, 8);
    auto ET = *i;
    ++i;
    for (; i != e; ++i) {
      if (ET != *i)
        return Type(Type::BV, 8);
    }
    return translateArrayRangeType(ET);
  }

  return translateType(T);
}

bugle::GlobalArray *TranslateModule::getGlobalArray(llvm::Value *V) {
  GlobalArray *&GA = ValueGlobalMap[V];
  if (GA)
    return GA;

  bugle::Type T(Type::BV, 8);
  auto PT = cast<PointerType>(V->getType());

  if (!(ModelAllAsByteArray ||
        ModelAsByteArray.find(V) != ModelAsByteArray.end())) {
    T = translateArrayRangeType(PT->getElementType());
    if (ModelBVAsByteArray && T.isKind(Type::BV)) {
      ModelAsByteArray.insert(V);
      T = Type(Type::BV, 8);
    }
  }
  GA = BM->addGlobal(V->getName(), getOriginalGlobalArrayName(V), T);
  addGlobalArrayAttribs(GA, PT);
  GlobalValueMap[GA] = V;
  return GA;
}

ref<Expr>
TranslateModule::translateGEP(ref<Expr> Ptr, klee::gep_type_iterator begin,
                              klee::gep_type_iterator end,
                              std::function<ref<Expr>(Value *)> xlate) {
  ref<Expr> PtrArr = ArrayIdExpr::create(Ptr, defaultRange()),
            PtrOfs = ArrayOffsetExpr::create(Ptr);
  for (auto i = begin; i != end; ++i) {
    if (StructType *st = dyn_cast<StructType>(*i)) {
      const StructLayout *sl = TD.getStructLayout(st);
      const ConstantInt *ci = cast<ConstantInt>(i.getOperand());
      uint64_t addend = sl->getElementOffset((unsigned)ci->getZExtValue());
      PtrOfs = BVAddExpr::create(
          PtrOfs, BVConstExpr::create(BM->getPointerWidth(), addend));
    } else if (SequentialType *set = cast<SequentialType>(*i)) {
      uint64_t elementSize = TD.getTypeStoreSize(set->getElementType());
      Value *operand = i.getOperand();
      ref<Expr> index = xlate(operand);
      index = BVZExtExpr::create(BM->getPointerWidth(), index);
      ref<Expr> addend = BVMulExpr::create(
          index, BVConstExpr::create(BM->getPointerWidth(), elementSize));
      PtrOfs = BVAddExpr::create(PtrOfs, addend);
    } else {
      ErrorReporter::reportImplementationLimitation("Unhandled GEP type");
    }
  }
  return PointerExpr::create(PtrArr, PtrOfs);
}

ref<Expr>
TranslateModule::translateEV(ref<Expr> Vec, klee::ev_type_iterator begin,
                             klee::ev_type_iterator end,
                             std::function<ref<Expr>(Value *)> xlate) {
  ref<Expr> ValElem = Vec;
  for (auto i = begin; i != end; ++i) {
    if (StructType *st = dyn_cast<StructType>(*i)) {
      const StructLayout *sl = TD.getStructLayout(st);
      const ConstantInt *ci = cast<ConstantInt>(i.getOperand());
      uint64_t addend = sl->getElementOffset((unsigned)ci->getZExtValue());
      llvm::Type *Ty = st->getElementType((unsigned)ci->getZExtValue());
      uint64_t size = TD.getTypeSizeInBits(Ty);
      ValElem = BVExtractExpr::create(ValElem, addend * 8, size);
    } else {
      ErrorReporter::reportImplementationLimitation("Unhandled EV type");
    }
  }
  return ValElem;
}

ref<Expr> TranslateModule::translateBitCast(llvm::Type *SrcTy,
                                            llvm::Type *DestTy, ref<Expr> Op) {
  if (SrcTy->isPointerTy() && DestTy->isPointerTy() &&
      SrcTy->getPointerElementType()->isFunctionTy() &&
      !DestTy->getPointerElementType()->isFunctionTy())
    return FuncPtrToPtrExpr::create(Op);
  else if (SrcTy->isPointerTy() && DestTy->isPointerTy() &&
           !SrcTy->getPointerElementType()->isFunctionTy() &&
           DestTy->getPointerElementType()->isFunctionTy())
    return PtrToFuncPtrExpr::create(Op);
  else
    return Op;
}

bool TranslateModule::isGPUEntryPoint(llvm::Function *F, llvm::Module *M,
                                      SourceLanguage SL,
                                      std::set<std::string> &EPS) {
  if (SL == SL_OpenCL || SL == SL_CUDA) {
    if (NamedMDNode *NMD = M->getNamedMetadata("nvvm.annotations")) {
      for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
        MDNode *MD = NMD->getOperand(i);
        if (MD->getOperand(0) == F)
          for (unsigned fi = 1, fe = MD->getNumOperands(); fi != fe; fi += 2)
            if (MD->getOperand(fi)->getName() == "kernel")
              return true;
      }
    }
  }

  if (SL == SL_OpenCL) {
    if (NamedMDNode *NMD = M->getNamedMetadata("opencl.kernels")) {
      for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
        MDNode *MD = NMD->getOperand(i);
        if (MD->getOperand(0) == F)
          return true;
      }
    }
  }

  return EPS.find(F->getName()) != EPS.end();
}

std::string TranslateModule::getOriginalFunctionName(llvm::Function *F) {
  auto SS = DIF.subprograms();
  for (auto i = SS.begin(), e = SS.end(); i != e; ++i) {
    DISubprogram subprogram(*i);
    if (subprogram.describes(F)) {
      return subprogram.getName();
    }
  }

  return F->getName().str();
}

std::string TranslateModule::getOriginalGlobalArrayName(llvm::Value *V) {
  GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
  if (!GV)
    return V->getName();

  auto GVS = DIF.global_variables();
  for (auto i = GVS.begin(), e = GVS.end(); i != e; ++i) {
    DIGlobalVariable globalVariable(*i);
    if (globalVariable.getGlobal() == GV) {
      return globalVariable.getName();
    }
  }

  return GV->getName().str();
}

// Convert the given unmodelled expression E to modelled form.
ref<Expr> TranslateModule::modelValue(Value *V, ref<Expr> E) {
  if (E->getType().isKind(Type::Pointer)) {
    auto OI = ModelPtrAsGlobalOffset.find(V);
    if (OI != ModelPtrAsGlobalOffset.end()) {
      auto GA = getGlobalArray(*OI->second.begin());
      auto Ofs = ArrayOffsetExpr::create(E);
      Ofs = Expr::createExactBVUDiv(Ofs, GA->getRangeType().width / 8);
      assert(!Ofs.isNull() && "Couldn't create div this time!");

      if (OI->second.size() == 1 &&
          PtrMayBeNull.find(V) == PtrMayBeNull.end()) {
        return Ofs;
      } else {
        return PointerExpr::create(ArrayIdExpr::create(E, defaultRange()), Ofs);
      }
    }
  }

  return E;
}

// If the given value is modelled, return its modelled type, else return
// its conventional Boogie type (translateType).
bugle::Type TranslateModule::getModelledType(Value *V) {
  auto OI = ModelPtrAsGlobalOffset.find(V);
  if (OI != ModelPtrAsGlobalOffset.end() && OI->second.size() == 1 &&
      PtrMayBeNull.find(V) == PtrMayBeNull.end()) {
    return Type(Type::BV, TD.getPointerSizeInBits());
  } else {
    llvm::Type *VTy = V->getType();
    if (auto F = dyn_cast<llvm::Function>(V))
      VTy = F->getReturnType();

    return translateType(VTy);
  }
}

// Convert the given modelled expression E to unmodelled form.
ref<Expr> TranslateModule::unmodelValue(Value *V, ref<Expr> E) {
  auto OI = ModelPtrAsGlobalOffset.find(V);
  if (OI != ModelPtrAsGlobalOffset.end()) {
    auto GA = getGlobalArray(*OI->second.begin());
    auto WidthCst = BVConstExpr::create(TD.getPointerSizeInBits(),
                                        GA->getRangeType().width / 8);
    if (OI->second.size() == 1 && PtrMayBeNull.find(V) == PtrMayBeNull.end()) {
      return PointerExpr::create(GlobalArrayRefExpr::create(GA),
                                 BVMulExpr::create(E, WidthCst));
    } else {
      std::set<GlobalArray *> Globals;
      std::transform(OI->second.begin(), OI->second.end(),
                     std::inserter(Globals, Globals.begin()),
                     [&](Value *V) { return getGlobalArray(V); });

      if (PtrMayBeNull.find(V) != PtrMayBeNull.end())
        Globals.insert((bugle::GlobalArray *)0);

      auto AI = ArrayIdExpr::create(E, defaultRange());
      auto AMO = ArrayMemberOfExpr::create(AI, Globals);
      auto AO = BVMulExpr::create(ArrayOffsetExpr::create(E), WidthCst);
      return PointerExpr::create(AMO, AO);
    }
  } else {
    return E;
  }
}

/// Given a value and all possible Boogie expressions to which it may be
/// assigned, compute a model for that value such that future invocations
/// of modelValue/getModelledType/unmodelValue use that model.
void TranslateModule::computeValueModel(Value *Val, Var *Var,
                                        llvm::ArrayRef<ref<Expr>> Assigns) {
  llvm::Type *VTy = Val->getType();
  if (auto F = dyn_cast<llvm::Function>(Val))
    VTy = F->getReturnType();

  if (!VTy->isPointerTy())
    return;
  if (VTy->getPointerElementType()->isFunctionTy())
    return;
  if (ModelPtrAsGlobalOffset.find(Val) != ModelPtrAsGlobalOffset.end())
    return;

  std::set<GlobalArray *> GlobalSet;
  for (auto ai = Assigns.begin(), ae = Assigns.end(); ai != ae; ++ai) {
    if ((*ai)->computeArrayCandidates(GlobalSet))
      continue;
    else
      return;
  }

  assert(!GlobalSet.empty() && "GlobalSet is empty?");

  // Now check that each array in GlobalSet has the same type.
  Type GlobalsType = Expr::getArrayCandidateType(GlobalSet);

  // Check that each offset is a multiple of the range type's byte width (or
  // that if the offset refers to the variable, it maintains the invariant).
  bool ModelGlobalsAsByteArray = false;
  if (GlobalsType.isKind(Type::Any) || GlobalsType.isKind(Type::Unknown)) {
    ModelGlobalsAsByteArray = true;
  } else {
    for (auto ai = Assigns.begin(), ae = Assigns.end(); ai != ae; ++ai) {
      auto AOE = ArrayOffsetExpr::create(*ai);
      if (Expr::createExactBVUDiv(AOE, GlobalsType.width / 8, Var).isNull()) {
        ModelGlobalsAsByteArray = true;
        break;
      }
    }
  }

  // Remove null pointer candidates
  auto null = (bugle::GlobalArray *)0;
  if (GlobalSet.find(null) != GlobalSet.end()) {
    NextPtrMayBeNull.insert(Val);
    GlobalSet.erase(GlobalSet.find(null));
  }

  // If we only had null pointers, there is nothing to do
  if (GlobalSet.size() == 0)
    return;

  // Success! Record the global set.
  auto &GlobalValSet = NextModelPtrAsGlobalOffset[Val];
  std::transform(GlobalSet.begin(), GlobalSet.end(),
                 std::inserter(GlobalValSet, GlobalValSet.begin()),
                 [&](GlobalArray *A) { return GlobalValueMap[A]; });
  NeedAdditionalGlobalOffsetModels = true;

  if (ModelGlobalsAsByteArray) {
    std::transform(GlobalSet.begin(), GlobalSet.end(),
                   std::inserter(ModelAsByteArray, ModelAsByteArray.begin()),
                   [&](GlobalArray *A) { return GlobalValueMap[A]; });
    NeedAdditionalByteArrayModels = true;
  }
}

Stmt *TranslateModule::modelCallStmt(llvm::Type *T, llvm::Function *F,
                                     ref<Expr> Val,
                                     std::vector<ref<Expr>> &args) {
  std::map<llvm::Function *, Function *> FS;

  if (F) {
    auto FI = FunctionMap.find(F);
    assert(FI != FunctionMap.end() && "Couldn't find function in map!");
    FS[F] = (FI->second);
  } else {
    for (auto i = FunctionMap.begin(), e = FunctionMap.end(); i != e; ++i) {
      if (i->first->getType() == T && !i->second->isEntryPoint())
        FS[i->first] = i->second;
    }
  }

  std::vector<Stmt *> CSS;
  for (auto i = FS.begin(), e = FS.end(); i != e; ++i) {
    std::vector<ref<Expr>> fargs;
    std::transform(args.begin(), args.end(), i->first->arg_begin(),
                   std::back_inserter(fargs), [&](ref<Expr> E, Argument &Arg) {
      return modelValue(&Arg, E);
    });
    auto CS = new CallStmt(i->second, fargs);
    CallSites[i->first].push_back(&CS->getArgs());
    CSS.push_back(CS);
  }

  if (CSS.size() == 0)
    ErrorReporter::reportFatalError("No functions for function pointer found");

  if (F)
    return *CSS.begin();
  else
    return new CallMemberOfStmt(Val, CSS);
}

ref<Expr> TranslateModule::modelCallExpr(llvm::Type *T, llvm::Function *F,
                                         ref<Expr> Val,
                                         std::vector<ref<Expr>> &args) {
  std::map<llvm::Function *, Function *> FS;

  if (F) {
    auto FI = FunctionMap.find(F);
    assert(FI != FunctionMap.end() && "Couldn't find function in map!");
    FS[F] = (FI->second);
  } else {
    for (auto i = FunctionMap.begin(), e = FunctionMap.end(); i != e; ++i)
      if (i->first->getType() == T && !i->second->isEntryPoint())
        FS[i->first] = i->second;
  }

  std::vector<ref<Expr>> CES;
  for (auto i = FS.begin(), e = FS.end(); i != e; ++i) {
    std::vector<ref<Expr>> fargs;
    std::transform(args.begin(), args.end(), i->first->arg_begin(),
                   std::back_inserter(fargs), [&](ref<Expr> E, Argument &Arg) {
      return modelValue(&Arg, E);
    });
    ref<Expr> E = CallExpr::create(i->second, fargs);
    auto CE = dyn_cast<CallExpr>(E);
    CallSites[i->first].push_back(&CE->getArgs());
    CES.push_back(CE);
  }

  if (CES.size() == 0)
    ErrorReporter::reportFatalError("No functions for function pointer found");

  if (F)
    return *CES.begin();
  else
    return CallMemberOfExpr::create(Val, CES);
}

void TranslateModule::translate() {
  do {
    NeedAdditionalByteArrayModels = false;
    NeedAdditionalGlobalOffsetModels = false;

    delete BM;
    BM = new bugle::Module;

    FunctionMap.clear();
    ConstantMap.clear();
    GlobalValueMap.clear();
    ValueGlobalMap.clear();
    CallSites.clear();

    BM->setPointerWidth(TD.getPointerSizeInBits());

    for (auto i = M->begin(), e = M->end(); i != e; ++i) {

      if (TranslateFunction::isUninterpretedFunction(i->getName())) {
        TranslateFunction::addUninterpretedFunction(SL, i->getName());
      }

      if (i->isIntrinsic() ||
          TranslateFunction::isAxiomFunction(i->getName()) ||
          TranslateFunction::isSpecialFunction(SL, i->getName()))
        continue;

      auto BF = FunctionMap[&*i] =
          BM->addFunction(i->getName(), getOriginalFunctionName(i));

      auto RT = i->getFunctionType()->getReturnType();
      if (!RT->isVoidTy())
        BF->addReturn(getModelledType(i), "ret");
    }

    for (auto i = M->begin(), e = M->end(); i != e; ++i) {
      if (i->isIntrinsic())
        continue;

      if (TranslateFunction::isAxiomFunction(i->getName())) {
        bugle::Function F("", "");
        Type RT = translateType(i->getFunctionType()->getReturnType());
        Var *RV = F.addReturn(RT, "ret");
        TranslateFunction TF(this, &F, &*i, false);
        TF.translate();
        assert(F.begin() + 1 == F.end() && "Expected one basic block");
        bugle::BasicBlock *BB = *F.begin();
        VarAssignStmt *S = cast<VarAssignStmt>(*(BB->end() - 2));
        assert(S->getVars()[0] == RV); (void)RV;
        BM->addAxiom(Expr::createNeZero(S->getValues()[0]));
      } else if (!TranslateFunction::isSpecialFunction(SL, i->getName())) {
        bool EP = isGPUEntryPoint(i, M, SL, GPUEntryPoints);
        TranslateFunction TF(this, FunctionMap[&*i], &*i, EP);
        TF.translate();
      }
    }

    // If this round gave us a case split, examine each pointer argument to
    // each call site for each function to see if the argument always refers to
    // the same global array, in which case we can model the parameter as an
    // offset, and potentially avoid the case split.
    if (!ModelAllAsByteArray && NextModelAllAsByteArray) {
      for (auto i = CallSites.begin(), e = CallSites.end(); i != e; ++i) {
        unsigned pidx = 0;
        for (auto pi = i->first->arg_begin(), pe = i->first->arg_end();
             pi != pe; ++pi, ++pidx) {
          std::vector<ref<Expr>> Parms;
          std::transform(
              i->second.begin(), i->second.end(), std::back_inserter(Parms),
              [&](const std::vector<ref<Expr>> *cs) { return (*cs)[pidx]; });
          computeValueModel(pi, 0, Parms);
        }
      }
    }

    if (NeedAdditionalGlobalOffsetModels) {
      // If we can model new pointers using global offsets, a previously
      // observed case split may become unnecessary.  So when we recompute the
      // fixed point, don't use byte array models for everything unless we're
      // stuck.
      ModelAllAsByteArray = NextModelAllAsByteArray = false;
    } else {
      ModelAllAsByteArray = NextModelAllAsByteArray;
    }

    ModelPtrAsGlobalOffset = NextModelPtrAsGlobalOffset;
    PtrMayBeNull = NextPtrMayBeNull;
  } while (NeedAdditionalByteArrayModels || NeedAdditionalGlobalOffsetModels);
}

TranslateModule::AddressSpaceMap::AddressSpaceMap(unsigned Global,
                                                  unsigned GroupShared,
                                                  unsigned Constant)
    : standard(0), global(Global), group_shared(GroupShared),
      constant(Constant) {
  assert(Global != 0 && Global != GroupShared && Global != Constant);
  assert(GroupShared != 0 && GroupShared != Global && GroupShared != Constant);
  assert(Constant != 0 && Constant != Global && Constant != GroupShared);
}
