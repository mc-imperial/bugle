#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/BasicBlock.h"
#include "bugle/Expr.h"
#include "bugle/GlobalArray.h"
#include "bugle/Module.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/raw_ostream.h"
#include "klee/util/GetElementPtrTypeIterator.h"

using namespace bugle;
using namespace llvm;

llvm::StringMap<TranslateFunction::SpecialFnHandler TranslateFunction::*>
  TranslateFunction::SpecialFunctionMap;

void TranslateFunction::translate() {
  if (SpecialFunctionMap.empty()) {
    SpecialFunctionMap["bugle_assert"] = &TranslateFunction::handleAssert;
    SpecialFunctionMap["bugle_assume"] = &TranslateFunction::handleAssume;
    SpecialFunctionMap["__assert_fail"] = &TranslateFunction::handleAssertFail;
  }

  for (auto i = F->arg_begin(), e = F->arg_end(); i != e; ++i) {
    Var *V = BF->addArgument(TM->translateType(i->getType()), i->getName());
    ValueExprMap[&*i] = VarRefExpr::create(V);
  }

  if (BF->return_begin() != BF->return_end())
    ReturnVar = *BF->return_begin();

  for (auto i = F->begin(), e = F->end(); i != e; ++i)
    BasicBlockMap[&*i] = BF->addBasicBlock(i->getName());

  for (auto i = F->begin(), e = F->end(); i != e; ++i)
    translateBasicBlock(BasicBlockMap[&*i], &*i);
}

ref<Expr> TranslateFunction::translateValue(llvm::Value *V) {
  if (isa<Instruction>(V) || isa<Argument>(V)) {
    auto MI = ValueExprMap.find(V);
    assert(MI != ValueExprMap.end());
    return MI->second;
  }

  if (auto C = dyn_cast<Constant>(V))
    return TM->translateConstant(C);

  assert(0 && "Unsupported value");
  return 0;
}

template <typename T, typename I, typename F>
T fold(T init, I begin, I end, F func) {
  T value = init;
  for (I i = begin; i != end; ++i)
    value = func(value, *i);
  return value;
}

Var *TranslateFunction::getPhiVariable(llvm::PHINode *PN) {
  auto &i = PhiVarMap[PN];
  if (i)
    return i;

  i = BF->addLocal(TM->translateType(PN->getType()), PN->getName());
  return i;
}

void TranslateFunction::addPhiAssigns(bugle::BasicBlock *BBB,
                                      llvm::BasicBlock *Pred,
                                      llvm::BasicBlock *Succ) {
  std::vector<Var *> Vars;
  std::vector<ref<Expr>> Exprs;
  for (auto i = Succ->begin(), e = Succ->end(); i != e && isa<PHINode>(i); ++i){
    PHINode *PN = cast<PHINode>(i);
    int idx = PN->getBasicBlockIndex(Pred);
    assert(idx != -1 && "No phi index?");

    Vars.push_back(getPhiVariable(PN));
    Exprs.push_back(translateValue(PN->getIncomingValue(idx)));
  }

  if (!Vars.empty())
    BBB->addStmt(new VarAssignStmt(Vars, Exprs));
}

ref<Expr> TranslateFunction::handleAssert(bugle::BasicBlock *BBB,
                                          const std::vector<ref<Expr>> &Args) {
  BBB->addStmt(new AssertStmt(
    NeExpr::create(Args[0],
                   BVConstExpr::createZero(Args[0]->getType().width))));
  return 0;
}

ref<Expr> TranslateFunction::handleAssertFail(bugle::BasicBlock *BBB,
                                           const std::vector<ref<Expr>> &Args) {
  BBB->addStmt(new AssertStmt(BoolConstExpr::create(false)));
  return 0;
}

ref<Expr> TranslateFunction::handleAssume(bugle::BasicBlock *BBB,
                                          const std::vector<ref<Expr>> &Args) {
  BBB->addStmt(new AssumeStmt(
    NeExpr::create(Args[0],
                   BVConstExpr::createZero(Args[0]->getType().width))));
  return 0;
}

ref<Expr> TranslateFunction::maybeTranslateSIMDInst(bugle::BasicBlock *BBB,
                               llvm::Type *OpType,
                               std::function<ref<Expr>(ref<Expr>, ref<Expr>)> F,
                               ref<Expr> LHS, ref<Expr> RHS) {
  if (!isa<VectorType>(OpType))
    return F(LHS, RHS);

  auto VT = cast<VectorType>(OpType);
  unsigned NumElems = VT->getNumElements();
  unsigned ElemWidth = LHS->getType().width / NumElems;
  std::vector<ref<Expr>> Elems;
  for (unsigned i = 0; i < NumElems; ++i) {
    ref<Expr> LHSi = BVExtractExpr::create(LHS, i*ElemWidth, ElemWidth);
    ref<Expr> RHSi = BVExtractExpr::create(RHS, i*ElemWidth, ElemWidth);
    if (VT->getElementType()->isFloatingPointTy()) {
      LHSi = BVToFloatExpr::create(LHSi);
      RHSi = BVToFloatExpr::create(RHSi);
    }
    ref<Expr> Elem = F(LHSi, RHSi);
    BBB->addStmt(new EvalStmt(Elem));
    if (VT->getElementType()->isFloatingPointTy()) {
      Elem = FloatToBVExpr::create(Elem);
      BBB->addStmt(new EvalStmt(Elem));
    }
    Elems.push_back(Elem);
  }
  return fold(Elems.back(), Elems.rbegin()+1, Elems.rend(),
              BVConcatExpr::create);
}

void TranslateFunction::translateInstruction(bugle::BasicBlock *BBB,
                                             Instruction *I) {
  ref<Expr> E;
  if (auto BO = dyn_cast<BinaryOperator>(I)) {
    ref<Expr> LHS = translateValue(BO->getOperand(0)),
              RHS = translateValue(BO->getOperand(1));
    ref<Expr> (*F)(ref<Expr>, ref<Expr>);
    switch (BO->getOpcode()) {
    case BinaryOperator::Add:  F = BVAddExpr::create;  break;
    case BinaryOperator::FAdd: F = FAddExpr::create;   break;
    case BinaryOperator::Sub:  F = BVSubExpr::create;  break;
    case BinaryOperator::FSub: F = FSubExpr::create;   break;
    case BinaryOperator::Mul:  F = BVMulExpr::create;  break;
    case BinaryOperator::FMul: F = FMulExpr::create;   break;
    case BinaryOperator::SDiv: F = BVSDivExpr::create; break;
    case BinaryOperator::UDiv: F = BVUDivExpr::create; break;
    case BinaryOperator::FDiv: F = FDivExpr::create;   break;
    case BinaryOperator::SRem: F = BVSRemExpr::create; break;
    case BinaryOperator::URem: F = BVURemExpr::create; break;
    case BinaryOperator::Shl:  F = BVShlExpr::create;  break;
    case BinaryOperator::AShr: F = BVAShrExpr::create; break;
    case BinaryOperator::LShr: F = BVLShrExpr::create; break;
    case BinaryOperator::And:  F = BVAndExpr::create;  break;
    case BinaryOperator::Or:   F = BVOrExpr::create;   break;
    case BinaryOperator::Xor:  F = BVXorExpr::create;  break;
    default:
      assert(0 && "Unsupported binary operator");
    }
    E = maybeTranslateSIMDInst(BBB, BO->getOperand(0)->getType(), F, LHS, RHS);
  } else if (auto GEPI = dyn_cast<GetElementPtrInst>(I)) {
    ref<Expr> Ptr = translateValue(GEPI->getPointerOperand());
    E = TM->translateGEP(Ptr, klee::gep_type_begin(GEPI),
                         klee::gep_type_end(GEPI),
                         [&](Value *V) { return translateValue(V); });
  } else if (auto AI = dyn_cast<AllocaInst>(I)) {
    GlobalArray *GA = TM->BM->addGlobal(AI->getName());
    E = PointerExpr::create(GlobalArrayRefExpr::create(GA),
                        BVConstExpr::createZero(TM->TD.getPointerSizeInBits()));
  } else if (auto LI = dyn_cast<LoadInst>(I)) {
    ref<Expr> Ptr = translateValue(LI->getPointerOperand()),
              PtrArr = ArrayIdExpr::create(Ptr),
              PtrOfs = ArrayOffsetExpr::create(Ptr);
    std::vector<ref<Expr> > BytesLoaded;
    Type LoadTy = TM->translateType(LI->getType());
    assert(LoadTy.width % 8 == 0);
    for (unsigned i = 0; i != LoadTy.width / 8; ++i) {
      ref<Expr> PtrByteOfs =
        BVAddExpr::create(PtrOfs,
                          BVConstExpr::create(PtrOfs->getType().width, i));
      ref<Expr> ValByte = LoadExpr::create(PtrArr, PtrByteOfs);
      BytesLoaded.push_back(ValByte);
      BBB->addStmt(new EvalStmt(ValByte));
    }
    E = fold(BytesLoaded.back(), BytesLoaded.rbegin()+1, BytesLoaded.rend(),
             BVConcatExpr::create);
    if (LoadTy.kind == Type::Pointer)
      E = BVToPtrExpr::create(E);
    else if (LoadTy.kind == Type::Float)
      E = BVToFloatExpr::create(E);
  } else if (auto SI = dyn_cast<StoreInst>(I)) {
    ref<Expr> Ptr = translateValue(SI->getPointerOperand()),
              Val = translateValue(SI->getValueOperand()),
              PtrArr = ArrayIdExpr::create(Ptr),
              PtrOfs = ArrayOffsetExpr::create(Ptr);
    Type StoreTy = Val->getType();
    assert(StoreTy.width % 8 == 0);
    if (StoreTy.kind == Type::Pointer) {
      Val = PtrToBVExpr::create(Val);
      BBB->addStmt(new EvalStmt(Val));
    } else if (StoreTy.kind == Type::Float) {
      Val = FloatToBVExpr::create(Val);
      BBB->addStmt(new EvalStmt(Val));
    }
    for (unsigned i = 0; i != Val->getType().width / 8; ++i) {
      ref<Expr> PtrByteOfs =
        BVAddExpr::create(PtrOfs,
                          BVConstExpr::create(PtrOfs->getType().width, i));
      ref<Expr> ValByte =
        BVExtractExpr::create(Val, i*8, 8); // Assumes little endian
      BBB->addStmt(new StoreStmt(PtrArr, PtrByteOfs, ValByte));
    }
    return;
  } else if (auto II = dyn_cast<ICmpInst>(I)) {
    ref<Expr> LHS = translateValue(II->getOperand(0)),
              RHS = translateValue(II->getOperand(1));
    if (II->getPredicate() == ICmpInst::ICMP_EQ)
      E = EqExpr::create(LHS, RHS);
    else if (II->getPredicate() == ICmpInst::ICMP_NE)
      E = NeExpr::create(LHS, RHS);
    else if (LHS->getType().kind == Type::Pointer) {
      assert(RHS->getType().kind == Type::Pointer);
      switch (II->getPredicate()) {
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT: E = Expr::createPtrLt(LHS, RHS); break;
      case ICmpInst::ICMP_ULE:
      case ICmpInst::ICMP_SLE: E = Expr::createPtrLe(LHS, RHS); break;
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT: E = Expr::createPtrLt(RHS, LHS); break;
      case ICmpInst::ICMP_UGE:
      case ICmpInst::ICMP_SGE: E = Expr::createPtrLe(RHS, LHS); break;
      default:
        assert(0 && "Unsupported ptr icmp");
      }
    } else {
      assert(RHS->getType().kind == Type::BV);
      switch (II->getPredicate()) {
      case ICmpInst::ICMP_UGT: E = BVUgtExpr::create(LHS, RHS); break;
      case ICmpInst::ICMP_UGE: E = BVUgeExpr::create(LHS, RHS); break;
      case ICmpInst::ICMP_ULT: E = BVUltExpr::create(LHS, RHS); break;
      case ICmpInst::ICMP_ULE: E = BVUleExpr::create(LHS, RHS); break;
      case ICmpInst::ICMP_SGT: E = BVSgtExpr::create(LHS, RHS); break;
      case ICmpInst::ICMP_SGE: E = BVSgeExpr::create(LHS, RHS); break;
      case ICmpInst::ICMP_SLT: E = BVSltExpr::create(LHS, RHS); break;
      case ICmpInst::ICMP_SLE: E = BVSleExpr::create(LHS, RHS); break;
      default:
        assert(0 && "Unsupported icmp");
      }
    }
    BBB->addStmt(new EvalStmt(E));
    E = BoolToBVExpr::create(E);
  } else if (auto ZEI = dyn_cast<ZExtInst>(I)) {
    ref<Expr> Op = translateValue(ZEI->getOperand(0));
    E = BVZExtExpr::create(cast<IntegerType>(ZEI->getType())->getBitWidth(),Op);
  } else if (auto SEI = dyn_cast<SExtInst>(I)) {
    ref<Expr> Op = translateValue(SEI->getOperand(0));
    E = BVSExtExpr::create(cast<IntegerType>(SEI->getType())->getBitWidth(),Op);
  } else if (auto TI = dyn_cast<TruncInst>(I)) {
    ref<Expr> Op = translateValue(TI->getOperand(0));
    unsigned Width = cast<IntegerType>(TI->getType())->getBitWidth();
    E = BVExtractExpr::create(Op, 0, Width);
  } else if (auto I2PI = dyn_cast<IntToPtrInst>(I)) {
    ref<Expr> Op = translateValue(I2PI->getOperand(0));
    E = BVToPtrExpr::create(Op);
  } else if (auto P2II = dyn_cast<PtrToIntInst>(I)) {
    ref<Expr> Op = translateValue(P2II->getOperand(0));
    E = PtrToBVExpr::create(Op);
  } else if (auto BCI = dyn_cast<BitCastInst>(I)) {
    ref<Expr> Op = translateValue(BCI->getOperand(0));
    if (BCI->getSrcTy()->isPointerTy() && BCI->getDestTy()->isPointerTy()) {
      ValueExprMap[I] = Op;
      return;
    } else if (BCI->getSrcTy()->isFloatingPointTy() &&
               BCI->getDestTy()->isIntegerTy()) {
      E = FloatToBVExpr::create(Op);
    } else if (BCI->getSrcTy()->isIntegerTy() &&
               BCI->getDestTy()->isFloatingPointTy()) {
      E = BVToFloatExpr::create(Op);
    } else {
      assert(0 && "Unsupported bitcast");
    }
  } else if (auto SI = dyn_cast<SelectInst>(I)) {
    ref<Expr> Cond = translateValue(SI->getCondition()),
              TrueVal = translateValue(SI->getTrueValue()),
              FalseVal = translateValue(SI->getFalseValue());
    Cond = BVToBoolExpr::create(Cond);
    E = IfThenElseExpr::create(Cond, TrueVal, FalseVal);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    auto F = CI->getCalledFunction();
    assert(F && "Only direct calls for now");
    auto FI = TM->FunctionMap.find(F);
    assert(FI != TM->FunctionMap.end() && "Could not find function in map!");

    CallSite CS(CI);
    std::vector<ref<Expr>> Args;
    std::transform(CS.arg_begin(), CS.arg_end(), std::back_inserter(Args),
                   [&](Value *V) { return translateValue(V); });

    auto SFI = SpecialFunctionMap.find(F->getName());
    if (SFI != SpecialFunctionMap.end()) {
      E = (this->*SFI->second)(BBB, Args);
      assert(E.isNull() == CI->getType()->isVoidTy());
      if (E.isNull())
        return;
    } else if (CI->getType()->isVoidTy()) {
      BBB->addStmt(new CallStmt(FI->second, Args));
      return;
    } else {
      E = CallExpr::create(FI->second, Args);
    }
  } else if (auto RI = dyn_cast<ReturnInst>(I)) {
    if (auto V = RI->getReturnValue()) {
      assert(ReturnVar && "Returning value without return variable?");
      ref<Expr> Val = translateValue(V);
      BBB->addStmt(new VarAssignStmt(ReturnVar, Val));
    }
    BBB->addStmt(new ReturnStmt);
    return;
  } else if (auto BI = dyn_cast<BranchInst>(I)) {
    if (BI->isConditional()) {
      ref<Expr> Cond = BVToBoolExpr::create(translateValue(BI->getCondition()));

      bugle::BasicBlock *TrueBB = BF->addBasicBlock("truebb");
      TrueBB->addStmt(new AssumeStmt(Cond));
      addPhiAssigns(TrueBB, I->getParent(), BI->getSuccessor(0));
      TrueBB->addStmt(new GotoStmt(BasicBlockMap[BI->getSuccessor(0)]));

      bugle::BasicBlock *FalseBB = BF->addBasicBlock("falsebb");
      FalseBB->addStmt(new AssumeStmt(NotExpr::create(Cond)));
      addPhiAssigns(FalseBB, I->getParent(), BI->getSuccessor(1));
      FalseBB->addStmt(new GotoStmt(BasicBlockMap[BI->getSuccessor(1)]));

      std::vector<bugle::BasicBlock *> BBs;
      BBs.push_back(TrueBB);
      BBs.push_back(FalseBB);
      BBB->addStmt(new GotoStmt(BBs));
    } else {
      addPhiAssigns(BBB, I->getParent(), BI->getSuccessor(0));
      BBB->addStmt(new GotoStmt(BasicBlockMap[BI->getSuccessor(0)]));
    }
    return;
  } else if (auto PN = dyn_cast<PHINode>(I)) {
    ValueExprMap[I] = VarRefExpr::create(getPhiVariable(PN));
    return;
  } else {
    assert(0 && "Unsupported instruction");
  }
  ValueExprMap[I] = E;
  BBB->addStmt(new EvalStmt(E));
}

void TranslateFunction::translateBasicBlock(bugle::BasicBlock *BBB,
                                            llvm::BasicBlock *BB) {
  for (auto i = BB->begin(), e = BB->end(); i != e; ++i)
    translateInstruction(BBB, &*i);
}
