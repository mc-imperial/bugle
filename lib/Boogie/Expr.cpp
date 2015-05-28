#include "bugle/Expr.h"
#include "bugle/BPLExprWriter.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/util/Functional.h"
#include "llvm/Support/raw_ostream.h"

using namespace bugle;

bool Expr::computeArrayCandidates(std::set<GlobalArray *> &GlobalSet) const {
  if (auto GARE = dyn_cast<GlobalArrayRefExpr>(this)) {
    GlobalSet.insert(GARE->getArray());
    return true;
  } else if (auto MOE = dyn_cast<ArrayMemberOfExpr>(this)) {
    GlobalSet.insert(MOE->getElems().begin(), MOE->getElems().end());
    return true;
  } else if (isa<NullArrayRefExpr>(this)) {
    GlobalSet.insert((bugle::GlobalArray *)0);
    return true;
  } else if (auto ITE = dyn_cast<IfThenElseExpr>(this)) {
    return ITE->getTrueExpr()->computeArrayCandidates(GlobalSet) &&
           ITE->getFalseExpr()->computeArrayCandidates(GlobalSet);
  } else if (auto AIE = dyn_cast<ArrayIdExpr>(this)) {
    return AIE->getSubExpr()->computeArrayCandidates(GlobalSet);
  } else if (auto PE = dyn_cast<PointerExpr>(this)) {
    return PE->getArray()->computeArrayCandidates(GlobalSet);
  } else {
    return false;
  }
}

ref<Expr> BVConstExpr::create(const llvm::APInt &bv) {
  return new BVConstExpr(bv);
}

ref<Expr> BVConstExpr::createZero(unsigned width) {
  return create(llvm::APInt(width, 0));
}

ref<Expr> BVConstExpr::create(unsigned width, uint64_t val, bool isSigned) {
  return create(llvm::APInt(width, val, isSigned));
}

ref<Expr> BoolConstExpr::create(bool val) { return new BoolConstExpr(val); }

ref<Expr> GlobalArrayRefExpr::create(GlobalArray *global) {
  return new GlobalArrayRefExpr(Type(Type::ArrayOf, global->getRangeType()),
                                global);
}

ref<Expr> NullArrayRefExpr::create() { return new NullArrayRefExpr(); }

ref<Expr> ConstantArrayRefExpr::create(llvm::ArrayRef<ref<Expr>> array) {
#ifndef NDEBUG
  assert(array.size() > 0);
  Type t = array[0]->getType();
  for (auto i = array.begin() + 1, e = array.end(); i != e; ++i) {
    assert((*i)->getType() == t);
  }
#endif

  for (auto i = array.begin(), e = array.end(); i != e; ++i)
    (*i)->preventEvalStmt = true;

  return new ConstantArrayRefExpr(array);
}

ref<Expr> PointerExpr::create(ref<Expr> array, ref<Expr> offset) {
  assert(array->getType().array);
  assert(offset->getType().isKind(Type::BV));

  return new PointerExpr(array, offset);
}

ref<Expr> NullFunctionPointerExpr::create(unsigned ptrWidth) {
  return new NullFunctionPointerExpr(ptrWidth);
}

ref<Expr> FunctionPointerExpr::create(std::string funcName, unsigned ptrWidth) {
  return new FunctionPointerExpr(funcName, ptrWidth);
}

ref<Expr> LoadExpr::create(ref<Expr> array, ref<Expr> offset, Type type,
                           bool isTemporal) {
  assert(array->getType().array);
  assert(offset->getType().isKind(Type::BV));

  if (auto CA = dyn_cast<ConstantArrayRefExpr>(array)) {
    // More restrictive than strictly necessary, but sufficient for CUDA.
    auto OfsCE = cast<BVConstExpr>(offset);
    uint64_t Ofs = OfsCE->getValue().getZExtValue();
    assert(Ofs < CA->getArray().size());
    return CA->getArray()[Ofs];
  }

  return new LoadExpr(type, array, offset, isTemporal);
}

ref<Expr> AtomicExpr::create(ref<Expr> array, ref<Expr> offset,
                             std::vector<ref<Expr>> args, std::string function,
                             unsigned parts, unsigned part) {
  Type at = array->getType();
  assert(at.array);
  assert(offset->getType().isKind(Type::BV));
  assert(at.range().isKind(Type::BV));

  return new AtomicExpr(at.range(), array, offset, args, function, parts, part);
}

ref<Expr> VarRefExpr::create(Var *var) { return new VarRefExpr(var); }

ref<Expr> SpecialVarRefExpr::create(Type t, const std::string &attr) {
  return new SpecialVarRefExpr(t, attr);
}

ref<Expr> BVExtractExpr::create(ref<Expr> expr, unsigned offset,
                                unsigned width) {
  assert(expr->getType().isKind(Type::BV));
  if (offset == 0 && width == expr->getType().width)
    return expr;

  if (auto e = dyn_cast<BVConstExpr>(expr))
    return BVConstExpr::create(e->getValue().ashr(offset).zextOrTrunc(width));

  if (auto e = dyn_cast<BVConcatExpr>(expr)) {
    unsigned RHSWidth = e->getRHS()->getType().width;
    if (offset + width <= RHSWidth)
      return BVExtractExpr::create(e->getRHS(), offset, width);
    if (offset >= RHSWidth)
      return BVExtractExpr::create(e->getLHS(), offset - RHSWidth, width);
    if (offset == 0 && RHSWidth < width) {
      ref<Expr> EE =
          BVExtractExpr::create(e->getLHS(), offset, width - RHSWidth);
      return BVConcatExpr::create(EE, e->getRHS());
    }
  }

  if (isa<BVZExtExpr>(expr) || isa<BVSExtExpr>(expr)) {
    auto UE = cast<UnaryExpr>(expr);
    if (offset + width <= UE->getSubExpr()->getType().width)
      return BVExtractExpr::create(UE->getSubExpr(), offset, width);
  }

  return new BVExtractExpr(expr, offset, width);
}

ref<Expr> BVCtlzExpr::create(ref<Expr> val, ref<Expr> isZeroUndef) {
  assert(val->getType().isKind(Type::BV));
  assert(isZeroUndef->getType().isKind(Type::Bool));

  return new BVCtlzExpr(val->getType(), val, isZeroUndef);
}

ref<Expr> NotExpr::create(ref<Expr> op) {
  assert(op->getType().isKind(Type::Bool));
  if (auto e = dyn_cast<BoolConstExpr>(op))
    return BoolConstExpr::create(!e->getValue());

  return new NotExpr(Type(Type::Bool), op);
}

Type Expr::getArrayCandidateType(const std::set<GlobalArray *> &Globals) {
  Type t(Type::Any);
  for (auto gi = Globals.begin(), ge = Globals.end(); gi != ge; ++gi) {
    if (*gi) {
      if (t.kind == Type::Any)
        t = (*gi)->getRangeType();
      else if (t != (*gi)->getRangeType())
        return Type(Type::Unknown);
    }
  }
  return t;
}

Type Expr::getPointerRange(ref<Expr> pointer, Type defaultRange) {
  assert(pointer->getType().isKind(Type::Pointer));
  Type range = defaultRange;
  std::set<GlobalArray *> Globals;
  if (pointer->computeArrayCandidates(Globals))
    range = getArrayCandidateType(Globals);
  return range;
}

ref<Expr> ArrayIdExpr::create(ref<Expr> pointer, Type defaultRange) {
  assert(pointer->getType().isKind(Type::Pointer));

  if (auto e = dyn_cast<PointerExpr>(pointer))
    return e->getArray();

  Type range = getPointerRange(pointer, defaultRange);

  return new ArrayIdExpr(Type(Type::ArrayOf, range), pointer);
}

ref<Expr> ArrayOffsetExpr::create(ref<Expr> pointer) {
  assert(pointer->getType().isKind(Type::Pointer));

  if (auto e = dyn_cast<PointerExpr>(pointer))
    return e->getOffset();

  return new ArrayOffsetExpr(Type(Type::BV, pointer->getType().width), pointer);
}

ref<Expr> BVZExtExpr::create(unsigned width, ref<Expr> bv) {
  const Type &ty = bv->getType();
  assert(ty.isKind(Type::BV));

  if (width == ty.width)
    return bv;
  if (width < ty.width)
    return BVExtractExpr::create(bv, 0, width);

  if (auto e = dyn_cast<BVConstExpr>(bv))
    return BVConstExpr::create(e->getValue().zext(width));

  return new BVZExtExpr(Type(Type::BV, width), bv);
}

ref<Expr> BVSExtExpr::create(unsigned width, ref<Expr> bv) {
  const Type &ty = bv->getType();
  assert(ty.isKind(Type::BV));

  if (width == ty.width)
    return bv;
  if (width < ty.width)
    return BVExtractExpr::create(bv, 0, width);

  if (auto e = dyn_cast<BVConstExpr>(bv))
    return BVConstExpr::create(e->getValue().sext(width));

  return new BVSExtExpr(Type(Type::BV, width), bv);
}

ref<Expr> FPConvExpr::create(unsigned width, ref<Expr> expr) {
  const Type &ty = expr->getType();
  assert(ty.isKind(Type::BV));

  if (width == ty.width)
    return expr;

  return new FPConvExpr(Type(Type::BV, width), expr);
}

ref<Expr> FPToSIExpr::create(unsigned width, ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FPToSIExpr(Type(Type::BV, width), expr);
}

ref<Expr> FPToUIExpr::create(unsigned width, ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FPToUIExpr(Type(Type::BV, width), expr);
}

ref<Expr> SIToFPExpr::create(unsigned width, ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new SIToFPExpr(Type(Type::BV, width), expr);
}

ref<Expr> UIToFPExpr::create(unsigned width, ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new UIToFPExpr(Type(Type::BV, width), expr);
}

ref<Expr> FAbsExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FAbsExpr(expr->getType(), expr);
}

ref<Expr> FCeilExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FCeilExpr(expr->getType(), expr);
}

ref<Expr> FCosExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FCosExpr(expr->getType(), expr);
}

ref<Expr> FExpExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FExpExpr(expr->getType(), expr);
}

ref<Expr> FLogExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FLogExpr(expr->getType(), expr);
}

ref<Expr> FrexpExpExpr::create(unsigned width, ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FrexpExpExpr(Type(Type::BV, width), expr);
}

ref<Expr> FrexpFracExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FrexpFracExpr(expr->getType(), expr);
}

ref<Expr> FFloorExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FFloorExpr(expr->getType(), expr);
}

ref<Expr> FRintExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FRintExpr(expr->getType(), expr);
}

ref<Expr> FSinExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FSinExpr(expr->getType(), expr);
}

ref<Expr> FRsqrtExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FRsqrtExpr(expr->getType(), expr);
}

ref<Expr> FSqrtExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FSqrtExpr(expr->getType(), expr);
}

ref<Expr> FTruncExpr::create(ref<Expr> expr) {
  assert(expr->getType().isKind(Type::BV));
  return new FTruncExpr(expr->getType(), expr);
}

ref<Expr> IfThenElseExpr::create(ref<Expr> cond, ref<Expr> trueExpr,
                                 ref<Expr> falseExpr) {
  assert(cond->getType().isKind(Type::Bool));
  assert(trueExpr->getType() == falseExpr->getType());

  if (auto e = dyn_cast<BoolConstExpr>(cond))
    return e->getValue() ? trueExpr : falseExpr;

  return new IfThenElseExpr(cond, trueExpr, falseExpr);
}

ref<Expr> HavocExpr::create(Type type) { return new HavocExpr(type); }

ref<Expr> ArrayMemberOfExpr::create(ref<Expr> expr,
                                    const std::set<GlobalArray *> &elems) {
  assert(expr->getType().array);
  assert(!elems.empty());

  Type t = Expr::getArrayCandidateType(elems);
#ifndef NDEBUG
  for (auto i = elems.begin(), e = elems.end(); i != e; ++i) {
    assert(*i == (bugle::GlobalArray *)0 || (*i)->getRangeType() == t);
  }
#endif

  return new ArrayMemberOfExpr(Type(Type::ArrayOf, t), expr, elems);
}

ref<Expr> BVToPtrExpr::create(unsigned ptrWidth, ref<Expr> bv) {
  assert(bv->getType().isKind(Type::BV));

  if (auto e = dyn_cast<PtrToBVExpr>(bv))
    return e->getSubExpr();

  if (auto e = dyn_cast<BVConstExpr>(bv))
    if (e->getValue() == 0)
      return PointerExpr::create(NullArrayRefExpr::create(),
                                 BVConstExpr::createZero(ptrWidth));

  return new BVToPtrExpr(Type(Type::Pointer, ptrWidth), bv);
}

ref<Expr> PtrToBVExpr::create(unsigned bvWidth, ref<Expr> ptr) {
  assert(ptr->getType().isKind(Type::Pointer));

  if (auto e = dyn_cast<BVToPtrExpr>(ptr))
    return BVZExtExpr::create(bvWidth, e->getSubExpr());

  if (auto e = dyn_cast<PointerExpr>(ptr))
    if (dyn_cast<NullArrayRefExpr>(e->getArray()))
      return BVZExtExpr::create(bvWidth, e->getOffset());

  return new PtrToBVExpr(Type(Type::BV, bvWidth), ptr);
}

ref<Expr> SafeBVToPtrExpr::create(unsigned ptrWidth, ref<Expr> bv) {
  assert(bv->getType().isKind(Type::BV));

  if (auto e = dyn_cast<SafePtrToBVExpr>(bv))
    return e->getSubExpr();

  if (auto e = dyn_cast<BVConstExpr>(bv))
    if (e->getValue() == 0)
      return PointerExpr::create(NullArrayRefExpr::create(),
                                 BVConstExpr::createZero(ptrWidth));

  return new SafeBVToPtrExpr(Type(Type::Pointer, ptrWidth),
                             BVZExtExpr::create(ptrWidth, bv));
}

ref<Expr> SafePtrToBVExpr::create(unsigned bvWidth, ref<Expr> ptr) {
  const Type &ty = ptr->getType();
  assert(ty.isKind(Type::Pointer));

  if (auto e = dyn_cast<SafeBVToPtrExpr>(ptr))
    return BVZExtExpr::create(bvWidth, e->getSubExpr());

  if (auto e = dyn_cast<PointerExpr>(ptr))
    if (dyn_cast<NullArrayRefExpr>(e->getArray()))
      return BVZExtExpr::create(bvWidth, e->getOffset());

  return BVZExtExpr::create(bvWidth,
                            new SafePtrToBVExpr(Type(Type::BV, ty.width), ptr));
}

ref<Expr> BVToFuncPtrExpr::create(unsigned ptrWidth, ref<Expr> bv) {
  assert(bv->getType().isKind(Type::BV));

  if (auto e = dyn_cast<FuncPtrToBVExpr>(bv))
    return e->getSubExpr();

  return new BVToFuncPtrExpr(Type(Type::FunctionPointer, ptrWidth), bv);
}

ref<Expr> FuncPtrToBVExpr::create(unsigned bvWidth, ref<Expr> ptr) {
  assert(ptr->getType().isKind(Type::FunctionPointer));

  if (auto e = dyn_cast<BVToFuncPtrExpr>(ptr))
    return BVZExtExpr::create(bvWidth, e->getSubExpr());

  return new FuncPtrToBVExpr(Type(Type::BV, bvWidth), ptr);
}

ref<Expr> PtrToFuncPtrExpr::create(ref<Expr> ptr) {
  const Type &ty = ptr->getType();
  if (ty.isKind(Type::FunctionPointer))
    return ptr;

  assert(ty.isKind(Type::Pointer));

  if (auto e = dyn_cast<FuncPtrToPtrExpr>(ptr))
    return e->getSubExpr();

  return new PtrToFuncPtrExpr(Type(Type::FunctionPointer, ty.width), ptr);
}

ref<Expr> FuncPtrToPtrExpr::create(ref<Expr> ptr) {
  const Type &ty = ptr->getType();
  if (ty.isKind(Type::Pointer))
    return ptr;

  assert(ty.isKind(Type::FunctionPointer));

  if (auto e = dyn_cast<PtrToFuncPtrExpr>(ptr))
    return e->getSubExpr();

  return new FuncPtrToPtrExpr(Type(Type::Pointer, ty.width), ptr);
}

ref<Expr> BVToBoolExpr::create(ref<Expr> bv) {
  assert(bv->getType().isKind(Type::BV));
  assert(bv->getType().width == 1);

  if (auto e = dyn_cast<BoolToBVExpr>(bv))
    return e->getSubExpr();

  return new BVToBoolExpr(Type(Type::Bool), bv);
}

ref<Expr> BoolToBVExpr::create(ref<Expr> bv) {
  assert(bv->getType().isKind(Type::Bool));

  if (auto e = dyn_cast<BVToBoolExpr>(bv))
    return e->getSubExpr();

  return new BoolToBVExpr(Type(Type::BV, 1), bv);
}

ref<Expr> EqExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert((lhs->getType() == Type(Type::ArrayOf, Type::Any) &&
          rhs->getType().array) ||
         (rhs->getType() == Type(Type::ArrayOf, Type::Any) &&
          lhs->getType().array) ||
         (lhs->getType() == rhs->getType()));

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BoolConstExpr::create(e1->getValue() == e2->getValue());

  if (auto e1 = dyn_cast<BoolConstExpr>(lhs))
    if (auto e2 = dyn_cast<BoolConstExpr>(rhs))
      return BoolConstExpr::create(e1->getValue() == e2->getValue());

  if (auto e1 = dyn_cast<GlobalArrayRefExpr>(lhs))
    if (auto e2 = dyn_cast<GlobalArrayRefExpr>(rhs))
      return BoolConstExpr::create(e1->getArray() == e2->getArray());

  return new EqExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> NeExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType() == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BoolConstExpr::create(e1->getValue() != e2->getValue());

  if (auto e1 = dyn_cast<BoolConstExpr>(lhs))
    if (auto e2 = dyn_cast<BoolConstExpr>(rhs))
      return BoolConstExpr::create(e1->getValue() != e2->getValue());

  if (auto e1 = dyn_cast<GlobalArrayRefExpr>(lhs))
    if (auto e2 = dyn_cast<GlobalArrayRefExpr>(rhs))
      return BoolConstExpr::create(e1->getArray() != e2->getArray());

  return new NeExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> Expr::createNeZero(ref<Expr> bv) {
  return NeExpr::create(bv, BVConstExpr::createZero(bv->getType().width));
}

ref<Expr> AndExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::Bool) &&
         rhs->getType().isKind(Type::Bool));

  if (auto e1 = dyn_cast<BoolConstExpr>(lhs))
    return e1->getValue() ? rhs : lhs;

  if (auto e2 = dyn_cast<BoolConstExpr>(rhs))
    return e2->getValue() ? lhs : rhs;

  return new AndExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> OrExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::Bool) &&
         rhs->getType().isKind(Type::Bool));

  if (auto e1 = dyn_cast<BoolConstExpr>(lhs))
    return e1->getValue() ? lhs : rhs;

  if (auto e2 = dyn_cast<BoolConstExpr>(rhs))
    return e2->getValue() ? rhs : lhs;

  return new OrExpr(Type(Type::Bool), lhs, rhs);
}

static ref<Expr> reassociateConstAdd(BVAddExpr *nonConstOp,
                                     BVConstExpr *constOp) {
  if (auto clhs = dyn_cast<BVConstExpr>(nonConstOp->getLHS()))
    return BVAddExpr::create(nonConstOp->getRHS(),
                             BVAddExpr::create(clhs, constOp));
  if (auto crhs = dyn_cast<BVConstExpr>(nonConstOp->getRHS()))
    return BVAddExpr::create(nonConstOp->getLHS(),
                             BVAddExpr::create(crhs, constOp));
  return ref<Expr>();
}

ref<Expr> BVAddExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs)) {
    if (e1->getValue().isMinValue())
      return rhs;
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue() + e2->getValue());
    if (auto e2 = dyn_cast<BVAddExpr>(rhs)) {
      auto ca = reassociateConstAdd(e2, e1);
      if (!ca.isNull())
        return ca;
    }
  }

  if (auto e2 = dyn_cast<BVConstExpr>(rhs)) {
    if (e2->getValue().isMinValue())
      return lhs;
    if (auto e1 = dyn_cast<BVAddExpr>(lhs)) {
      auto ca = reassociateConstAdd(e1, e2);
      if (!ca.isNull())
        return ca;
    }
  }

  return new BVAddExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVSubExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue() - e2->getValue());

  if (auto e2 = dyn_cast<BVConstExpr>(rhs))
    if (e2->getValue().isMinValue())
      return lhs;

  return new BVSubExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVMulExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs)) {
    if (e1->getValue().getLimitedValue() == 1)
      return rhs;
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue() * e2->getValue());
  }

  if (auto e2 = dyn_cast<BVConstExpr>(rhs))
    if (e2->getValue().getLimitedValue() == 1)
      return lhs;

  return new BVMulExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVSDivExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      if (e2->getValue().getSExtValue() != 0)
        return BVConstExpr::create(e1->getValue().sdiv(e2->getValue()));

  return new BVSDivExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVUDivExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      if (e2->getValue().getZExtValue() != 0)
        return BVConstExpr::create(e1->getValue().udiv(e2->getValue()));

  return new BVUDivExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

static ref<Expr> createExactBVSDivMul(Expr *nonConstOp, BVConstExpr *constOp,
                                      int64_t div) {
  int64_t mul = constOp->getValue().getSExtValue();
  if (mul % div == 0) {
    return BVMulExpr::create(nonConstOp,
                             BVConstExpr::create(nonConstOp->getType().width,
                                                 (uint64_t)(mul / div)));
  }
  return ref<Expr>();
}

ref<Expr> Expr::createExactBVSDiv(ref<Expr> lhs, uint64_t rhs, Var *base) {
  assert(rhs <= INT64_MAX);
  if (rhs == 1)
    return lhs;
  if ((rhs & (rhs - 1)) != 0)
    return ref<Expr>();

  if (auto CE = dyn_cast<BVConstExpr>(lhs)) {
    int64_t val = CE->getValue().getSExtValue();
    if (val % ((int64_t)rhs) == 0)
      return BVConstExpr::create(CE->getType().width,
                                 (uint64_t)(val / ((int64_t)rhs)));
  } else if (auto AE = dyn_cast<BVAddExpr>(lhs)) {
    auto lhsDiv = createExactBVSDiv(AE->getLHS(), rhs, base);
    if (lhsDiv.isNull())
      return ref<Expr>();
    auto rhsDiv = createExactBVSDiv(AE->getRHS(), rhs, base);
    if (!rhsDiv.isNull())
      return BVAddExpr::create(lhsDiv, rhsDiv);
  } else if (auto ME = dyn_cast<BVMulExpr>(lhs)) {
    if (auto CE = dyn_cast<BVConstExpr>(ME->getLHS()))
      return createExactBVSDivMul(ME->getRHS().get(), CE, (int64_t)rhs);
    if (auto CE = dyn_cast<BVConstExpr>(ME->getRHS()))
      return createExactBVSDivMul(ME->getLHS().get(), CE, (int64_t)rhs);
  } else if (base) {
    if (auto AOE = dyn_cast<ArrayOffsetExpr>(lhs)) {
      if (auto VRE = dyn_cast<VarRefExpr>(AOE->getSubExpr())) {
        if (VRE->getVar() == base)
          return lhs;
      }
    }
  }

  return ref<Expr>();
}

ref<Expr> BVSRemExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      if (e2->getValue().getSExtValue() != 0)
        return BVConstExpr::create(e1->getValue().srem(e2->getValue()));

  return new BVSRemExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVURemExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      if (e2->getValue().getZExtValue() != 0)
        return BVConstExpr::create(e1->getValue().urem(e2->getValue()));

  return new BVURemExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVShlExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue().shl(e2->getValue()));

  return new BVShlExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVAShrExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue().ashr(e2->getValue()));

  return new BVAShrExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVLShrExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue().lshr(e2->getValue()));

  return new BVLShrExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVAndExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue() & e2->getValue());

  return new BVAndExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVOrExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue() | e2->getValue());

  return new BVOrExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVXorExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType();
  assert(lhsTy.isKind(Type::BV));
  assert(lhsTy == rhs->getType());

  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs))
      return BVConstExpr::create(e1->getValue() ^ e2->getValue());

  return new BVXorExpr(Type(Type::BV, lhsTy.width), lhs, rhs);
}

ref<Expr> BVConcatExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  auto &lhsTy = lhs->getType(), &rhsTy = rhs->getType();
  assert(lhsTy.isKind(Type::BV) && rhsTy.isKind(Type::BV));

  unsigned resWidth = lhsTy.width + rhsTy.width;
  if (auto e1 = dyn_cast<BVConstExpr>(lhs))
    if (auto e2 = dyn_cast<BVConstExpr>(rhs)) {
      llvm::APInt Tmp = e1->getValue().zext(resWidth);
      Tmp <<= rhsTy.width;
      Tmp |= e2->getValue().zext(resWidth);
      return BVConstExpr::create(Tmp);
    }

  return new BVConcatExpr(Type(Type::BV, resWidth), lhs, rhs);
}

ref<Expr> Expr::createBVConcatN(const std::vector<ref<Expr>> &exprs) {
  assert(!exprs.empty());
  return fold(exprs.back(), exprs.rbegin() + 1, exprs.rend(),
              BVConcatExpr::create);
}

#define ICMP_EXPR_CREATE(cls, method)                                          \
  ref<Expr> cls::create(ref<Expr> lhs, ref<Expr> rhs) {                        \
    assert(lhs->getType().isKind(Type::BV));                                   \
    assert(lhs->getType() == rhs->getType());                                  \
                                                                               \
    if (auto e1 = dyn_cast<BVConstExpr>(lhs))                                  \
      if (auto e2 = dyn_cast<BVConstExpr>(rhs))                                \
        return BoolConstExpr::create(e1->getValue().method(e2->getValue()));   \
                                                                               \
    return new cls(Type(Type::Bool), lhs, rhs);                                \
  }

ICMP_EXPR_CREATE(BVUgtExpr, ugt)
ICMP_EXPR_CREATE(BVUgeExpr, uge)
ICMP_EXPR_CREATE(BVUltExpr, ult)
ICMP_EXPR_CREATE(BVUleExpr, ule)
ICMP_EXPR_CREATE(BVSgtExpr, sgt)
ICMP_EXPR_CREATE(BVSgeExpr, sge)
ICMP_EXPR_CREATE(BVSltExpr, slt)
ICMP_EXPR_CREATE(BVSleExpr, sle)

ref<Expr> FAddExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FAddExpr(lhs->getType(), lhs, rhs);
}

ref<Expr> FSubExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FSubExpr(lhs->getType(), lhs, rhs);
}

ref<Expr> FMulExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FMulExpr(lhs->getType(), lhs, rhs);
}

ref<Expr> FDivExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FDivExpr(lhs->getType(), lhs, rhs);
}

ref<Expr> FPowExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FPowExpr(lhs->getType(), lhs, rhs);
}

ref<Expr> FMinExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FMinExpr(lhs->getType(), lhs, rhs);
}

ref<Expr> FPowiExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(rhs->getType().isKind(Type::BV));

  return new FPowiExpr(lhs->getType(), lhs, rhs);
}

ref<Expr> FLtExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FLtExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> FEqExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FEqExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> FUnoExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::BV));
  assert(lhs->getType() == rhs->getType());

  return new FUnoExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> Expr::createPtrLt(ref<Expr> lhs, ref<Expr> rhs) {
  return IfThenElseExpr::create(
      EqExpr::create(ArrayIdExpr::create(lhs, Type(Type::Unknown)),
                     ArrayIdExpr::create(rhs, Type(Type::Unknown))),
      BVSltExpr::create(ArrayOffsetExpr::create(lhs),
                        ArrayOffsetExpr::create(rhs)),
      PtrLtExpr::create(lhs, rhs));
}

ref<Expr> Expr::createPtrLe(ref<Expr> lhs, ref<Expr> rhs) {
  return IfThenElseExpr::create(
      EqExpr::create(ArrayIdExpr::create(lhs, Type(Type::Unknown)),
                     ArrayIdExpr::create(rhs, Type(Type::Unknown))),
      BVSleExpr::create(ArrayOffsetExpr::create(lhs),
                        ArrayOffsetExpr::create(rhs)),
      PtrLtExpr::create(lhs, rhs));
}

ref<Expr> PtrLtExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::Pointer));
  assert(rhs->getType().isKind(Type::Pointer));

  return new PtrLtExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> Expr::createFuncPtrLt(ref<Expr> lhs, ref<Expr> rhs) {
  return FuncPtrLtExpr::create(lhs, rhs);
}

ref<Expr> Expr::createFuncPtrLe(ref<Expr> lhs, ref<Expr> rhs) {
  return IfThenElseExpr::create(EqExpr::create(lhs, rhs),
                                BoolConstExpr::create(true),
                                FuncPtrLtExpr::create(lhs, rhs));
}

ref<Expr> FuncPtrLtExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::FunctionPointer));
  assert(rhs->getType().isKind(Type::FunctionPointer));

  return new FuncPtrLtExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> ImpliesExpr::create(ref<Expr> lhs, ref<Expr> rhs) {
  assert(lhs->getType().isKind(Type::Bool));
  assert(rhs->getType().isKind(Type::Bool));

  return new ImpliesExpr(Type(Type::Bool), lhs, rhs);
}

ref<Expr> CallExpr::create(Function *f, const std::vector<ref<Expr>> &args) {
  assert(f->return_begin() + 1 == f->return_end());
  return new CallExpr((*f->return_begin())->getType(), f, args);
}

ref<Expr> CallMemberOfExpr::create(ref<Expr> f, std::vector<ref<Expr>> &ces) {
  assert(f->getType().isKind(Type::FunctionPointer));
  assert(ces.size() > 0);
  Type T = (*ces.begin())->getType();
#ifndef NDEBUG
  for (auto i = ces.begin(), e = ces.end(); i != e; ++i) {
    auto CE = dyn_cast<CallExpr>(*i);
    assert(CE);
    assert(CE->getType() == T);
  }
#endif
  return new CallMemberOfExpr(T, f, ces);
}

ref<Expr> OldExpr::create(ref<Expr> op) {
  return new OldExpr(op->getType(), op);
}

ref<Expr> GetImageWidthExpr::create(ref<Expr> op) {
  return new GetImageWidthExpr(Type(Type::BV, 32), op);
}

ref<Expr> GetImageHeightExpr::create(ref<Expr> op) {
  return new GetImageHeightExpr(Type(Type::BV, 32), op);
}

ref<Expr> OtherBoolExpr::create(ref<Expr> op) {
  assert(op->getType().isKind(Type::Bool));
  return new OtherBoolExpr(Type(Type::Bool), op);
}

ref<Expr> OtherIntExpr::create(ref<Expr> op) {
  assert(op->getType().isKind(Type::BV));
  return new OtherIntExpr(Type(Type::BV, op->getType().width), op);
}

ref<Expr> OtherPtrBaseExpr::create(ref<Expr> op) {
  return new OtherPtrBaseExpr(op->getType(), op);
}

ref<Expr> AccessHasOccurredExpr::create(ref<Expr> array, bool isWrite) {
  assert(array->getType().array);
  return new AccessHasOccurredExpr(array, isWrite);
}

ref<Expr> AccessOffsetExpr::create(ref<Expr> array, unsigned pointerSize,
                                   bool isWrite) {
  assert(array->getType().array);
  return new AccessOffsetExpr(array, pointerSize, isWrite);
}

ref<Expr> ArraySnapshotExpr::create(ref<Expr> dst, ref<Expr> src) {
  assert(dst->getType().array);
  assert(src->getType().array);

  return new ArraySnapshotExpr(dst, src);
}

ref<Expr> UnderlyingArrayExpr::create(ref<Expr> array) {
  assert(array->getType().array);

  return new UnderlyingArrayExpr(array);
}

ref<Expr> AddNoovflExpr::create(ref<Expr> first, ref<Expr> second,
                                bool isSigned) {
  assert(first->getType().isKind(Type::BV));
  assert(second->getType().isKind(Type::BV));
  assert(first->getType().width == second->getType().width);

  return new AddNoovflExpr(first, second, isSigned);
}

ref<Expr> AddNoovflPredicateExpr::create(const std::vector<ref<Expr>> &exprs) {
  assert(!exprs.empty());
#ifndef NDEBUG
  for (auto i = exprs.begin(), e = exprs.end(); i != e; ++i) {
    assert((*i)->getType().isKind(Type::BV));
    assert((*i)->getType().width == exprs[0]->getType().width);
  }
#endif

  return new AddNoovflPredicateExpr(exprs);
}

ref<Expr>
UninterpretedFunctionExpr::create(const std::string &name, Type returnType,
                                  const std::vector<ref<Expr>> &args) {
  return new UninterpretedFunctionExpr(name, returnType, args);
}

ref<Expr> AtomicHasTakenValueExpr::create(ref<Expr> atomicArray,
                                          ref<Expr> offset, ref<Expr> value) {
  assert(atomicArray->getType().array);
  assert(offset->getType().isKind(Type::BV));
  assert(value->getType().isKind(Type::BV));

  return new AtomicHasTakenValueExpr(atomicArray, offset, value);
}

ref<Expr> AsyncWorkGroupCopyExpr::create(ref<Expr> dst, ref<Expr> dstOffset,
                                         ref<Expr> src, ref<Expr> srcOffset,
                                         ref<Expr> size, ref<Expr> handle) {
  assert(dst->getType().array);
  assert(dstOffset->getType().isKind(Type::BV));
  assert(src->getType().array);
  assert(srcOffset->getType().isKind(Type::BV));
  assert(size->getType().isKind(Type::BV));
  assert(handle->getType().isKind(Type::BV));
  return new AsyncWorkGroupCopyExpr(dst, dstOffset, src, srcOffset, size,
                                    handle);
}
