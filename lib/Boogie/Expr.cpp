#include "bugle/Expr.h"
#include "llvm/Value.h"

using namespace bugle;

ref<Expr> ArrayRefExpr::create(llvm::Value *v) {
  return new ArrayRefExpr(v);
}

ref<Expr> PointerExpr::create(ref<Expr> array, ref<Expr> offset) {
  return new PointerExpr(array, offset);
}

ref<Expr> PhiExpr::create(Var *var) {
  return new PhiExpr(var);
}

ref<Expr> ArrayIdExpr::create(ref<Expr> pointer) {
  if (PointerExpr *e = dyn_cast<PointerExpr>(pointer))
    return e->getArray();

  return new ArrayIdExpr(Type(Type::ArrayId), pointer);
}

ref<Expr> ArrayOffsetExpr::create(ref<Expr> pointer) {
  if (PointerExpr *e = dyn_cast<PointerExpr>(pointer))
    return e->getOffset();

  return new ArrayOffsetExpr(Type(Type::BV, pointer->getType().width), pointer);
}
