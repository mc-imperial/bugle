#include "bugle/Stmt.h"
#include "bugle/GlobalArray.h"
#include "bugle/Ref.h"

using namespace bugle;

void VarAssignStmt::check() {
  assert(!vars.empty() && vars.size() == values.size());
#ifndef NDEBUG
       auto li = vars.begin(), le = vars.end();
  for (auto ri = values.begin(); li != le; ++li, ++ri) {
    assert((*li)->getType() == (*ri)->getType());
  }
#endif
}

StoreStmt::StoreStmt(ref<Expr> array, ref<Expr> offset, ref<Expr> value) :
    array(array), offset(offset), value(value) {
  assert(array->getType().kind == Type::ArrayId);
  assert(offset->getType().kind == Type::BV);
  if (auto GARE = dyn_cast<GlobalArrayRefExpr>(array))
    assert(value->getType() == GARE->getArray()->getRangeType());
  else
    assert(value->getType() == Type(Type::BV, 8));
}
