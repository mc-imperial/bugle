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
  assert(array->getType().array);
  assert(offset->getType().isKind(Type::BV));
  assert(array->getType().kind == Type::Any || value->getType().isKind(array->getType().kind));
  assert(array->getType().kind == Type::Any || value->getType().width == array->getType().width);
}

CallMemberOfStmt::CallMemberOfStmt(ref<Expr> func,
                                   std::vector<Stmt *> &callStmts) :
    func(func), callStmts(callStmts) {
#ifndef NDEBUG
  for (auto i = callStmts.begin(), e = callStmts.end(); i != e; ++i)
    assert(isa<CallStmt>(*i));
#endif
}
