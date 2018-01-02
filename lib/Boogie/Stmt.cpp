#include "bugle/Stmt.h"
#include "bugle/Expr.h"
#include "bugle/GlobalArray.h"
#include "bugle/Ref.h"

using namespace bugle;

EvalStmt *EvalStmt::create(ref<Expr> expr, const SourceLocsRef &sourcelocs) {
  assert(!expr->hasEvalStmt);
  expr->hasEvalStmt = true;
  return new EvalStmt(expr, sourcelocs);
}

EvalStmt::~EvalStmt() { expr->hasEvalStmt = false; }

StoreStmt *StoreStmt::create(ref<Expr> array, ref<Expr> offset, ref<Expr> value,
                             const SourceLocsRef &sourcelocs) {
  assert(array->getType().array);
  assert(offset->getType().isKind(Type::BV));
  assert(array->getType().kind == Type::Any ||
         value->getType().isKind(array->getType().kind));
  assert(array->getType().kind == Type::Any ||
         value->getType().width == array->getType().width);
  return new StoreStmt(array, offset, value, sourcelocs);
}

VarAssignStmt *VarAssignStmt::create(Var *var, ref<Expr> value) {
  assert(var->getType() == value->getType());
  std::vector<Var *> vars(1, var);
  std::vector<ref<Expr>> values(1, value);
  return new VarAssignStmt(vars, values);
}

VarAssignStmt *VarAssignStmt::create(const std::vector<Var *> &vars,
                                     const std::vector<ref<Expr>> &values) {
  assert(!vars.empty() && vars.size() == values.size());
#ifndef NDEBUG
  auto li = vars.begin(), le = vars.end();
  for (auto ri = values.begin(); li != le; ++li, ++ri) {
    assert((*li)->getType() == (*ri)->getType());
  }
#endif
  return new VarAssignStmt(vars, values);
}

GotoStmt *GotoStmt::create(BasicBlock *block) {
  std::vector<BasicBlock *> blocks(1, block);
  return new GotoStmt(blocks);
}

GotoStmt *GotoStmt::create(const std::vector<BasicBlock *> &blocks) {
  return new GotoStmt(blocks);
}

ReturnStmt *ReturnStmt::create() { return new ReturnStmt(); }

AssumeStmt *AssumeStmt::create(ref<Expr> pred) {
  return new AssumeStmt(pred, false);
}

AssumeStmt *AssumeStmt::createPartition(ref<Expr> pred) {
  return new AssumeStmt(pred, true);
}

AssertStmt *AssertStmt::create(ref<Expr> pred, bool global, bool candidate,
                               const SourceLocsRef &sourcelocs) {
  AssertStmt *AS = new AssertStmt(pred, sourcelocs);
  AS->global = global;
  AS->candidate = candidate;
  return AS;
}

AssertStmt *AssertStmt::createInvariant(ref<Expr> pred, bool global,
                                        bool candidate,
                                        const SourceLocsRef &sourcelocs) {
  AssertStmt *AS = new AssertStmt(pred, sourcelocs);
  AS->invariant = true;
  AS->global = global;
  AS->candidate = candidate;
  return AS;
}

AssertStmt *AssertStmt::createBadAccess(const SourceLocsRef &sourcelocs) {
  AssertStmt *AS = new AssertStmt(BoolConstExpr::create(false), sourcelocs);
  AS->badAccess = true;
  return AS;
}

AssertStmt *AssertStmt::createBlockSourceLoc(const SourceLocsRef &sourcelocs) {
  AssertStmt *AS = new AssertStmt(BoolConstExpr::create(true), sourcelocs);
  AS->blockSourceLoc = true;
  return AS;
}

CallStmt *CallStmt::create(Function *callee, const std::vector<ref<Expr>> &args,
                           const SourceLocsRef &sourcelocs) {
  return new CallStmt(callee, args, sourcelocs);
}

CallMemberOfStmt *CallMemberOfStmt::create(ref<Expr> func,
                                           std::vector<Stmt *> &callStmts,
                                           const SourceLocsRef &sourcelocs) {
  assert(std::all_of(callStmts.begin(), callStmts.end(),
                     [](Stmt *S) { return isa<CallStmt>(S); }));
  return new CallMemberOfStmt(func, callStmts, sourcelocs);
}

WaitGroupEventStmt *
WaitGroupEventStmt::create(ref<Expr> handle, const SourceLocsRef &sourcelocs) {
  assert(handle->getType().isKind(Type::BV));
  return new WaitGroupEventStmt(handle, sourcelocs);
}
