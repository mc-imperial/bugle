#ifndef BUGLE_STMT_H
#define BUGLE_STMT_H

#include "bugle/Expr.h"
#include "bugle/Ref.h"
#include "bugle/SourceLoc.h"

namespace bugle {

class BasicBlock;
class Expr;
class Var;

class Stmt {
public:
  enum Kind {
    Eval,
    Store,
    VarAssign,
    Goto,
    Return,
    Assume,
    Assert,
    Call,
    CallMemberOf
  };

  virtual ~Stmt() {}
  virtual Kind getKind() const = 0;
  void setSourceLocs(const SourceLocsRef &sourcelocs) {
    Stmt::sourcelocs = sourcelocs;
  }
  SourceLocsRef &getSourceLocs() { return sourcelocs; }

private:
  SourceLocsRef sourcelocs;
};

#define STMT_KIND(kind)                                                        \
  Kind getKind() const { return kind; }                                        \
  static bool classof(const Stmt *S) { return S->getKind() == kind; }          \
  static bool classof(const kind##Stmt *) { return true; }

class EvalStmt : public Stmt {
  ref<Expr> expr;

public:
  EvalStmt(ref<Expr> expr) : expr(expr) {
    assert(!expr->hasEvalStmt);
    expr->hasEvalStmt = true;
  }
  ~EvalStmt() { expr->hasEvalStmt = false; }

  STMT_KIND(Eval)
  ref<Expr> getExpr() const { return expr; }
};

class StoreStmt : public Stmt {
  ref<Expr> array;
  ref<Expr> offset;
  ref<Expr> value;

public:
  StoreStmt(ref<Expr> array, ref<Expr> offset, ref<Expr> value);

  STMT_KIND(Store)
  ref<Expr> getArray() const { return array; }
  ref<Expr> getOffset() const { return offset; }
  ref<Expr> getValue() const { return value; }
};

class VarAssignStmt : public Stmt {
  std::vector<Var *> vars;
  std::vector<ref<Expr>> values;

  void check();

public:
  VarAssignStmt(Var *var, ref<Expr> value) : vars(1, var), values(1, value) {
    check();
  }
  VarAssignStmt(const std::vector<Var *> &vars,
                const std::vector<ref<Expr>> &values)
      : vars(vars), values(values) {
    check();
  }

  STMT_KIND(VarAssign)
  const std::vector<Var *> &getVars() const { return vars; }
  const std::vector<ref<Expr>> &getValues() const { return values; }
};

class GotoStmt : public Stmt {
  std::vector<BasicBlock *> blocks;

public:
  GotoStmt(const std::vector<BasicBlock *> &blocks) : blocks(blocks) {}
  GotoStmt(BasicBlock *block) : blocks(1, block) {}
  STMT_KIND(Goto)
  const std::vector<BasicBlock *> &getBlocks() { return blocks; }
};

class ReturnStmt : public Stmt {
public:
  ReturnStmt() {}
  STMT_KIND(Return)
};

class AssumeStmt : public Stmt {
  ref<Expr> pred;
  bool partition;

public:
  AssumeStmt(ref<Expr> pred, bool partition = false)
      : pred(pred), partition(partition) {}

  STMT_KIND(Assume)
  ref<Expr> getPredicate() const { return pred; }
  bool isPartition() const { return partition; }
};

class AssertStmt : public Stmt {
  ref<Expr> pred;
  bool global;
  bool candidate;
  bool invariant;

public:
  AssertStmt(ref<Expr> pred, bool global, bool candidate, bool invariant)
      : pred(pred), global(global), candidate(candidate), invariant(invariant) {
  }

  STMT_KIND(Assert)
  ref<Expr> getPredicate() const { return pred; }
  bool isGlobal() const { return global; }
  bool isCandidate() const { return candidate; }
  bool isInvariant() const { return invariant; }
};

class CallStmt : public Stmt {
  Function *callee;
  std::vector<ref<Expr>> args;

public:
  CallStmt(Function *callee, const std::vector<ref<Expr>> &args)
      : callee(callee), args(args) {}

  STMT_KIND(Call)
  Function *getCallee() const { return callee; }
  const std::vector<ref<Expr>> &getArgs() const { return args; }
};

class CallMemberOfStmt : public Stmt {
  ref<Expr> func;
  std::vector<Stmt *> callStmts;

public:
  CallMemberOfStmt(ref<Expr> func, std::vector<Stmt *> &callStmts);

  STMT_KIND(CallMemberOf)
  ref<Expr> getFunc() const { return func; }
  std::vector<Stmt *> getCallStmts() const { return callStmts; }
};
}

#endif
