#include "bugle/Expr.h"
#include "bugle/Ref.h"

#ifndef BUGLE_STMT_H
#define BUGLE_STMT_H

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
    Return
  };

  virtual ~Stmt() {}
  virtual Kind getKind() const = 0;
};

#define STMT_KIND(kind) \
  Kind getKind() const { return kind; } \
  static bool classof(const Stmt *S) { return S->getKind() == kind; } \
  static bool classof(const kind##Stmt *) { return true; }

class EvalStmt : public Stmt {
  ref<Expr> expr;

public:
  EvalStmt(ref<Expr> expr) : expr(expr) {}

  STMT_KIND(Eval)
  ref<Expr> getExpr() const { return expr; }
};

class StoreStmt : public Stmt {
  ref<Expr> pointer;
  ref<Expr> value;

public:
  StoreStmt(ref<Expr> pointer, ref<Expr> value) :
    pointer(pointer), value(value) {}

  STMT_KIND(Store)
  ref<Expr> getPointer() const { return pointer; }
  ref<Expr> getValue() const { return value; }
};

class VarAssignStmt : public Stmt {
  Var *var;
  ref<Expr> value;

public:
  VarAssignStmt(Var *var, ref<Expr> value) : var(var), value(value) {}

  STMT_KIND(VarAssign)
  Var *getVar() const { return var; }
  ref<Expr> getValue() const { return value; }
};

class GotoStmt : public Stmt {
  std::vector<BasicBlock *> blocks;

public:
  STMT_KIND(Goto)
  const std::vector<BasicBlock *> &getBlocks() { return blocks; }
};

class ReturnStmt : public Stmt {
  ref<Expr> value;

public:
  ReturnStmt() {}
  STMT_KIND(Return)
};

}

#endif
