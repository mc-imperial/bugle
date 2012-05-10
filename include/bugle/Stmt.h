#include "bugle/Ref.h"

#ifndef BUGLE_STMT_H
#define BUGLE_STMT_H

namespace bugle {

class BasicBlock;
class Expr;
class Var;

class Stmt {
public:
  virtual ~Stmt() {}
};

class ExprStmt : public Stmt {
  ref<Expr> expr;

public:
  ExprStmt(ref<Expr> expr) : expr(expr) {}
};

class StoreStmt : public Stmt {
  ref<Expr> pointer;
  ref<Expr> value;

public:
  StoreStmt(ref<Expr> pointer, ref<Expr> value) :
    pointer(pointer), value(value) {}
};

class GotoStmt : public Stmt {
  std::vector<BasicBlock *> blocks;
};

class ReturnStmt : public Stmt {
  ref<Expr> value;

public:
  ReturnStmt(ref<Expr> value) : value(value) {}
};

}

#endif
