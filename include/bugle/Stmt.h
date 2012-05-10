#include "bugle/Ref.h"

#ifndef BUGLE_STMT_H
#define BUGLE_STMT_H

namespace bugle {

class BasicBlock;
class Expr;
class Var;

class Stmt {
};

class ExprStmt : public Stmt {
  ref<Var> var;
  ref<Expr> expr;
};

class ArrayWriteStmt : public Stmt {
  ref<Expr> array;
  ref<Expr> value;
};

class GotoStmt : public Stmt {
  std::vector<BasicBlock *> blocks;
};

class ReturnStmt : public Stmt {
  ref<Expr> value;
};

}

#endif
