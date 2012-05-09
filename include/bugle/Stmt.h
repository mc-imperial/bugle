#include "bugle/Ref.h"

namespace bugle {

class Expr;
class BasicBlock;

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
