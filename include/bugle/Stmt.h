#include "bugle/Ref.h"

namespace bugle {

class Expr;

class Stmt {
};

class ExprStmt : public Stmt {
  ref<Expr> expr;
};

class ArrayWriteStmt : public Stmt {
  ref<Expr> array;
  ref<Expr> value;
};

class GotoStmt : public Stmt {
};

}
