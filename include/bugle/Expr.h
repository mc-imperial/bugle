#include "bugle/Ref.h"
#include "bugle/Type.h"

namespace bugle {

class Array;
class Function;
class Var;

class Expr {
  enum {
    Array,
    ArrayId,
    ArrayOffset,
    Phi,
    Unary,
    Binary
  } Kind;

  Type type;
};

class ArrayExpr : public Expr {
  ref<Expr> array;
  ref<Expr> offset;

public:

};

class ArrayIdExpr : public Expr {
  ref<Expr> array;
};

class ArrayOffsetExpr : public Expr {
  ref<Expr> array;
};

class PhiExpr : public Expr {
  ref<Var> var;
};

class UnaryExpr : public Expr {
  ref<Expr> expr;
};

class BinaryExpr : public Expr {
  ref<Expr> lhs, rhs;
};

class CallExpr : public Expr {
  ref<Function> callee;
  std::vector<ref<Expr> > args;
};

}

};
