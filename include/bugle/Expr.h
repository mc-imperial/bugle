#include "bugle/Ref.h"

namespace bugle {

class Array;
class Function;
class Var;

class Expr {
  enum {
    ArrayPtr,
    Phi,
    Binary
  } Kind;

  enum {
    BV,
    Float,
    Pointer
  } Type;
};

class ArrayPtrExpr : public Expr {
  ref<Array> array;
  ref<Expr> offset;

public:

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
