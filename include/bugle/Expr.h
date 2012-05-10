#include "bugle/Ref.h"
#include "bugle/Type.h"
#include "bugle/Var.h"
#include "llvm/ADT/APInt.h"
#include <vector>

#ifndef BUGLE_EXPR_H
#define BUGLE_EXPR_H

namespace llvm {
 
class Value;

}

namespace bugle {

class Array;
class Function;
class Var;

class Expr {
public:
  enum Kind {
    BVConst,
    Arg,
    ArrayRef,
    Pointer,
    Phi,
    Call,

    // Unary
    ArrayId,
    ArrayOffset,
    BVToFloat,
    FloatToBV,
    BVToPtr,
    PtrToBV,

    UnaryFirst = ArrayId,
    UnaryLast = PtrToBV,

    // Binary
    BVAdd,

    BinaryFirst = BVAdd,
    BinaryLast = BVAdd
  };

  unsigned refCount;

private:
  Type type;

protected:
  Expr(Type type) : type(type) {}

public:
  virtual ~Expr() {}
  virtual Kind getKind() const = 0;
  const Type &getType() const { return type; }

  static bool classof(const Expr *) { return true; }
};

#define EXPR_KIND(kind) \
  Kind getKind() const { return kind; } \
  static bool classof(const Expr *E) { return E->getKind() == kind; } \
  static bool classof(const kind##Expr *) { return true; }

class BVConstExpr : public Expr {
  BVConstExpr(const llvm::APInt &bv) :
    Expr(Type(Type::BV, bv.getBitWidth())), bv(bv) {}
  llvm::APInt bv;

public:
  static ref<Expr> create(const llvm::APInt &bv);
  static ref<Expr> createZero(unsigned width);

  EXPR_KIND(BVConst)
  const llvm::APInt &getValue() const { return bv; }
};

class ArgExpr : public Expr {
  ArgExpr(Type type) : Expr(type) {}

public:
  static ref<Expr> create(Type type);

  EXPR_KIND(Arg)
};

class ArrayRefExpr : public Expr {
  ArrayRefExpr(llvm::Value *array) : Expr(Type(Type::ArrayId)), array(array) {}
  llvm::Value *array;

public:
  static ref<Expr> create(llvm::Value *array);

  EXPR_KIND(ArrayRef)
};

class PointerExpr : public Expr {
  PointerExpr(ref<Expr> array, ref<Expr> offset) :
    Expr(Type(Type::Pointer, offset->getType().width)),
    array(array), offset(offset) {}
  ref<Expr> array, offset;

public:
  static ref<Expr> create(ref<Expr> array, ref<Expr> offset);

  EXPR_KIND(Pointer)
  ref<Expr> getArray() const { return array; }
  ref<Expr> getOffset() const { return offset; }
};

class PhiExpr : public Expr {
  Var *var;
  PhiExpr(Var *var) : Expr(var->getType()), var(var) {}

public:
  static ref<Expr> create(Var *var);
  EXPR_KIND(Phi)
};

class UnaryExpr : public Expr {
  ref<Expr> expr;

protected:
  UnaryExpr(Type type, ref<Expr> expr) :
    Expr(type), expr(expr) {}

public:
  ref<Expr> getSubExpr() const { return expr; }
};

#define UNARY_EXPR(kind) \
  class kind##Expr : public UnaryExpr { \
    kind##Expr(Type type, ref<Expr> expr) : UnaryExpr(type, expr) {} \
\
  public: \
    static ref<Expr> create(ref<Expr> var); \
    EXPR_KIND(kind) \
  };

UNARY_EXPR(ArrayId)
UNARY_EXPR(ArrayOffset)
UNARY_EXPR(BVToFloat)
UNARY_EXPR(FloatToBV)
UNARY_EXPR(BVToPtr)
UNARY_EXPR(PtrToBV)

#undef UNARY_EXPR

class BinaryExpr : public Expr {
  ref<Expr> lhs, rhs;

protected:
  BinaryExpr(Type type, ref<Expr> lhs, ref<Expr> rhs) :
    Expr(type), lhs(lhs), rhs(rhs) {}

public:
  ref<Expr> getLHS() const { return lhs; }
  ref<Expr> getRHS() const { return rhs; }
};

#define BINARY_EXPR(kind) \
  class kind##Expr : public BinaryExpr { \
    kind##Expr(Type type, ref<Expr> lhs, ref<Expr> rhs) : \
      BinaryExpr(type, lhs, rhs) {} \
\
  public: \
    static ref<Expr> create(ref<Expr> lhs, ref<Expr> rhs); \
    EXPR_KIND(kind) \
  };

BINARY_EXPR(BVAdd)

#undef BINARY_EXPR

class CallExpr : public Expr {
  Function *callee;
  std::vector<ref<Expr> > args;

public:
  EXPR_KIND(Call)
};

}

#undef EXPR_KIND

#endif
