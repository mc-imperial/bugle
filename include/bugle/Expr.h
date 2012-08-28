#include "bugle/Ref.h"
#include "bugle/Type.h"
#include "bugle/Var.h"
#include "llvm/ADT/APInt.h"
#include <set>
#include <vector>

#ifndef BUGLE_EXPR_H
#define BUGLE_EXPR_H

namespace llvm {
 
class Value;

}

namespace bugle {

class Function;
class GlobalArray;
class Var;

class Expr {
public:
  enum Kind {
    BVConst,
    BoolConst,
    GlobalArrayRef,
    NullArrayRef,
    ConstantArrayRef,
    Pointer,
    Load,
    VarRef,
    SpecialVarRef,
    Call,
    BVExtract,
    IfThenElse,
    AccessHasOccurred,
    AccessOffset,
    MemberOf,

    // Unary
    Not,
    ArrayId,
    ArrayOffset,
    BVToFloat,
    FloatToBV,
    BVToPtr,
    PtrToBV,
    BVToBool,
    BoolToBV,
    BVZExt,
    BVSExt,
    FPConv,
    FPToSI,
    FPToUI,
    SIToFP,
    UIToFP,
    FAbs,
    FCos,
    FExp,
    FFloor,
    FLog,
    FrexpExp,
    FrexpFrac,
    FSin,
    FSqrt,
    OtherInt,
    OtherBool,
    OtherPtrBase,
    Old,

    UnaryFirst = Not,
    UnaryLast = Old,

    // Binary
    Eq,
    Ne,
    And,
    Or,
    BVAdd,
    BVSub,
    BVMul,
    BVSDiv,
    BVUDiv,
    BVSRem,
    BVURem,
    BVShl,
    BVAShr,
    BVLShr,
    BVAnd,
    BVOr,
    BVXor,
    BVConcat,
    BVUgt,
    BVUge,
    BVUlt,
    BVUle,
    BVSgt,
    BVSge,
    BVSlt,
    BVSle,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FPow,
    FLt,
    FEq,
    FUno,
    PtrLt,
    PtrLe,
	Implies,

    BinaryFirst = Eq,
    BinaryLast = Implies
  };

  unsigned refCount;
  bool isDerivedFromConstant:1, hasEvalStmt:1;

  static ref<Expr> createPtrLt(ref<Expr> lhs, ref<Expr> rhs);
  static ref<Expr> createPtrLe(ref<Expr> lhs, ref<Expr> rhs);
  static ref<Expr> createBVConcatN(const std::vector<ref<Expr>> &args);
  static ref<Expr> createNeZero(ref<Expr> bv);
  static ref<Expr> createExactBVUDiv(ref<Expr> lhs, uint64_t rhs,
                                     Var *base = 0);

  static Type getArrayCandidateType(const std::set<GlobalArray *> &Globals);
  bool computeArrayCandidates(std::set<GlobalArray *> &GlobalSet) const;

private:
  Type type;

protected:
  Expr(Type type) : refCount(0), isDerivedFromConstant(false),
                    hasEvalStmt(false), type(type) {}

public:
  virtual ~Expr() {}
  virtual Kind getKind() const = 0;
  const Type &getType() const { return type; }

  void print(llvm::raw_ostream &OS);
  void dump();

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
  static ref<Expr> create(unsigned width, uint64_t val, bool isSigned = false);
  static ref<Expr> createZero(unsigned width);

  EXPR_KIND(BVConst)
  const llvm::APInt &getValue() const { return bv; }
};

class BoolConstExpr : public Expr {
  BoolConstExpr(bool val) : Expr(Type(Type::Bool)), val(val) {}
  bool val;

public:
  static ref<Expr> create(bool val);

  EXPR_KIND(BoolConst)
  bool getValue() const { return val; }
};

class GlobalArrayRefExpr : public Expr {
  GlobalArrayRefExpr(Type t, GlobalArray *array) :
    Expr(t), array(array) {}
  GlobalArray *array;

public:
  static ref<Expr> create(GlobalArray *array);

  EXPR_KIND(GlobalArrayRef)
  GlobalArray *getArray() const { return array; }
};

class NullArrayRefExpr : public Expr {
  NullArrayRefExpr() : Expr(Type(Type::ArrayOf, Type::Any)) {}

public:
  static ref<Expr> create();

  EXPR_KIND(NullArrayRef)
};

class ConstantArrayRefExpr : public Expr {
  ConstantArrayRefExpr(llvm::ArrayRef<ref<Expr>> array) :
    Expr(Type(Type::ArrayOf, array[0]->getType())),
    array(array.begin(), array.end()) {}
  std::vector<ref<Expr>> array;

public:
  static ref<Expr> create(llvm::ArrayRef<ref<Expr>> array);

  EXPR_KIND(ConstantArrayRef)
  const std::vector<ref<Expr>> &getArray() const { return array; }
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

class LoadExpr : public Expr {
  LoadExpr(Type t, ref<Expr> array, ref<Expr> offset) :
    Expr(t), array(array), offset(offset) {}
  ref<Expr> array, offset;

public:
  static ref<Expr> create(ref<Expr> array, ref<Expr> offset);

  EXPR_KIND(Load)
  ref<Expr> getArray() const { return array; }
  ref<Expr> getOffset() const { return offset; }
};

/// Local variable reference.  Used for phi nodes, parameters and return
/// variables.
class VarRefExpr : public Expr {
  Var *var;
  VarRefExpr(Var *var) : Expr(var->getType()), var(var) {}

public:
  static ref<Expr> create(Var *var);
  EXPR_KIND(VarRef)
  Var *getVar() const { return var; }
};

/// A reference to the special variable marked with the given attribute.
class SpecialVarRefExpr : public Expr {
  std::string attr;
  SpecialVarRefExpr(Type t, const std::string &attr) : Expr(t), attr(attr) {}

public:
  static ref<Expr> create(Type t, const std::string &attr);
  EXPR_KIND(SpecialVarRef)
  const std::string &getAttr() const { return attr; }
};

class BVExtractExpr : public Expr {
  BVExtractExpr(ref<Expr> expr, unsigned offset, unsigned width) :
    Expr(Type(Type::BV, width)), expr(expr), offset(offset) {}
  ref<Expr> expr;
  unsigned offset;

public:
  static ref<Expr> create(ref<Expr> expr, unsigned offset, unsigned width);

  EXPR_KIND(BVExtract)
  ref<Expr> getSubExpr() const { return expr; }
  unsigned getOffset() const { return offset; }
};

class IfThenElseExpr : public Expr {
  IfThenElseExpr(ref<Expr> cond, ref<Expr> trueExpr, ref<Expr> falseExpr) :
    Expr(trueExpr->getType()), cond(cond), trueExpr(trueExpr),
    falseExpr(falseExpr) {}
  ref<Expr> cond, trueExpr, falseExpr;

public:
  static ref<Expr> create(ref<Expr> cond, ref<Expr> trueExpr,
                          ref<Expr> falseExpr);
  EXPR_KIND(IfThenElse)
  ref<Expr> getCond() const { return cond; }
  ref<Expr> getTrueExpr() const { return trueExpr; }
  ref<Expr> getFalseExpr() const { return falseExpr; }
};

/// Expression which denotes that its subexpression is an arrayId and a member
/// of the elems set.  This is an unusual expression in that it only shows
/// up in the output indirectly via case splits.
class MemberOfExpr : public Expr {
  MemberOfExpr(Type t, ref<Expr> expr, const std::set<GlobalArray *> &elems) :
    Expr(t), expr(expr), elems(elems) {}
  ref<Expr> expr;
  std::set<GlobalArray *> elems;

public:
  static ref<Expr> create(ref<Expr> expr, const std::set<GlobalArray *> &elems);

  EXPR_KIND(MemberOf)
  ref<Expr> getSubExpr() const { return expr; }
  const std::set<GlobalArray *> &getElems() const { return elems; }
};

class UnaryExpr : public Expr {
  ref<Expr> expr;

protected:
  UnaryExpr(Type type, ref<Expr> expr) :
    Expr(type), expr(expr) {}

public:
  ref<Expr> getSubExpr() const { return expr; }
  static bool classof(const Expr *E) {
    Kind k = E->getKind();
    return k >= UnaryFirst && k <= UnaryLast;
  }
  static bool classof(const UnaryExpr *) { return true; }
};

#define UNARY_EXPR(kind) \
  class kind##Expr : public UnaryExpr { \
    kind##Expr(Type type, ref<Expr> expr) : UnaryExpr(type, expr) {} \
\
  public: \
    static ref<Expr> create(ref<Expr> var); \
    EXPR_KIND(kind) \
  };

UNARY_EXPR(Not)

class ArrayIdExpr : public UnaryExpr {
  ArrayIdExpr(Type type, ref<Expr> expr) : UnaryExpr(type, expr) {}

public:
  static ref<Expr> create(ref<Expr> var, Type defaultRange);
  EXPR_KIND(ArrayId)
};

UNARY_EXPR(ArrayOffset)
UNARY_EXPR(BVToFloat)
UNARY_EXPR(FloatToBV)
UNARY_EXPR(BVToPtr)
UNARY_EXPR(PtrToBV)
UNARY_EXPR(BVToBool)
UNARY_EXPR(BoolToBV)
UNARY_EXPR(FAbs)
UNARY_EXPR(FCos)
UNARY_EXPR(FExp)
UNARY_EXPR(FFloor)
UNARY_EXPR(FLog)
UNARY_EXPR(FrexpFrac)
UNARY_EXPR(FSin)
UNARY_EXPR(FSqrt)
UNARY_EXPR(OtherInt)
UNARY_EXPR(OtherBool)
UNARY_EXPR(OtherPtrBase)
UNARY_EXPR(Old)

#define UNARY_CONV_EXPR(kind) \
  class kind##Expr : public UnaryExpr { \
    kind##Expr(Type type, ref<Expr> expr) : UnaryExpr(type, expr) {} \
\
  public: \
    static ref<Expr> create(unsigned width, ref<Expr> var); \
    EXPR_KIND(kind) \
  };

UNARY_CONV_EXPR(BVSExt)
UNARY_CONV_EXPR(BVZExt)
UNARY_CONV_EXPR(FPConv)
UNARY_CONV_EXPR(FPToSI)
UNARY_CONV_EXPR(FPToUI)
UNARY_CONV_EXPR(SIToFP)
UNARY_CONV_EXPR(UIToFP)
UNARY_CONV_EXPR(FrexpExp)

#undef UNARY_EXPR
#undef UNARY_CONV_EXPR

class BinaryExpr : public Expr {
  ref<Expr> lhs, rhs;

protected:
  BinaryExpr(Type type, ref<Expr> lhs, ref<Expr> rhs) :
    Expr(type), lhs(lhs), rhs(rhs) {}

public:
  ref<Expr> getLHS() const { return lhs; }
  ref<Expr> getRHS() const { return rhs; }
  static bool classof(const Expr *E) {
    Kind k = E->getKind();
    return k >= BinaryFirst && k <= BinaryLast;
  }
  static bool classof(const BinaryExpr *) { return true; }
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

BINARY_EXPR(Eq)
BINARY_EXPR(Ne)
BINARY_EXPR(And)
BINARY_EXPR(Or)
BINARY_EXPR(BVAdd)
BINARY_EXPR(BVSub)
BINARY_EXPR(BVMul)
BINARY_EXPR(BVSDiv)
BINARY_EXPR(BVUDiv)
BINARY_EXPR(BVSRem)
BINARY_EXPR(BVURem)
BINARY_EXPR(BVShl)
BINARY_EXPR(BVAShr)
BINARY_EXPR(BVLShr)
BINARY_EXPR(BVAnd)
BINARY_EXPR(BVOr)
BINARY_EXPR(BVXor)
BINARY_EXPR(BVConcat)
BINARY_EXPR(BVUgt)
BINARY_EXPR(BVUge)
BINARY_EXPR(BVUlt)
BINARY_EXPR(BVUle)
BINARY_EXPR(BVSgt)
BINARY_EXPR(BVSge)
BINARY_EXPR(BVSlt)
BINARY_EXPR(BVSle)
BINARY_EXPR(FAdd)
BINARY_EXPR(FSub)
BINARY_EXPR(FMul)
BINARY_EXPR(FDiv)
BINARY_EXPR(FPow)
BINARY_EXPR(FLt)
BINARY_EXPR(FEq)
BINARY_EXPR(FUno)
BINARY_EXPR(PtrLt)
BINARY_EXPR(PtrLe)
BINARY_EXPR(Implies)

#undef BINARY_EXPR

class CallExpr : public Expr {
  Function *callee;
  std::vector<ref<Expr>> args;
  CallExpr(Type t, Function *callee, const std::vector<ref<Expr>> &args) :
    Expr(t), callee(callee), args(args) {}

public:
  static ref<Expr> create(Function *callee, const std::vector<ref<Expr>> &args);

  EXPR_KIND(Call)
  Function *getCallee() const { return callee; }
  const std::vector<ref<Expr>> &getArgs() const { return args; }
};

class AccessHasOccurredExpr : public Expr {
  AccessHasOccurredExpr(ref<Expr> array, bool isWrite) :
    Expr(Type::Bool), array(array), isWrite(isWrite) {}
  ref<Expr> array;
  bool isWrite;

public:
  static ref<Expr> create(ref<Expr> array, bool isWrite);

  EXPR_KIND(AccessHasOccurred)
  ref<Expr> getArray() const { return array; }
  std::string getAccessKind() { return isWrite ? "WRITE" : "READ"; }
};

class AccessOffsetExpr : public Expr {
  AccessOffsetExpr(ref<Expr> array, bool isWrite) :
    Expr(Type(Type::BV, 32)), array(array), isWrite(isWrite) {}
  ref<Expr> array;
  bool isWrite;

public:
  static ref<Expr> create(ref<Expr> array, bool isWrite);

  EXPR_KIND(AccessOffset)
  ref<Expr> getArray() const { return array; }
  std::string getAccessKind() { return isWrite ? "WRITE" : "READ"; }
};

}

#undef EXPR_KIND

#endif
