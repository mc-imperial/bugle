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
    NullFunctionPointer,
    FunctionPointer,
    Load,
    Atomic,
    VarRef,
    SpecialVarRef,
    Call,
    CallMemberOf,
    BVExtract,
    BVCtlz,
    IfThenElse,
    Havoc,
    AccessHasOccurred,
    AccessOffset,
    ArraySnapshot,
    UnderlyingArray,
    AddNoovfl,
    AddNoovflPredicate,
    UninterpretedFunction,
    ArrayMemberOf,
    AtomicHasTakenValue,
    AsyncWorkGroupCopy,

    // Unary
    Not,
    ArrayId,
    ArrayOffset,
    BVToPtr,
    PtrToBV,
    SafeBVToPtr,
    SafePtrToBV,
    BVToFuncPtr,
    FuncPtrToBV,
    PtrToFuncPtr,
    FuncPtrToPtr,
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
    FCeil,
    FCos,
    FExp,
    FFloor,
    FLog,
    FLog2,
    FrexpExp,
    FrexpFrac,
    FRsqrt,
    FRint,
    FSin,
    FSqrt,
    FTrunc,
    OtherInt,
    OtherBool,
    OtherPtrBase,
    Old,
    GetImageWidth,
    GetImageHeight,

    UnaryFirst = Not,
    UnaryLast = GetImageHeight,

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
    FRem,
    FPow,
    FMax,
    FMin,
    FPowi,
    FLt,
    FEq,
    FUno,
    PtrLt,
    FuncPtrLt,
    Implies,

    BinaryFirst = Eq,
    BinaryLast = Implies
  };

  unsigned refCount;
  bool preventEvalStmt : 1, hasEvalStmt : 1;

  static ref<Expr> createPtrLt(ref<Expr> lhs, ref<Expr> rhs);
  static ref<Expr> createPtrLe(ref<Expr> lhs, ref<Expr> rhs);
  static ref<Expr> createFuncPtrLt(ref<Expr> lhs, ref<Expr> rhs);
  static ref<Expr> createFuncPtrLe(ref<Expr> lhs, ref<Expr> rhs);
  static ref<Expr> createBVConcatN(const std::vector<ref<Expr>> &args);
  static ref<Expr> createNeZero(ref<Expr> bv);
  static ref<Expr> createExactBVSDiv(ref<Expr> lhs, uint64_t rhs,
                                     Var *base = nullptr);

  static Type getArrayCandidateType(const std::set<GlobalArray *> &Globals);
  static Type getPointerRange(ref<Expr> pointer, Type defaultRange);
  bool computeArrayCandidates(std::set<GlobalArray *> &GlobalSet) const;

private:
  Type type;

protected:
  Expr(Type type)
      : refCount(0), preventEvalStmt(false), hasEvalStmt(false), type(type) {}

public:
  virtual ~Expr() {}
  virtual Kind getKind() const = 0;
  const Type &getType() const { return type; }

  static bool classof(const Expr *) { return true; }
};

#define EXPR_KIND(kind)                                                        \
  Kind getKind() const { return kind; }                                        \
  static bool classof(const Expr *E) { return E->getKind() == kind; }          \
  static bool classof(const kind##Expr *) { return true; }

class BVConstExpr : public Expr {
  BVConstExpr(const llvm::APInt &bv)
      : Expr(Type(Type::BV, bv.getBitWidth())), bv(bv) {}
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
  GlobalArrayRefExpr(Type t, GlobalArray *array) : Expr(t), array(array) {}
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
  ConstantArrayRefExpr(llvm::ArrayRef<ref<Expr>> array)
      : Expr(Type(Type::ArrayOf, array[0]->getType())),
        array(array.begin(), array.end()) {}
  std::vector<ref<Expr>> array;

public:
  static ref<Expr> create(llvm::ArrayRef<ref<Expr>> array);

  EXPR_KIND(ConstantArrayRef)
  const std::vector<ref<Expr>> &getArray() const { return array; }
};

class PointerExpr : public Expr {
  PointerExpr(ref<Expr> array, ref<Expr> offset)
      : Expr(Type(Type::Pointer, offset->getType().width)), array(array),
        offset(offset) {}
  ref<Expr> array, offset;

public:
  static ref<Expr> create(ref<Expr> array, ref<Expr> offset);

  EXPR_KIND(Pointer)
  ref<Expr> getArray() const { return array; }
  ref<Expr> getOffset() const { return offset; }
};

class NullFunctionPointerExpr : public Expr {
  NullFunctionPointerExpr(unsigned ptrWidth)
      : Expr(Type(Type::FunctionPointer, ptrWidth)) {}

public:
  static ref<Expr> create(unsigned ptrWidth);

  EXPR_KIND(NullFunctionPointer)
};

class FunctionPointerExpr : public Expr {
  FunctionPointerExpr(std::string funcName, unsigned ptrWidth)
      : Expr(Type(Type::FunctionPointer, ptrWidth)), funcName(funcName) {}
  std::string funcName;

public:
  static ref<Expr> create(std::string funcName, unsigned ptrWidth);

  EXPR_KIND(FunctionPointer)
  std::string getFuncName() const { return funcName; }
};

class LoadExpr : public Expr {
  LoadExpr(Type t, ref<Expr> array, ref<Expr> offset, bool isTemporal)
      : Expr(t), array(array), offset(offset), isTemporal(isTemporal) {}
  ref<Expr> array, offset;
  bool isTemporal;

public:
  static ref<Expr> create(ref<Expr> array, ref<Expr> offset, Type type,
                          bool isTemporal);

  EXPR_KIND(Load)
  ref<Expr> getArray() const { return array; }
  ref<Expr> getOffset() const { return offset; }
  bool getIsTemporal() const { return isTemporal; }
};

class AtomicExpr : public Expr {
  AtomicExpr(Type t, ref<Expr> array, ref<Expr> offset,
             std::vector<ref<Expr>> args, std::string function,
             unsigned int parts, unsigned int part)
      : Expr(t), array(array), offset(offset), args(args), function(function),
        parts(parts), part(part) {}
  ref<Expr> array, offset;
  std::vector<ref<Expr>> args;
  std::string function;
  unsigned int parts, part;

public:
  static ref<Expr> create(ref<Expr> array, ref<Expr> offset,
                          std::vector<ref<Expr>> args, std::string function,
                          unsigned int parts, unsigned int part);

  EXPR_KIND(Atomic)
  ref<Expr> getArray() const { return array; }
  ref<Expr> getOffset() const { return offset; }
  std::vector<ref<Expr>> getArgs() const { return args; }
  std::string getFunction() const { return function; }
  unsigned int getParts() const { return parts; }
  unsigned int getPart() const { return part; }
};

/// Local variable reference.  Used for phi nodes, parameters and return
/// variables.
class VarRefExpr : public Expr {
  Var *var;
  VarRefExpr(Var *var) : Expr(var->getType()), var(var) {
    preventEvalStmt = true;
  }

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
  BVExtractExpr(ref<Expr> expr, unsigned offset, unsigned width)
      : Expr(Type(Type::BV, width)), expr(expr), offset(offset) {}
  ref<Expr> expr;
  unsigned offset;

public:
  static ref<Expr> create(ref<Expr> expr, unsigned offset, unsigned width);

  EXPR_KIND(BVExtract)
  ref<Expr> getSubExpr() const { return expr; }
  unsigned getOffset() const { return offset; }
};

class BVCtlzExpr : public Expr {
  BVCtlzExpr(Type type, ref<Expr> val, ref<Expr> isZeroUndef)
      : Expr(type), val(val), isZeroUndef(isZeroUndef) {}
  ref<Expr> val, isZeroUndef;

public:
  static ref<Expr> create(ref<Expr> val, ref<Expr> isZeroUndef);

  EXPR_KIND(BVCtlz)
  ref<Expr> getVal() const { return val; }
  ref<Expr> getIsZeroUndef() const { return isZeroUndef; }
};

class IfThenElseExpr : public Expr {
  IfThenElseExpr(ref<Expr> cond, ref<Expr> trueExpr, ref<Expr> falseExpr)
      : Expr(trueExpr->getType()), cond(cond), trueExpr(trueExpr),
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

class HavocExpr : public Expr {
  HavocExpr(Type type) : Expr(type) {}

public:
  static ref<Expr> create(Type type);
  EXPR_KIND(Havoc)
};

/// Expression which denotes that its subexpression is an arrayId and a member
/// of the elems set.  This is an unusual expression in that it only shows
/// up in the output indirectly via case splits.
class ArrayMemberOfExpr : public Expr {
  ArrayMemberOfExpr(Type t, ref<Expr> expr,
                    const std::set<GlobalArray *> &elems)
      : Expr(t), expr(expr), elems(elems) {}
  ref<Expr> expr;
  std::set<GlobalArray *> elems;

public:
  static ref<Expr> create(ref<Expr> expr, const std::set<GlobalArray *> &elems);

  EXPR_KIND(ArrayMemberOf)
  ref<Expr> getSubExpr() const { return expr; }
  const std::set<GlobalArray *> &getElems() const { return elems; }
};

class UnaryExpr : public Expr {
  ref<Expr> expr;

protected:
  UnaryExpr(Type type, ref<Expr> expr) : Expr(type), expr(expr) {}

public:
  ref<Expr> getSubExpr() const { return expr; }
  static bool classof(const Expr *E) {
    Kind k = E->getKind();
    return k >= UnaryFirst && k <= UnaryLast;
  }
  static bool classof(const UnaryExpr *) { return true; }
};

#define UNARY_EXPR(kind)                                                       \
  class kind##Expr : public UnaryExpr {                                        \
    kind##Expr(Type type, ref<Expr> expr) : UnaryExpr(type, expr) {}           \
                                                                               \
  public:                                                                      \
    static ref<Expr> create(ref<Expr> var);                                    \
    EXPR_KIND(kind)                                                            \
  };

UNARY_EXPR(Not)

class ArrayIdExpr : public UnaryExpr {
  ArrayIdExpr(Type type, ref<Expr> expr) : UnaryExpr(type, expr) {}

public:
  static ref<Expr> create(ref<Expr> var, Type defaultRange);
  EXPR_KIND(ArrayId)
};

UNARY_EXPR(ArrayOffset)
UNARY_EXPR(PtrToFuncPtr)
UNARY_EXPR(FuncPtrToPtr)
UNARY_EXPR(BVToBool)
UNARY_EXPR(BoolToBV)
UNARY_EXPR(FAbs)
UNARY_EXPR(FCeil)
UNARY_EXPR(FCos)
UNARY_EXPR(FExp)
UNARY_EXPR(FFloor)
UNARY_EXPR(FLog)
UNARY_EXPR(FLog2)
UNARY_EXPR(FrexpFrac)
UNARY_EXPR(FRint)
UNARY_EXPR(FSin)
UNARY_EXPR(FRsqrt)
UNARY_EXPR(FSqrt)
UNARY_EXPR(FTrunc)
UNARY_EXPR(OtherInt)
UNARY_EXPR(OtherBool)
UNARY_EXPR(OtherPtrBase)
UNARY_EXPR(Old)
UNARY_EXPR(GetImageWidth)
UNARY_EXPR(GetImageHeight)

#define UNARY_CONV_EXPR(kind)                                                  \
  class kind##Expr : public UnaryExpr {                                        \
    kind##Expr(Type type, ref<Expr> expr) : UnaryExpr(type, expr) {}           \
                                                                               \
  public:                                                                      \
    static ref<Expr> create(unsigned width, ref<Expr> var);                    \
    EXPR_KIND(kind)                                                            \
  };

UNARY_CONV_EXPR(BVToPtr)
UNARY_CONV_EXPR(PtrToBV)
UNARY_CONV_EXPR(SafeBVToPtr)
UNARY_CONV_EXPR(SafePtrToBV)
UNARY_CONV_EXPR(BVToFuncPtr)
UNARY_CONV_EXPR(FuncPtrToBV)
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
  BinaryExpr(Type type, ref<Expr> lhs, ref<Expr> rhs)
      : Expr(type), lhs(lhs), rhs(rhs) {}

public:
  ref<Expr> getLHS() const { return lhs; }
  ref<Expr> getRHS() const { return rhs; }
  static bool classof(const Expr *E) {
    Kind k = E->getKind();
    return k >= BinaryFirst && k <= BinaryLast;
  }
  static bool classof(const BinaryExpr *) { return true; }
};

#define BINARY_EXPR(kind)                                                      \
  class kind##Expr : public BinaryExpr {                                       \
    kind##Expr(Type type, ref<Expr> lhs, ref<Expr> rhs)                        \
        : BinaryExpr(type, lhs, rhs) {}                                        \
                                                                               \
  public:                                                                      \
    static ref<Expr> create(ref<Expr> lhs, ref<Expr> rhs);                     \
    EXPR_KIND(kind)                                                            \
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
BINARY_EXPR(FRem)
BINARY_EXPR(FPow)
BINARY_EXPR(FMax)
BINARY_EXPR(FMin)
BINARY_EXPR(FPowi)
BINARY_EXPR(FLt)
BINARY_EXPR(FEq)
BINARY_EXPR(FUno)
BINARY_EXPR(PtrLt)
BINARY_EXPR(FuncPtrLt)
BINARY_EXPR(Implies)

#undef BINARY_EXPR

class CallExpr : public Expr {
  Function *callee;
  std::vector<ref<Expr>> args;
  CallExpr(Type t, Function *callee, const std::vector<ref<Expr>> &args)
      : Expr(t), callee(callee), args(args) {}

public:
  static ref<Expr> create(Function *callee, const std::vector<ref<Expr>> &args);

  EXPR_KIND(Call)
  Function *getCallee() const { return callee; }
  const std::vector<ref<Expr>> &getArgs() const { return args; }
};

class CallMemberOfExpr : public Expr {
  ref<Expr> func;
  std::vector<ref<Expr>> callExprs;
  CallMemberOfExpr(Type t, ref<Expr> func, std::vector<ref<Expr>> callExprs)
      : Expr(t), func(func), callExprs(callExprs) {}

public:
  static ref<Expr> create(ref<Expr> func, std::vector<ref<Expr>> &callExprs);

  EXPR_KIND(CallMemberOf)
  ref<Expr> getFunc() const { return func; }
  std::vector<ref<Expr>> getCallExprs() const { return callExprs; }
};

class AccessHasOccurredExpr : public Expr {
  AccessHasOccurredExpr(ref<Expr> array, bool isWrite)
      : Expr(Type::Bool), array(array), isWrite(isWrite) {}
  ref<Expr> array;
  bool isWrite;

public:
  static ref<Expr> create(ref<Expr> array, bool isWrite);

  EXPR_KIND(AccessHasOccurred)
  ref<Expr> getArray() const { return array; }
  std::string getAccessKind() { return isWrite ? "WRITE" : "READ"; }
};

class AccessOffsetExpr : public Expr {
  AccessOffsetExpr(ref<Expr> array, unsigned pointerSize, bool isWrite)
      : Expr(Type(Type::BV, pointerSize)), array(array), isWrite(isWrite) {}
  ref<Expr> array;
  bool isWrite;

public:
  static ref<Expr> create(ref<Expr> array, unsigned pointerSize, bool isWrite);

  EXPR_KIND(AccessOffset)
  ref<Expr> getArray() const { return array; }
  std::string getAccessKind() { return isWrite ? "WRITE" : "READ"; }
};

class ArraySnapshotExpr : public Expr {
  ArraySnapshotExpr(ref<Expr> dst, ref<Expr> src)
      : Expr(Type::BV), dst(dst), src(src) {}
  ref<Expr> dst;
  ref<Expr> src;

public:
  static ref<Expr> create(ref<Expr> dst, ref<Expr> src);

  EXPR_KIND(ArraySnapshot)
  ref<Expr> getDst() const { return dst; }
  ref<Expr> getSrc() const { return src; }
};

class UnderlyingArrayExpr : public Expr {
  UnderlyingArrayExpr(ref<Expr> array) : Expr(array->getType()), array(array) {}
  ref<Expr> array;

public:
  static ref<Expr> create(ref<Expr> array);

  EXPR_KIND(UnderlyingArray)
  ref<Expr> getArray() const { return array; }
};

class AddNoovflExpr : public Expr {
  AddNoovflExpr(ref<Expr> first, ref<Expr> second, bool isSigned)
      : Expr(Type(Type::BV, first->getType().width)), first(first),
        second(second), isSigned(isSigned) {}
  ref<Expr> first;
  ref<Expr> second;
  bool isSigned;

public:
  static ref<Expr> create(ref<Expr> first, ref<Expr> second, bool isSigned);

  EXPR_KIND(AddNoovfl)
  ref<Expr> getFirst() const { return first; }
  ref<Expr> getSecond() const { return second; }
  bool getIsSigned() const { return isSigned; }
};

class AddNoovflPredicateExpr : public Expr {
  std::vector<ref<Expr>> exprs;
  AddNoovflPredicateExpr(const std::vector<ref<Expr>> &exprs)
      : Expr(Type(Type::BV, 1)), exprs(exprs) {}

public:
  static ref<Expr> create(const std::vector<ref<Expr>> &exprs);

  EXPR_KIND(AddNoovflPredicate)
  const std::vector<ref<Expr>> &getExprs() const { return exprs; }
};

class UninterpretedFunctionExpr : public Expr {
  UninterpretedFunctionExpr(const std::string &name, Type returnType,
                            const std::vector<ref<Expr>> &args)
      : Expr(returnType), name(name), args(args) {}
  const std::string name;
  const std::vector<ref<Expr>> args;

public:
  static ref<Expr> create(const std::string &name, Type returnType,
                          const std::vector<ref<Expr>> &args);

  EXPR_KIND(UninterpretedFunction)
  const std::string &getName() { return name; }
  unsigned getNumOperands() const { return args.size(); }
  ref<Expr> getOperand(unsigned index) const { return args[index]; }
};

class AtomicHasTakenValueExpr : public Expr {
  AtomicHasTakenValueExpr(ref<Expr> atomicArray, ref<Expr> offset,
                          ref<Expr> value)
      : Expr(Type::Bool), atomicArray(atomicArray), offset(offset),
        value(value) {}
  ref<Expr> atomicArray;
  ref<Expr> offset;
  ref<Expr> value;

public:
  static ref<Expr> create(ref<Expr> atomicArray, ref<Expr> offset,
                          ref<Expr> value);

  EXPR_KIND(AtomicHasTakenValue)
  ref<Expr> getArray() const { return atomicArray; }
  ref<Expr> getOffset() const { return offset; }
  ref<Expr> getValue() const { return value; }
};

class AsyncWorkGroupCopyExpr : public Expr {
  AsyncWorkGroupCopyExpr(ref<Expr> dst, ref<Expr> dstOffset, ref<Expr> src,
                         ref<Expr> srcOffset, ref<Expr> size, ref<Expr> handle)
      : Expr(handle->getType()), dst(dst), dstOffset(dstOffset), src(src),
        srcOffset(srcOffset), size(size), handle(handle) {}
  ref<Expr> dst;
  ref<Expr> dstOffset;
  ref<Expr> src;
  ref<Expr> srcOffset;
  ref<Expr> size;
  ref<Expr> handle;

public:
  static ref<Expr> create(ref<Expr> dst, ref<Expr> dstOffset, ref<Expr> src,
                          ref<Expr> srcOffset, ref<Expr> size,
                          ref<Expr> handle);

  ref<Expr> getDst() const { return dst; }
  ref<Expr> getDstOffset() const { return dstOffset; }
  ref<Expr> getSrc() const { return src; }
  ref<Expr> getSrcOffset() const { return srcOffset; }
  ref<Expr> getSize() const { return size; }
  ref<Expr> getHandle() const { return handle; }

  EXPR_KIND(AsyncWorkGroupCopy)
};
}

#undef EXPR_KIND

#endif
