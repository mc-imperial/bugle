#ifndef BUGLE_STMT_H
#define BUGLE_STMT_H

#include "bugle/Expr.h"
#include "bugle/Ref.h"
#include "bugle/SourceLoc.h"

namespace bugle {

class BasicBlock;
class Expr;
class Function;
class Var;

class Stmt {
public:
  enum Kind {
    Eval,
    Store,
    VarAssign,
    Goto,
    Return,
    Assume,
    Assert,
    Call,
    CallMemberOf
  };

  virtual ~Stmt() {}
  virtual Kind getKind() const = 0;
  void setSourceLocs(const SourceLocsRef &sourcelocs) {
    Stmt::sourcelocs = sourcelocs;
  }
  SourceLocsRef &getSourceLocs() { return sourcelocs; }

protected:
  Stmt() {};
  Stmt(const SourceLocsRef &sourcelocs) : sourcelocs(sourcelocs) {};

private:
  SourceLocsRef sourcelocs;
};

#define STMT_KIND(kind)                                                        \
  Kind getKind() const { return kind; }                                        \
  static bool classof(const Stmt *S) { return S->getKind() == kind; }          \
  static bool classof(const kind##Stmt *) { return true; }

class EvalStmt : public Stmt {
  EvalStmt(ref<Expr> expr) : expr(expr) {};
  ref<Expr> expr;

public:
  static EvalStmt *create(ref<Expr> expr);
  ~EvalStmt();

  STMT_KIND(Eval)
  ref<Expr> getExpr() const { return expr; }
};

class StoreStmt : public Stmt {
  StoreStmt(ref<Expr> array, ref<Expr> offset, ref<Expr> value)
      : array(array), offset(offset), value(value){};
  ref<Expr> array;
  ref<Expr> offset;
  ref<Expr> value;

public:
  static StoreStmt *create(ref<Expr> array, ref<Expr> offset, ref<Expr> value);

  STMT_KIND(Store)
  ref<Expr> getArray() const { return array; }
  ref<Expr> getOffset() const { return offset; }
  ref<Expr> getValue() const { return value; }
};

class VarAssignStmt : public Stmt {
  VarAssignStmt(const std::vector<Var *> &vars,
                const std::vector<ref<Expr>> &values)
      : vars(vars), values(values) {};
  std::vector<Var *> vars;
  std::vector<ref<Expr>> values;

public:
  static VarAssignStmt *create(Var *var, ref<Expr> value);
  static VarAssignStmt *create(const std::vector<Var *> &vars,
                               const std::vector<ref<Expr>> &values);

  STMT_KIND(VarAssign)
  const std::vector<Var *> &getVars() const { return vars; }
  const std::vector<ref<Expr>> &getValues() const { return values; }
};

class GotoStmt : public Stmt {
  GotoStmt(const std::vector<BasicBlock *> &blocks) : blocks(blocks) {}
  std::vector<BasicBlock *> blocks;

public:
  static GotoStmt *create(BasicBlock *block);
  static GotoStmt *create(const std::vector<BasicBlock *> &blocks);

  STMT_KIND(Goto)
  const std::vector<BasicBlock *> &getBlocks() { return blocks; }
};

class ReturnStmt : public Stmt {
  ReturnStmt() {}

public:
  static ReturnStmt *create();

  STMT_KIND(Return)
};

class AssumeStmt : public Stmt {
  AssumeStmt(ref<Expr> pred, bool partition)
      : pred(pred), partition(partition) {};
  ref<Expr> pred;
  bool partition;

public:
  static AssumeStmt *create(ref<Expr> pred);
  static AssumeStmt *createPartition(ref<Expr> pred);

  STMT_KIND(Assume)
  ref<Expr> getPredicate() const { return pred; }
  bool isPartition() const { return partition; }
};

class AssertStmt : public Stmt {
  AssertStmt(ref<Expr> pred, const SourceLocsRef &sourcelocs)
      : Stmt(sourcelocs), pred(pred), global(false), candidate(false),
        invariant(false), badAccess(false) {};
  ref<Expr> pred;
  bool global;
  bool candidate;
  bool invariant;
  bool badAccess;

public:
  static AssertStmt *create(ref<Expr> pred, bool global, bool candidate,
                            const SourceLocsRef &sourcelocs);
  static AssertStmt *createInvariant(ref<Expr> pred, bool global,
                                     bool candidate,
                                     const SourceLocsRef &sourcelocs);
  static AssertStmt *createBadAccess(const SourceLocsRef &sourcelocs);

  STMT_KIND(Assert)
  ref<Expr> getPredicate() const { return pred; }
  bool isGlobal() const { return global; }
  bool isCandidate() const { return candidate; }
  bool isInvariant() const { return invariant; }
  bool isBadAccess() const { return badAccess; }
};

class CallStmt : public Stmt {
  CallStmt(Function *callee, const std::vector<ref<Expr>> &args,
           const SourceLocsRef &sourcelocs)
      : Stmt(sourcelocs), callee(callee), args(args) {}
  Function *callee;
  std::vector<ref<Expr>> args;

public:
  static CallStmt *create(Function *callee, const std::vector<ref<Expr>> &args,
                          const SourceLocsRef &sourcelocs);

  STMT_KIND(Call)
  Function *getCallee() const { return callee; }
  const std::vector<ref<Expr>> &getArgs() const { return args; }
};

class CallMemberOfStmt : public Stmt {
  CallMemberOfStmt(ref<Expr> func, std::vector<Stmt *> &callStmts,
                   const SourceLocsRef &sourcelocs)
      : Stmt(sourcelocs), func(func), callStmts(callStmts) {};
  ref<Expr> func;
  std::vector<Stmt *> callStmts;

public:
  static CallMemberOfStmt *create(ref<Expr> func,
                                  std::vector<Stmt *> &callStmts,
                                  const SourceLocsRef &sourcelocs);

  STMT_KIND(CallMemberOf)
  ref<Expr> getFunc() const { return func; }
  std::vector<Stmt *> getCallStmts() const { return callStmts; }
};
}

#endif
