#include "bugle/Transform/SimplifyStmt.h"
#include "bugle/Module.h"
#include "bugle/Function.h"
#include "bugle/BasicBlock.h"

using namespace bugle;

namespace {

bool hasSideEffects(Expr *e) {
  return isa<CallExpr>(e);
}

bool isTemporal(Expr *e) {
  return isa<LoadExpr>(e) || isa<HavocExpr>(e);
}

void ProcessBasicBlock(BasicBlock *BB) {
  OwningPtrVector<Stmt> &V = BB->getStmtVector();
  if (V.empty())
    return;
  for (auto i = V.end()-1;;) {
    if (auto ES = dyn_cast<EvalStmt>(*i)) {
      Expr *E = ES->getExpr().get();
      if (hasSideEffects(E)) {
        if (i == V.begin())
          break;
        --i;
        continue;
      }

      if (E->refCount == 1 || (!isTemporal(E) && E->refCount <= 2)) {
        auto ii = i;
        bool begin = false;
        if (i == V.begin())
          begin = true;
        else
          --i;
        delete *ii;
        V.erase(ii);
        if (begin)
          break;
        else
          continue;
      }
    }
    if (i == V.begin())
      break;
    --i;
  }
}

void ProcessFunction(Function *F) {
  for (auto i = F->begin(), e = F->end(); i != e; ++i)
    ProcessBasicBlock(*i);
}

void ProcessModule(Module *M) {
  for (auto i = M->begin(), e = M->end(); i != e; ++i)
    ProcessFunction(*i);
}

}

void bugle::simplifyStmt(Module *M) {
  ProcessModule(M);
}
