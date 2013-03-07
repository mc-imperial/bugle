#ifndef BUGLE_BPLFUNCTIONWRITER_H
#define BUGLE_BPLFUNCTIONWRITER_H

#include "bugle/BPLExprWriter.h"
#include "llvm/ADT/DenseMap.h"
#include <set>
#include <functional>

namespace llvm {

class raw_ostream;

}

namespace bugle {

class BPLModuleWriter;
class BasicBlock;
class Expr;
class Function;
class GlobalArray;
class SourceLoc;
class Stmt;
class Var;

class BPLFunctionWriter : BPLExprWriter {
  llvm::raw_ostream &OS;
  bugle::Function *F;
  llvm::DenseMap<Expr *, unsigned> SSAVarIds;
  std::set<GlobalArray *> ModifiesSet;

  void maybeWriteCaseSplit(llvm::raw_ostream &OS, Expr *PtrArr,
                           SourceLoc *SLoc,
                           std::function<void(GlobalArray *)> F);
  void writeVar(llvm::raw_ostream &OS, Var *V);
  void writeExpr(llvm::raw_ostream &OS, Expr *E, unsigned Depth = 0);
  void writeStmt(llvm::raw_ostream &OS, Stmt *S);
  void writeBasicBlock(llvm::raw_ostream &OS, BasicBlock *BB);
  void writeSourceLoc(llvm::raw_ostream &OS, const SourceLoc *sourceloc);
  void writeSourceLocMarker(llvm::raw_ostream &OS, const SourceLoc *sourceloc);

public:
  BPLFunctionWriter(BPLModuleWriter *MW, llvm::raw_ostream &OS,
                    bugle::Function *F)
    : BPLExprWriter(MW), OS(OS), F(F) {}
  void write();
};

}

#endif
