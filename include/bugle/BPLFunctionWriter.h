#ifndef BUGLE_BPLFUNCTIONWRITER_H
#define BUGLE_BPLFUNCTIONWRITER_H

#include "bugle/BPLExprWriter.h"
#include "bugle/SourceLoc.h"
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace llvm {

class raw_ostream;
}

namespace bugle {

class BPLModuleWriter;
class BasicBlock;
class CallStmt;
class Expr;
class Function;
class GlobalArray;
class Stmt;
class Var;

class BPLFunctionWriter : BPLExprWriter {
  llvm::raw_ostream &OS;
  bugle::Function *F;
  std::map<Expr *, unsigned> SSAVarIds;
  std::set<GlobalArray *> ModifiesSet;

  void maybeWriteCaseSplit(llvm::raw_ostream &OS, Expr *PtrArr,
                           const SourceLocsRef &SLocs,
                           std::function<void(GlobalArray *, unsigned)> F,
                           unsigned indent = 2);
  void writeVar(llvm::raw_ostream &OS, Var *V);
  void writeExpr(llvm::raw_ostream &OS, Expr *E, unsigned Depth = 0) override;
  void writeCallStmt(llvm::raw_ostream &OS, CallStmt *CS);
  void writeStmt(llvm::raw_ostream &OS, Stmt *S);
  void writeBasicBlock(llvm::raw_ostream &OS, BasicBlock *BB);
  void writeSourceLocs(llvm::raw_ostream &OS, const SourceLocsRef &sourcelocs);
  void writeSourceLocsMarker(llvm::raw_ostream &OS,
                             const SourceLocsRef &sourcelocs,
                             const unsigned int indentLevel);

public:
  BPLFunctionWriter(BPLModuleWriter *MW, llvm::raw_ostream &OS,
                    bugle::Function *F)
      : BPLExprWriter(MW), OS(OS), F(F) {}
  void write();
};
}

#endif
