#ifndef BUGLE_BPLFUNCTIONWRITER_H
#define BUGLE_BPLFUNCTIONWRITER_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

class raw_ostream;

}

namespace bugle {

class BPLModuleWriter;
class BasicBlock;
class Expr;
class Function;
class Stmt;
class Var;

class BPLFunctionWriter {
  BPLModuleWriter *MW;
  llvm::raw_ostream &OS;
  bugle::Function *F;
  llvm::DenseMap<Expr *, unsigned> SSAVarIds;

  void writeVar(Var *V);
  void writeExpr(Expr *E, unsigned Depth);
  void writeStmt(Stmt *S);
  void writeBasicBlock(BasicBlock *BB);

public:
  BPLFunctionWriter(BPLModuleWriter *MW, llvm::raw_ostream &OS,
                    bugle::Function *F)
    : MW(MW), OS(OS), F(F) {}
  void write();
};

}

#endif
