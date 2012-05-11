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
class Stmt;

class BPLFunctionWriter {
  BPLModuleWriter *MW;
  llvm::raw_ostream &OS;
  llvm::DenseMap<Expr *, unsigned> SSAVarIds;

  void writeExpr(Expr *E);
  void writeStmt(Stmt *S);

public:
  BPLFunctionWriter(BPLModuleWriter *MW, llvm::raw_ostream &OS)
    : MW(MW), OS(OS) {}
  void writeBasicBlock(BasicBlock *BB);
};

}

#endif
