#ifndef BUGLE_BPLFUNCTIONWRITER_H
#define BUGLE_BPLFUNCTIONWRITER_H

#include "llvm/ADT/DenseMap.h"
#include <set>

namespace llvm {

class raw_ostream;

}

namespace bugle {

class BPLModuleWriter;
class BasicBlock;
class Expr;
class Function;
class GlobalArray;
class Stmt;
class Var;

class BPLFunctionWriter {
  BPLModuleWriter *MW;
  llvm::raw_ostream &OS;
  bugle::Function *F;
  llvm::DenseMap<Expr *, unsigned> SSAVarIds;
  std::set<GlobalArray *> ModifiesSet;

  void writeVar(llvm::raw_ostream &OS, Var *V);
  void writeExpr(llvm::raw_ostream &OS, Expr *E, unsigned Depth);
  void writeStmt(llvm::raw_ostream &OS, Stmt *S);
  void writeBasicBlock(llvm::raw_ostream &OS, BasicBlock *BB);

public:
  BPLFunctionWriter(BPLModuleWriter *MW, llvm::raw_ostream &OS,
                    bugle::Function *F)
    : MW(MW), OS(OS), F(F) {}
  void write();
};

}

#endif
