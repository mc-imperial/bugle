#ifndef BUGLE_BPLEXPRWRITER_H
#define BUGLE_BPLEXPRWRITER_H

namespace llvm {

class raw_ostream;

}

namespace bugle {

class BPLModuleWriter;
class Expr;

class BPLExprWriter {
public:
  virtual ~BPLExprWriter();
  virtual BPLModuleWriter *getModuleWriter() = 0;
  virtual void writeExpr(llvm::raw_ostream &OS, Expr *E, unsigned Depth = 0);
};

}

#endif
