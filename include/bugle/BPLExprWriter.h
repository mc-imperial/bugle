#ifndef BUGLE_BPLEXPRWRITER_H
#define BUGLE_BPLEXPRWRITER_H

namespace llvm {

class raw_ostream;

}

namespace bugle {

class BPLModuleWriter;
class Expr;

class BPLExprWriter {
protected:
  BPLModuleWriter *MW;

public:
  BPLExprWriter(BPLModuleWriter *MW) : MW(MW) {}
  virtual ~BPLExprWriter();
  virtual void writeExpr(llvm::raw_ostream &OS, Expr *E, unsigned Depth = 0);
};

}

#endif
