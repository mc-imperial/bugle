#ifndef BUGLE_SPECIFICATIONINFO_H
#define BUGLE_SPECIFICATIONINFO_H

#include <bugle/Ref.h>

namespace bugle {

class Expr;
class SourceLoc;

class SpecificationInfo
{
private:
  ref<Expr> expr;
  std::unique_ptr<SourceLoc> sourceloc;

public:
  SpecificationInfo(Expr *expr, SourceLoc *sourceloc) :
    expr(expr), sourceloc(sourceloc) {}
  ref<Expr> getExpr() const { return expr; }
  SourceLoc *getSourceLoc() const { return sourceloc.get(); }
};
}

#endif
