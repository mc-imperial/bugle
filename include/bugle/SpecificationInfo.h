#ifndef BUGLE_SPECIFICATIONINFO_H
#define BUGLE_SPECIFICATIONINFO_H

#include "bugle/SourceLoc.h"
#include "bugle/Ref.h"
#include <vector>

namespace bugle {

class Expr;
class SourceLoc;

class SpecificationInfo {
private:
  ref<Expr> expr;
  SourceLocsRef sourcelocs;

public:
  SpecificationInfo(Expr *expr, const SourceLocsRef &sourcelocs)
      : expr(expr), sourcelocs(sourcelocs) {}
  ref<Expr> getExpr() const { return expr; }
  const SourceLocsRef &getSourceLocs() const { return sourcelocs; }
};
}

#endif
