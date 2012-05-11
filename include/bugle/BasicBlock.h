#include "bugle/OwningPtrVector.h"
#include "bugle/Ref.h"
#include "bugle/Stmt.h"
#include <vector>

#ifndef BUGLE_BASICBLOCK_H
#define BUGLE_BASICBLOCK_H

namespace bugle {

class Stmt;

class BasicBlock {
  OwningPtrVector<Stmt> stmts;

public:
  void addStmt(Stmt *stmt) {
    stmts.push_back(stmt);
  }

  OwningPtrVector<Stmt>::const_iterator begin() const { return stmts.begin(); }
  OwningPtrVector<Stmt>::const_iterator end() const   { return stmts.end(); }
};

}

#endif
