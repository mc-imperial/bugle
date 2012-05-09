#include "bugle/Ref.h"
#include <vector>

#ifndef BUGLE_BASICBLOCK_H
#define BUGLE_BASICBLOCK_H

namespace bugle {

class Stmt;

class BasicBlock {
  std::vector<ref<Stmt> > stmts;

  void addStmt(Stmt *stmt) {
    stmts.push_back(stmt);
  }
};

}

#endif
