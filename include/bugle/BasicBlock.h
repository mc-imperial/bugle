#include "bugle/OwningPtrVector.h"
#include "bugle/Ref.h"
#include "bugle/Stmt.h"
#include <vector>

#ifndef BUGLE_BASICBLOCK_H
#define BUGLE_BASICBLOCK_H

namespace bugle {

class Stmt;

class BasicBlock {
  std::string name;
  OwningPtrVector<Stmt> stmts;

public:
  BasicBlock(const std::string &name) : name(name) {}
  void addStmt(Stmt *stmt) {
    stmts.push_back(stmt);
  }

  const std::string &getName() { return name; }

  OwningPtrVector<Stmt>::const_iterator begin() const { return stmts.begin(); }
  OwningPtrVector<Stmt>::const_iterator end() const   { return stmts.end(); }
};

}

#endif
