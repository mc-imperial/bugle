#ifndef BUGLE_FUNCTION_H
#define BUGLE_FUNCTION_H

#include <string>
#include "bugle/BasicBlock.h"
#include "bugle/OwningPtrVector.h"
#include "bugle/util/UniqueNameSet.h"

namespace bugle {

class Function {
  std::string name;
  OwningPtrVector<BasicBlock> blocks;
  OwningPtrVector<Var> args, returns, locals;
  UniqueNameSet bbNames, varNames;
  
public:
  Function(const std::string &name) : name(name) {}
  BasicBlock *addBasicBlock(const std::string &name) {
    BasicBlock *BB = new BasicBlock(bbNames.makeName(name));
    blocks.push_back(BB);
    return BB;
  }
  Var *addArgument(Type t, const std::string &name) {
    Var *V = new Var(t, varNames.makeName(name));
    args.push_back(V);
    return V;
  }
  Var *addReturn(Type t, const std::string &name) {
    Var *V = new Var(t, varNames.makeName(name));
    returns.push_back(V);
    return V;
  }
  Var *addLocal(Type t, const std::string &name) {
    Var *V = new Var(t, varNames.makeName(name));
    locals.push_back(V);
    return V;
  }

  const std::string &getName() { return name; }

  OwningPtrVector<BasicBlock>::const_iterator begin() const {
    return blocks.begin();
  }
  OwningPtrVector<BasicBlock>::const_iterator end() const {
    return blocks.end();
  }

  OwningPtrVector<Var>::const_iterator arg_begin() const {
    return args.begin();
  }
  OwningPtrVector<Var>::const_iterator arg_end() const {
    return args.end();
  }

  OwningPtrVector<Var>::const_iterator return_begin() const {
    return returns.begin();
  }
  OwningPtrVector<Var>::const_iterator return_end() const {
    return returns.end();
  }

  OwningPtrVector<Var>::const_iterator local_begin() const {
    return locals.begin();
  }
  OwningPtrVector<Var>::const_iterator local_end() const {
    return locals.end();
  }
};

}

#endif
