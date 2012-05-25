#ifndef BUGLE_FUNCTION_H
#define BUGLE_FUNCTION_H

#include <set>
#include <string>
#include "bugle/BasicBlock.h"
#include "bugle/Ident.h"
#include "bugle/OwningPtrVector.h"
#include "bugle/util/UniqueNameSet.h"

namespace bugle {

class Function {
  std::string name;
  std::set<std::string> attributes;
  bool entryPoint;
  OwningPtrVector<BasicBlock> blocks;
  OwningPtrVector<Var> args, returns, locals;
  UniqueNameSet bbNames, varNames;
  
public:
  Function(const std::string &name) : name(name), entryPoint(false) {}
  BasicBlock *addBasicBlock(const std::string &name) {
    BasicBlock *BB = new BasicBlock(bbNames.makeName(makeBoogieIdent(name)));
    blocks.push_back(BB);
    return BB;
  }
  Var *addArgument(Type t, const std::string &name) {
    Var *V = new Var(t, varNames.makeName(makeBoogieIdent(name)));
    args.push_back(V);
    return V;
  }
  Var *addReturn(Type t, const std::string &name) {
    Var *V = new Var(t, varNames.makeName(makeBoogieIdent(name)));
    returns.push_back(V);
    return V;
  }
  Var *addLocal(Type t, const std::string &name) {
    Var *V = new Var(t, varNames.makeName(makeBoogieIdent(name)));
    locals.push_back(V);
    return V;
  }
  void addAttribute(const std::string &attrib) {
    attributes.insert(attrib);
  }

  const std::string &getName() { return name; }
  bool isEntryPoint() const { return entryPoint; }
  void setEntryPoint(bool ep) { entryPoint = ep; }

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

  std::set<std::string>::const_iterator attrib_begin() const {
    return attributes.begin();
  }
  std::set<std::string>::const_iterator attrib_end() const {
    return attributes.end();
  }
};

}

#endif
