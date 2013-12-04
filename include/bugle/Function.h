#ifndef BUGLE_FUNCTION_H
#define BUGLE_FUNCTION_H

#include "bugle/BasicBlock.h"
#include "bugle/Ident.h"
#include "bugle/OwningPtrVector.h"
#include "bugle/SourceLoc.h"
#include "bugle/SpecificationInfo.h"
#include "bugle/util/UniqueNameSet.h"
#include <vector>
#include <set>
#include <string>

namespace bugle {

class Function {
  std::string name;
  std::set<std::string> attributes;
  bool entryPoint, specification;
  OwningPtrVector<SpecificationInfo> requires, globalRequires, ensures,
                                     globalEnsures, modifies;
  OwningPtrVector<BasicBlock> blocks;
  OwningPtrVector<Var> args, returns, locals;
  UniqueNameSet bbNames, varNames;

public:
  Function(const std::string &name) : name(name), entryPoint(false), specification(false) {}
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
  void addRequires(ref<Expr> r, const SourceLocsRef &ss) {
    requires.push_back(new SpecificationInfo(r.get(), ss));
  }
  void addGlobalRequires(ref<Expr> r, const SourceLocsRef &ss) {
    globalRequires.push_back(new SpecificationInfo(r.get(), ss));
  }
  void addEnsures(ref<Expr> e, const SourceLocsRef &ss) {
    ensures.push_back(new SpecificationInfo(e.get(), ss));
  }
  void addGlobalEnsures(ref<Expr> e, const SourceLocsRef &ss) {
    globalEnsures.push_back(new SpecificationInfo(e.get(), ss));
  }
  void addModifies(ref<Expr> e, const SourceLocsRef &ss) {
    modifies.push_back(new SpecificationInfo(e.get(), ss));
  }

  const std::string &getName() { return name; }
  bool isEntryPoint() const { return entryPoint; }
  void setEntryPoint(bool ep) { entryPoint = ep; }

  bool isSpecification() const { return specification; }
  void setSpecification(bool s) { specification = s; }

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

  OwningPtrVector<SpecificationInfo>::const_iterator requires_begin() const {
    return requires.begin();
  }
  OwningPtrVector<SpecificationInfo>::const_iterator requires_end() const {
    return requires.end();
  }

  OwningPtrVector<SpecificationInfo>::const_iterator globalRequires_begin() const {
    return globalRequires.begin();
  }
  OwningPtrVector<SpecificationInfo>::const_iterator globalRequires_end() const {
    return globalRequires.end();
  }

  OwningPtrVector<SpecificationInfo>::const_iterator ensures_begin() const {
    return ensures.begin();
  }
  OwningPtrVector<SpecificationInfo>::const_iterator ensures_end() const {
    return ensures.end();
  }

  OwningPtrVector<SpecificationInfo>::const_iterator globalEnsures_begin() const {
    return globalEnsures.begin();
  }
  OwningPtrVector<SpecificationInfo>::const_iterator globalEnsures_end() const {
    return globalEnsures.end();
  }

  OwningPtrVector<SpecificationInfo>::const_iterator modifies_begin() const {
    return modifies.begin();
  }
  OwningPtrVector<SpecificationInfo>::const_iterator modifies_end() const {
    return modifies.end();
  }

};

}

#endif
