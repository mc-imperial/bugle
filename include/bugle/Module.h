#ifndef BUGLE_MODULE_H
#define BUGLE_MODULE_H

#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/Ident.h"
#include "bugle/OwningPtrVector.h"
#include "bugle/util/UniqueNameSet.h"
#include "bugle/Ref.h"

namespace bugle {

class Expr;

struct GlobalInit {
  GlobalArray *array;
  uint64_t offset;
  ref<Expr> init;
  GlobalInit(GlobalArray *array, uint64_t offset, ref<Expr> init) :
    array(array), offset(offset), init(init) {}
};

class Module {
  std::vector<ref<Expr>> axioms;
  OwningPtrVector<Function> functions;
  OwningPtrVector<GlobalArray> globals;
  std::vector<GlobalInit> globalInits;
  UniqueNameSet functionNames, globalNames;
  unsigned pointerWidth;

public:
  Function *addFunction(const std::string &name) {
    Function *F = new Function(functionNames.makeName(makeBoogieIdent(name)));
    functions.push_back(F);
    return F;
  }

  GlobalArray *addGlobal(const std::string &name) {
    GlobalArray *GA =
      new GlobalArray(globalNames.makeName(makeBoogieIdent(name)));
    globals.push_back(GA);
    return GA;
  }

  OwningPtrVector<Function>::const_iterator begin() const {
    return functions.begin();
  }
  OwningPtrVector<Function>::const_iterator end() const {
    return functions.end();
  }

  OwningPtrVector<GlobalArray>::const_iterator global_begin() const {
    return globals.begin();
  }
  OwningPtrVector<GlobalArray>::const_iterator global_end() const {
    return globals.end();
  }

  std::vector<GlobalInit>::const_iterator global_init_begin() const {
    return globalInits.begin();
  }
  std::vector<GlobalInit>::const_iterator global_init_end() const {
    return globalInits.end();
  }

  std::vector<ref<Expr>>::const_iterator axiom_begin() const {
    return axioms.begin();
  }
  std::vector<ref<Expr>>::const_iterator axiom_end() const {
    return axioms.end();
  }

  unsigned getPointerWidth() const { return pointerWidth; }
  void setPointerWidth(unsigned pw) { pointerWidth = pw; }

  void addAxiom(ref<Expr> axiom) { axioms.push_back(axiom); }
  void addGlobalInit(GlobalArray *array, uint64_t offset, ref<Expr> init) {
    globalInits.push_back(GlobalInit(array, offset, init));
  }
};

}

#endif

