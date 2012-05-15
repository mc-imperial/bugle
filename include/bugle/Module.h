#ifndef BUGLE_MODULE_H
#define BUGLE_MODULE_H

#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/OwningPtrVector.h"
#include "bugle/util/UniqueNameSet.h"

namespace bugle {

class Module {
  OwningPtrVector<Function> functions;
  OwningPtrVector<GlobalArray> globals;
  UniqueNameSet functionNames, globalNames;
  unsigned pointerWidth;

public:
  Function *addFunction(const std::string &name) {
    Function *F = new Function(functionNames.makeName(name));
    functions.push_back(F);
    return F;
  }

  GlobalArray *addGlobal(const std::string &name) {
    GlobalArray *GA = new GlobalArray(globalNames.makeName(name));
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

  unsigned getPointerWidth() const { return pointerWidth; }
  void setPointerWidth(unsigned pw) { pointerWidth = pw; }
};

}

#endif

