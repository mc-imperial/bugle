#ifndef BUGLE_TRANSLATOR_TRANSLATEMODULE_H
#define BUGLE_TRANSLATOR_TRANSLATEMODULE_H

#include "bugle/Ref.h"

namespace llvm {

class Constant;
class Module;

}

namespace bugle {

class Expr;

class TranslateModule {
  llvm::Module *M;

  ref<Expr> translateConstant(llvm::Constant *C);

public:
  TranslateModule(llvm::Module *M) : M(M) {}
  void translate();

  friend class TranslateFunction;
};

}

#endif
