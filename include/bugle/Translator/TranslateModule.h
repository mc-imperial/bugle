#ifndef BUGLE_TRANSLATOR_TRANSLATEMODULE_H
#define BUGLE_TRANSLATOR_TRANSLATEMODULE_H

#include "bugle/Ref.h"
#include "bugle/Type.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

class Constant;
class Module;

}

namespace bugle {

class Expr;

class TranslateModule {
  llvm::Module *M;
  llvm::TargetData TD;

  ref<Expr> translateConstant(llvm::Constant *C);
  Type translateType(llvm::Type *T);

public:
  TranslateModule(llvm::Module *M) : M(M), TD(M) {}
  void translate();

  friend class TranslateFunction;
};

}

#endif
