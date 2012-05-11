#ifndef BUGLE_TRANSLATOR_TRANSLATEMODULE_H
#define BUGLE_TRANSLATOR_TRANSLATEMODULE_H

#include "bugle/Ref.h"
#include "bugle/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

class Constant;
class Module;

}

namespace bugle {

class Expr;
class Function;
class Module;

class TranslateModule {
  bugle::Module *BM;
  llvm::Module *M;
  llvm::TargetData TD;

  llvm::DenseMap<llvm::Function *, bugle::Function *> FunctionMap;

  ref<Expr> translateConstant(llvm::Constant *C);
  Type translateType(llvm::Type *T);

public:
  TranslateModule(bugle::Module *BM, llvm::Module *M) : BM(BM), M(M), TD(M) {}
  void translate();

  friend class TranslateFunction;
};

}

#endif
