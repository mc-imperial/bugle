#ifndef BUGLE_TRANSLATOR_TRANSLATEMODULE_H
#define BUGLE_TRANSLATOR_TRANSLATEMODULE_H

#include "bugle/Ref.h"
#include "bugle/Type.h"
#include "klee/util/GetElementPtrTypeIterator.h"
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
  llvm::DenseMap<llvm::Constant *, ref<Expr>> ConstantMap;

  ref<Expr> translateConstant(llvm::Constant *C);
  ref<Expr> doTranslateConstant(llvm::Constant *C);

  Type translateType(llvm::Type *T);
  ref<Expr> translateGEP(ref<Expr> Ptr,
                         klee::gep_type_iterator begin,
                         klee::gep_type_iterator end,
                         std::function<ref<Expr>(llvm::Value *)> xlate);

public:
  TranslateModule(bugle::Module *BM, llvm::Module *M) : BM(BM), M(M), TD(M) {}
  void translate();

  friend class TranslateFunction;
};

}

#endif
