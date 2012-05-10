#ifndef BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H
#define BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H

#include "bugle/Ref.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class BasicBlock;
class Function;
class Instruction;
class Value;

}

namespace bugle {

class BasicBlock;
class Expr;
class TranslateModule;

class TranslateFunction {
  TranslateModule *TM;
  llvm::Function *F;
  llvm::DenseMap<llvm::Value *, ref<Expr> > ValueExprMap;

  ref<Expr> translateValue(llvm::Value *V);
  void translateBasicBlock(BasicBlock *BBB, llvm::BasicBlock *BB);
  void translateInstruction(BasicBlock *BBB, llvm::Instruction *I);

public:
  TranslateFunction(TranslateModule *TM, llvm::Function *F) : TM(TM), F(F) {}
  void translate();
};

}

#endif
