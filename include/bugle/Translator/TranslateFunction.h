#ifndef BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H
#define BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H

#include "bugle/Ref.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class BasicBlock;
class Function;
class Instruction;
class PHINode;
class Value;

}

namespace bugle {

class BasicBlock;
class Expr;
class Function;
class TranslateModule;
class Var;

class TranslateFunction {
  TranslateModule *TM;
  Function *BF;
  llvm::Function *F;
  llvm::DenseMap<llvm::BasicBlock *, BasicBlock *> BasicBlockMap;
  llvm::DenseMap<llvm::Value *, ref<Expr> > ValueExprMap;
  llvm::DenseMap<llvm::PHINode *, Var *> PhiVarMap;
  Var *ReturnVar;

  ref<Expr> translateValue(llvm::Value *V);
  void translateBasicBlock(BasicBlock *BBB, llvm::BasicBlock *BB);
  void translateInstruction(BasicBlock *BBB, llvm::Instruction *I);
  Var *getPhiVariable(llvm::PHINode *PN);
  void addPhiAssigns(BasicBlock *BBB, llvm::BasicBlock *Pred,
                     llvm::BasicBlock *Succ);

public:
  TranslateFunction(TranslateModule *TM, bugle::Function *BF,
                    llvm::Function *F)
    : TM(TM), BF(BF), F(F), ReturnVar(0) {}
  void translate();
};

}

#endif
