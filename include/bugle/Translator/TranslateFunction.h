#ifndef BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H
#define BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H

#include "bugle/Ref.h"
#include "bugle/Translator/TranslateModule.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <vector>
#include <functional>

namespace llvm {

class BasicBlock;
class Function;
class Instruction;
class PHINode;
class Type;
class Value;

}

namespace bugle {

class BasicBlock;
class Expr;
class Function;
class TranslateModule;
class Type;
class Var;

class TranslateFunction {
  typedef ref<Expr> SpecialFnHandler(BasicBlock *,
                                     Type,
                                     const std::vector<klee::ref<Expr>> &);
  typedef llvm::StringMap<SpecialFnHandler TranslateFunction::*> SpecialFnMapTy;

  TranslateModule *TM;
  Function *BF;
  llvm::Function *F;
  llvm::DenseMap<llvm::BasicBlock *, BasicBlock *> BasicBlockMap;
  llvm::DenseMap<llvm::Value *, ref<Expr> > ValueExprMap;
  llvm::DenseMap<llvm::PHINode *, Var *> PhiVarMap;

  SpecialFnMapTy &SpecialFunctionMap;
  static SpecialFnMapTy SpecialFunctionMaps[TranslateModule::SL_Count];

  Var *ReturnVar;

  SpecialFnHandler handleAssert, handleAssertFail, handleAssume;

  SpecialFnHandler handleGetLocalId, handleGetGroupId, handleGetLocalSize,
                   handleGetNumGroups, handleGetGlobalId, handleGetGlobalSize;

  ref<Expr> maybeTranslateSIMDInst(bugle::BasicBlock *BBB,
                           llvm::Type *Ty, llvm::Type *OpTy,
                           ref<Expr> Op,
                           std::function<ref<Expr>(llvm::Type *, ref<Expr>)> F);
  ref<Expr> maybeTranslateSIMDInst(bugle::BasicBlock *BBB,
                              llvm::Type *Ty, llvm::Type *OpTy,
                              ref<Expr> LHS, ref<Expr> RHS,
                              std::function<ref<Expr>(ref<Expr>, ref<Expr>)> F);
  ref<Expr> translateValue(llvm::Value *V);
  void translateBasicBlock(BasicBlock *BBB, llvm::BasicBlock *BB);
  void translateInstruction(BasicBlock *BBB, llvm::Instruction *I);
  Var *getPhiVariable(llvm::PHINode *PN);
  void addPhiAssigns(BasicBlock *BBB, llvm::BasicBlock *Pred,
                     llvm::BasicBlock *Succ);

public:
  TranslateFunction(TranslateModule *TM, bugle::Function *BF,
                    llvm::Function *F)
    : TM(TM), BF(BF), F(F), SpecialFunctionMap(SpecialFunctionMaps[TM->SL]),
      ReturnVar(0) {}
  bool isGPUEntryPoint;

  void translate();
};

}

#endif
