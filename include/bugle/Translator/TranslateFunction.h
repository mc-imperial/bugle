#ifndef BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H
#define BUGLE_TRANSLATOR_TRANSLATEFUNCTION_H

#include "bugle/Ref.h"
#include "bugle/Stmt.h"
#include "bugle/Translator/TranslateModule.h"
#include "llvm/ADT/StringMap.h"
#include <functional>
#include <map>
#include <vector>

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
struct Type;
class Var;

class TranslateFunction {
  typedef ref<Expr> SpecialFnHandler(BasicBlock *, llvm::CallInst *,
                                     const std::vector<klee::ref<Expr>> &);
  struct SpecialFnMapTy {
    llvm::StringMap<SpecialFnHandler TranslateFunction::*> Functions;
    std::map<unsigned, SpecialFnHandler TranslateFunction::*> Intrinsics;
  };

  typedef std::pair<llvm::Value *, ref<Expr>> PhiPair;

  TranslateModule *TM;
  Function *BF;
  llvm::Function *F;
  bool isGPUEntryPoint;
  std::map<llvm::BasicBlock *, BasicBlock *> BasicBlockMap;
  std::map<llvm::Value *, ref<Expr>> ValueExprMap;
  std::map<llvm::PHINode *, Var *> PhiVarMap;
  std::map<llvm::PHINode *, std::vector<PhiPair>> PhiAssignsMap;
  Var *ReturnVar;
  std::vector<ref<Expr>> ReturnVals;
  bool LoadsAreTemporal;
  std::map<unsigned, bugle::Function *> BarrierInvariants;
  std::map<unsigned, bugle::Function *> BinaryBarrierInvariants;
  SourceLocsRef currentSourceLocs;

  SpecialFnMapTy &SpecialFunctionMap;
  static SpecialFnMapTy SpecialFunctionMaps[TranslateModule::SL_Count];

  SpecialFnHandler handleNoop, handleAssertFail, handleAssume, handleAssert,
      handleGlobalAssert, handleCandidateAssert, handleCandidateGlobalAssert,
      handleInvariant, handleGlobalInvariant, handleCandidateInvariant,
      handleCandidateGlobalInvariant, handleNonTemporalLoadsBegin,
      handleNonTemporalLoadsEnd, handleRequires, handleEnsures,
      handleGlobalRequires, handleGlobalEnsures, handleReadsFrom,
      handleWritesTo, handleAll, handleExclusive, handleEnabled, handleOtherInt,
      handleOtherBool, handleOtherPtrBase, handleOld, handleReturnVal,
      handleImplies, handleReadHasOccurred, handleWriteHasOccurred,
      handleReadOffset, handleWriteOffset, handlePtrOffset, handlePtrBase,
      handleArraySnapshot, handleBarrierInvariant, handleBarrierInvariantBinary,
      handleAddNoovflUnsigned, handleAddNoovflSigned, handleAddNoovflPredicate,
      handleAdd, handleIte, handleUninterpretedFunction,
      handleAtomicHasTakenValue, handleMemset, handleMemcpy, handleTrap;

  SpecialFnHandler handleGetLocalId, handleGetGroupId, handleGetLocalSize,
      handleGetNumGroups, handleGetImageWidth, handleGetImageHeight,
      handleAsyncWorkGroupCopy, handleWaitGroupEvents;

  SpecialFnHandler handleCeil, handleCos, handleExp, handleFabs, handleFloor,
      handleFrexpExp, handleFrexpFrac, handleFma, handleSqrt, handleLog,
      handlePow, handleSin, handleRsqrt;

  SpecialFnHandler handleAtomic;

  static SpecialFnMapTy &
  initSpecialFunctionMap(TranslateModule::SourceLanguage SL);

  ref<Expr>
  maybeTranslateSIMDInst(bugle::BasicBlock *BBB, llvm::Type *Ty,
                         llvm::Type *OpTy, ref<Expr> Op,
                         std::function<ref<Expr>(llvm::Type *, ref<Expr>)> F);
  ref<Expr>
  maybeTranslateSIMDInst(bugle::BasicBlock *BBB, llvm::Type *Ty,
                         llvm::Type *OpTy, ref<Expr> LHS, ref<Expr> RHS,
                         std::function<ref<Expr>(ref<Expr>, ref<Expr>)> F);
  ref<Expr> translateValue(llvm::Value *V, bugle::BasicBlock *BBB);
  void translateBasicBlock(BasicBlock *BBB, llvm::BasicBlock *BB);
  void translateInstruction(BasicBlock *BBB, llvm::Instruction *I);
  Var *getPhiVariable(llvm::PHINode *PN);
  void computeClosure(std::vector<PhiPair> &currentAssigns,
                      std::set<llvm::PHINode *> &foundPhiNodes,
                      std::vector<ref<Expr>> &assigns);
  void addPhiAssigns(BasicBlock *BBB, llvm::BasicBlock *Pred,
                     llvm::BasicBlock *Succ);
  SourceLocsRef extractSourceLocs(llvm::Instruction *I);

public:
  TranslateFunction(TranslateModule *TM, bugle::Function *BF, llvm::Function *F,
                    bool isGPUEntryPoint)
      : TM(TM), BF(BF), F(F), isGPUEntryPoint(isGPUEntryPoint), ReturnVar(0),
        LoadsAreTemporal(true), currentSourceLocs(new SourceLocs),
        SpecialFunctionMap(initSpecialFunctionMap(TM->SL)) {}

  static bool isSpecialFunction(TranslateModule::SourceLanguage SL,
                                const std::string &fnName);
  static void addUninterpretedFunction(TranslateModule::SourceLanguage SL,
                                       const std::string &fnName);
  static bool isAxiomFunction(llvm::StringRef fnName);
  static bool isUninterpretedFunction(llvm::StringRef fnName);
  static bool isSpecificationFunction(llvm::StringRef fnName);
  static bool isPreOrPostCondition(llvm::StringRef fnName);
  static bool isBarrierFunction(TranslateModule::SourceLanguage SL,
                                llvm::StringRef fnName);
  static bool isNormalFunction(TranslateModule::SourceLanguage SL,
                               llvm::Function *F);
  static bool isStandardEntryPoint(TranslateModule::SourceLanguage SL,
                                   llvm::StringRef fnName);

  void translate();
};
}

#endif
