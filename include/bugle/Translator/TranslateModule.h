#ifndef BUGLE_TRANSLATOR_TRANSLATEMODULE_H
#define BUGLE_TRANSLATOR_TRANSLATEMODULE_H

#include "bugle/RaceInstrumenter.h"
#include "bugle/Ref.h"
#include "bugle/Type.h"
#include "klee/util/GetElementPtrTypeIterator.h"
#include "llvm/DebugInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DataLayout.h"
#include <functional>
#include <map>
#include <set>

namespace llvm {

class CallInst;
class Constant;
class GlobalVariable;
class Module;
class PointerType;
}

namespace bugle {

class Expr;
class Function;
class GlobalArray;
class Module;
class Stmt;
class Var;

class TranslateModule {
public:
  enum SourceLanguage {
    SL_C,
    SL_CUDA,
    SL_OpenCL,

    SL_Count
  };

  struct AddressSpaceMap {
    const unsigned standard;
    const unsigned global;
    const unsigned group_shared;
    const unsigned constant;
    AddressSpaceMap(unsigned Global, unsigned GroupShared, unsigned Constant);
  };

private:
  bugle::Module *BM;
  llvm::Module *M;
  llvm::DebugInfoFinder DIF;
  llvm::DataLayout TD;
  SourceLanguage SL;
  std::set<std::string> GPUEntryPoints;
  RaceInstrumenter RaceInst;
  AddressSpaceMap AddressSpaces;

  std::map<llvm::Function *, bugle::Function *> FunctionMap;
  std::map<llvm::Constant *, ref<Expr>> ConstantMap;

  std::map<GlobalArray *, llvm::Value *> GlobalValueMap;
  std::map<llvm::Value *, GlobalArray *> ValueGlobalMap;

  bool NeedAdditionalByteArrayModels;
  std::set<llvm::Value *> ModelAsByteArray;
  bool ModelAllAsByteArray, NextModelAllAsByteArray;

  std::map<llvm::Function *, std::vector<const std::vector<ref<Expr>> *>>
  CallSites;
  bool NeedAdditionalGlobalOffsetModels;
  std::map<llvm::Value *, std::set<llvm::Value *>> ModelPtrAsGlobalOffset,
      NextModelPtrAsGlobalOffset;
  std::set<llvm::Value *> PtrMayBeNull, NextPtrMayBeNull;

  ref<Expr> translateCUDABuiltinGlobal(std::string Prefix,
                                       llvm::GlobalVariable *GV);

  void translateGlobalInit(GlobalArray *GA, unsigned Offset,
                           llvm::Constant *Init);
  bool hasInitializer(llvm::GlobalVariable *GV);
  ref<Expr> translateGlobalVariable(llvm::GlobalVariable *GV);
  void addGlobalArrayAttribs(GlobalArray *GA, llvm::PointerType *PT);
  bugle::GlobalArray *getGlobalArray(llvm::Value *V);

  ref<Expr> translateConstant(llvm::Constant *C);
  ref<Expr> doTranslateConstant(llvm::Constant *C);

  Type translateType(llvm::Type *T);
  Type translateArrayRangeType(llvm::Type *T);
  Type translateSourceArrayRangeType(llvm::Type *T);
  bool sourceArrayIsMultiDimensional(llvm::Type *T, bool IsGlobal);

  ref<Expr> translateGEP(ref<Expr> Ptr, klee::gep_type_iterator begin,
                         klee::gep_type_iterator end,
                         std::function<ref<Expr>(llvm::Value *)> xlate);
  ref<Expr> translateEV(ref<Expr> Vec, klee::ev_type_iterator begin,
                        klee::ev_type_iterator end,
                        std::function<ref<Expr>(llvm::Value *)> xlate);
  ref<Expr> translateBitCast(llvm::Type *SrcTy, llvm::Type *DestTy,
                             ref<Expr> Op);
  ref<Expr> translateArbitrary(Type t);

  ref<Expr> modelValue(llvm::Value *V, ref<Expr> E);
  Type getModelledType(llvm::Value *V);
  ref<Expr> unmodelValue(llvm::Value *V, ref<Expr> E);
  void computeValueModel(llvm::Value *Val, Var *Var,
                         llvm::ArrayRef<ref<Expr>> Assigns);

  Stmt *modelCallStmt(llvm::Type *T, llvm::Function *F, ref<Expr> Val,
                      std::vector<ref<Expr>> &args);
  ref<Expr> modelCallExpr(llvm::Type *T, llvm::Function *F, ref<Expr> Val,
                          std::vector<ref<Expr>> &args);

  Type defaultRange() {
    return ModelAllAsByteArray ? Type(Type::BV, 8) : Type(Type::Unknown);
  }

public:
  TranslateModule(llvm::Module *M, SourceLanguage SL, std::set<std::string> &EP,
                  RaceInstrumenter &RaceInst, AddressSpaceMap &AS)
      : BM(0), M(M), TD(M), SL(SL), GPUEntryPoints(EP), RaceInst(RaceInst),
        AddressSpaces(AS), NeedAdditionalByteArrayModels(false),
        ModelAllAsByteArray(false), NextModelAllAsByteArray(false) {
    DIF.processModule(*M);
  }
  static bool isGPUEntryPoint(llvm::Function *F, llvm::Module *M,
                              SourceLanguage SL, std::set<std::string> &EPS);
  std::string getSourceFunctionName(llvm::Function *F);
  std::string getSourceGlobalArrayName(llvm::Value *V);
  void translate();
  bugle::Module *takeModule() { return BM; }

  friend class TranslateFunction;
};
}

#endif
