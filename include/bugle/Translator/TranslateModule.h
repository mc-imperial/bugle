#ifndef BUGLE_TRANSLATOR_TRANSLATEMODULE_H
#define BUGLE_TRANSLATOR_TRANSLATEMODULE_H

#include "bugle/RaceInstrumenter.h"
#include "bugle/Ref.h"
#include "bugle/SourceLoc.h"
#include "bugle/Type.h"
#include "klee/util/GetElementPtrTypeIterator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include <functional>
#include <map>
#include <set>
#include <vector>

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

typedef std::vector<std::pair<bool, uint64_t>> ArraySpec;

class TranslateModule {
public:
  enum SourceLanguage {
    SL_C,
    SL_CUDA,
    SL_OpenCL,

    SL_Count
  };

  struct AddressSpaceMap {
    const unsigned generic;
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
  std::map<std::string, ArraySpec> GPUArraySizes;

  std::map<llvm::Function *, bugle::Function *> FunctionMap;
  std::map<llvm::Function *, std::vector<llvm::Instruction *> *> StructMap;
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

  ref<Expr> translate1dCUDABuiltinGlobal(std::string Prefix,
                                         llvm::GlobalVariable *GV);
  ref<Expr> translate3dCUDABuiltinGlobal(std::string Prefix,
                                         llvm::GlobalVariable *GV);

  void translateGlobalInit(GlobalArray *GA, unsigned Offset,
                           llvm::Constant *Init);
  bool hasInitializer(llvm::GlobalVariable *GV);
  ref<Expr> translateGlobalVariable(llvm::GlobalVariable *GV);
  void addGlobalArrayAttribs(GlobalArray *GA, llvm::PointerType *PT);
  bugle::GlobalArray *getGlobalArray(llvm::Value *V, bool IsParameter = false);

  ref<Expr> translateConstant(llvm::Constant *C);
  ref<Expr> doTranslateConstant(llvm::Constant *C);

  Type translateType(llvm::Type *T);
  Type handlePadding(Type ElTy, llvm::Type *T);
  Type translateArrayRangeType(llvm::Type *T);
  Type translateSourceType(llvm::Type *T);
  Type translateSourceArrayRangeType(llvm::Type *T);
  void getSourceArrayDimensions(llvm::Type *T, std::vector<uint64_t> &dim);

  ref<Expr> translateGEP(ref<Expr> Ptr, klee::gep_type_iterator begin,
                         klee::gep_type_iterator end,
                         std::function<ref<Expr>(llvm::Value *)> xlate);
  ref<Expr> translateEV(ref<Expr> Agg, klee::ev_type_iterator begin,
                        klee::ev_type_iterator end,
                        std::function<ref<Expr>(llvm::Value *)> xlate);
  ref<Expr> translateIV(ref<Expr> Agg, ref<Expr> Val,
                        klee::iv_type_iterator begin,
                        klee::iv_type_iterator end,
                        std::function<ref<Expr>(llvm::Value *)> xlate);
  ref<Expr> translateBitCast(llvm::Type *SrcTy, llvm::Type *DestTy,
                             ref<Expr> Op);
  ref<Expr> translateArbitrary(Type t);
  ref<Expr> translateICmp(llvm::CmpInst::Predicate P, ref<Expr> LHS,
                          ref<Expr> RHS);

  ref<Expr>
  maybeTranslateSIMDInst(llvm::Type *Ty, llvm::Type *OpTy, ref<Expr> Op,
                         std::function<ref<Expr>(llvm::Type *, ref<Expr>)> F);
  ref<Expr>
  maybeTranslateSIMDInst(llvm::Type *Ty, llvm::Type *OpTy, ref<Expr> LHS,
                         ref<Expr> RHS,
                         std::function<ref<Expr>(ref<Expr>, ref<Expr>)> F);

  ref<Expr> modelValue(llvm::Value *V, ref<Expr> E);
  Type getModelledType(llvm::Value *V);
  ref<Expr> unmodelValue(llvm::Value *V, ref<Expr> E);
  void computeValueModel(llvm::Value *Val, Var *Var,
                         llvm::ArrayRef<ref<Expr>> Assigns);

  Stmt *modelCallStmt(llvm::Type *T, llvm::Function *F, ref<Expr> Val,
                      std::vector<ref<Expr>> &args, SourceLocsRef &sourcelocs);
  ref<Expr> modelCallExpr(llvm::Type *T, llvm::Function *F, ref<Expr> Val,
                          std::vector<ref<Expr>> &args);

  Type defaultRange() {
    return ModelAllAsByteArray ? Type(Type::BV, 8) : Type(Type::Unknown);
  }

public:
  TranslateModule(llvm::Module *M, SourceLanguage SL, std::set<std::string> &EP,
                  RaceInstrumenter RI, AddressSpaceMap &AS,
                  std::map<std::string, ArraySpec> &GAS)
      : BM(nullptr), M(M), TD(M), SL(SL), GPUEntryPoints(EP), RaceInst(RI),
        AddressSpaces(AS), GPUArraySizes(GAS),
        NeedAdditionalByteArrayModels(false), ModelAllAsByteArray(false),
        NextModelAllAsByteArray(false),
        NeedAdditionalGlobalOffsetModels(false) {
    DIF.processModule(*M);
  }

  ~TranslateModule() {
    for (auto i = StructMap.begin(), e = StructMap.end(); i != e; ++i) {
      std::for_each(i->second->rbegin(), i->second->rend(),
                    [](llvm::Instruction *p) { p->deleteValue(); });
      delete i->second;
    }
  }

  static bool isGPUEntryPoint(llvm::Function *F, llvm::Module *M,
                              SourceLanguage SL, std::set<std::string> &EPS);
  std::string getSourceFunctionName(llvm::Function *F);
  std::string getSourceGlobalArrayName(llvm::Value *V);
  static std::string getSourceName(llvm::Value *V, llvm::Function *F);
  void translate();
  bugle::Module *takeModule() { return BM; }

  friend class TranslateFunction;
};
}

#endif
