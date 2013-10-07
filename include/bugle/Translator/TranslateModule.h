#ifndef BUGLE_TRANSLATOR_TRANSLATEMODULE_H
#define BUGLE_TRANSLATOR_TRANSLATEMODULE_H

#include "bugle/Ref.h"
#include "bugle/Type.h"
#include "klee/util/GetElementPtrTypeIterator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/DataLayout.h"
#include <functional>
#include <set>

namespace llvm {

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
class Var;

class TranslateModule {
public:
  enum SourceLanguage {
    SL_C,
    SL_CUDA,
    SL_OpenCL,

    SL_Count
  };

  enum AddressSpaces {
    // These constants match NVPTXAddrSpaceMap in Targets.cpp
    // There does not appear to be a header file in which they
    // are symbolically defined
    global = 1, // opencl_global, cuda_device
    group_shared = 3, // opencl_local, cuda_shared
    constant = 4 // opencl_constant, cuda_constant
  };

private:
  bugle::Module *BM;
  llvm::Module *M;
  llvm::DataLayout TD;
  SourceLanguage SL;

  llvm::DenseMap<llvm::Function *, bugle::Function *> FunctionMap;
  llvm::DenseMap<llvm::Constant *, ref<Expr>> ConstantMap;

  std::set<std::string> GPUEntryPoints;

  llvm::DenseMap<GlobalArray *, llvm::Value *> GlobalValueMap;
  llvm::DenseMap<llvm::Value *, GlobalArray *> ValueGlobalMap;

  bool NeedAdditionalByteArrayModels;
  std::set<llvm::Value *> ModelAsByteArray;
  bool ModelAllAsByteArray, NextModelAllAsByteArray;

  std::map<llvm::Function *, std::vector<const std::vector<ref<Expr>> *>>
    CallSites;
  bool NeedAdditionalGlobalOffsetModels;
  std::map<llvm::Value *, std::set<llvm::Value *>>
    ModelPtrAsGlobalOffset, NextModelPtrAsGlobalOffset;
  std::set<llvm::Value *>
    PtrMayBeNull, NextPtrMayBeNull;

  ref<Expr> translateCUDABuiltinGlobal(std::string Prefix,
                                       llvm::GlobalVariable *GV);

  void translateGlobalInit(GlobalArray *GA, unsigned Offset,
                           llvm::Constant *Init);
  ref<Expr> translateGlobalVariable(llvm::GlobalVariable *GV);
  void addGlobalArrayAttribs(GlobalArray *GA, llvm::PointerType *PT);
  bugle::GlobalArray *getGlobalArray(llvm::Value *V);

  ref<Expr> translateConstant(llvm::Constant *C);
  ref<Expr> doTranslateConstant(llvm::Constant *C);

  Type translateType(llvm::Type *T);
  Type translateArrayRangeType(llvm::Type *T);

  ref<Expr> translateGEP(ref<Expr> Ptr,
                         klee::gep_type_iterator begin,
                         klee::gep_type_iterator end,
                         std::function<ref<Expr>(llvm::Value *)> xlate);
  ref<Expr> translateBitCast(llvm::Type *SrcTy, llvm::Type *DestTy,
                             ref<Expr> Op);
  ref<Expr> translateArbitrary(Type t);

  ref<Expr> modelValue(llvm::Value *V, ref<Expr> E);
  Type getModelledType(llvm::Value *V);
  ref<Expr> unmodelValue(llvm::Value *V, ref<Expr> E);
  void computeValueModel(llvm::Value *Val, Var *Var,
                         llvm::ArrayRef<ref<Expr>> Assigns);

  Type defaultRange() {
    return ModelAllAsByteArray ? Type(Type::BV, 8) : Type(Type::Unknown);
  }

public:
  TranslateModule(llvm::Module *M, SourceLanguage SL) :
    BM(0), M(M), TD(M), SL(SL),
    NeedAdditionalByteArrayModels(false),
    ModelAllAsByteArray(false),
    NextModelAllAsByteArray(false) {}
  void addGPUEntryPoint(llvm::StringRef Name);
  void translate();
  bugle::Module *takeModule() { return BM; }

  friend class TranslateFunction;
};

}

#endif
