#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/BasicBlock.h"
#include "bugle/Expr.h"
#include "bugle/GlobalArray.h"
#include "bugle/Module.h"
#include "bugle/RaceInstrumenter.h"
#include "bugle/util/ErrorReporter.h"
#include "bugle/util/Functional.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

using namespace bugle;
using namespace llvm;

typedef std::vector<ref<Expr>> ExprVec;

TranslateFunction::SpecialFnMapTy
    TranslateFunction::SpecialFunctionMaps[TranslateModule::SL_Count];

// Appends at least the given basic block to the given list BBList (if not
// already present), so as to maintain the invariants that:
//  1) Each element of BBList is also a member of BBSet and vice versa;
//  2) Each element of BBList with a single predecessor must appear after
//     that predecessor.
// This invariant is important when translating basic blocks so that we do not
// see a use of an instruction in a basic block other than that currently
// being processed (i.e., in a phi node) before its definition.
static void AddBasicBlockInOrder(std::set<llvm::BasicBlock *> &BBSet,
                                 std::vector<llvm::BasicBlock *> &BBList,
                                 llvm::BasicBlock *BB) {
  if (BBSet.find(BB) != BBSet.end())
    return;

  // If the basic block has one predecessor, ...
  auto PredB = pred_begin(BB), PredI = PredB, PredE = pred_end(BB);
  if (PredI != PredE) {
    ++PredI;
    if (PredI == PredE) {
      // ... add that predecessor first.
      AddBasicBlockInOrder(BBSet, BBList, *PredB);
    }
  }

  BBSet.insert(BB);
  BBList.push_back(BB);
}

bool TranslateFunction::isSpecialFunction(TranslateModule::SourceLanguage SL,
                                          const std::string &fnName) {
  SpecialFnMapTy &SpecialFunctionMap = initSpecialFunctionMap(SL);
  return SpecialFunctionMap.Functions.find(fnName) !=
         SpecialFunctionMap.Functions.end();
}

void
TranslateFunction::addUninterpretedFunction(TranslateModule::SourceLanguage SL,
                                            const std::string &fnName) {
  SpecialFnMapTy &SpecialFunctionMap = initSpecialFunctionMap(SL);
  SpecialFunctionMap.Functions[fnName] =
      &TranslateFunction::handleUninterpretedFunction;
}

bool TranslateFunction::isAxiomFunction(StringRef fnName) {
  return fnName.startswith("__axiom");
}

bool TranslateFunction::isUninterpretedFunction(StringRef fnName) {
  return fnName.startswith("__uninterpreted_function_");
}

bool TranslateFunction::isSpecificationFunction(StringRef fnName) {
  return fnName.startswith("__spec");
}

bool TranslateFunction::isPreOrPostCondition(StringRef fnName) {
  if (fnName == "bugle_requires")
    return true;
  if (fnName == "__requires")
    return true;
  if (fnName == "__global_requires")
    return true;
  if (fnName == "bugle_ensures")
    return true;
  if (fnName == "__ensures")
    return true;
  if (fnName == "__global_ensures")
    return true;
  return false;
}

bool TranslateFunction::isBarrierFunction(TranslateModule::SourceLanguage SL,
                                          StringRef fnName) {
  return (SL == TranslateModule::SL_OpenCL || SL == TranslateModule::SL_CUDA) &&
         fnName == "bugle_barrier";
}

bool TranslateFunction::isGridBarrierFunction(TranslateModule::SourceLanguage SL,
                                              StringRef fnName) {
  return SL == TranslateModule::SL_CUDA && fnName == "bugle_grid_barrier";
}

bool TranslateFunction::isNormalFunction(TranslateModule::SourceLanguage SL,
                                         llvm::Function *F) {
  if (F->isIntrinsic())
    return false;
  if (isAxiomFunction(F->getName()))
    return false;
  if (isUninterpretedFunction(F->getName()))
    return false;
  if (isSpecialFunction(SL, F->getName()))
    return false;
  if (isSpecificationFunction(F->getName()))
    return false;
  if (isBarrierFunction(SL, F->getName()))
    return false;
  if (isGridBarrierFunction(SL, F->getName()))
    return false;
  return true;
}

bool TranslateFunction::isStandardEntryPoint(TranslateModule::SourceLanguage SL,
                                             StringRef fnName) {
  return SL == TranslateModule::SL_C && fnName == "main";
}

bool TranslateFunction::isRequiresFreshArrayFunction(llvm::StringRef fnName) {
  return fnName == "__requires_fresh_array";
}

llvm::StringRef
TranslateFunction::trimForRequiresFreshArrayFunction(llvm::StringRef fnName) {
  return fnName.startswith("__requires_fresh_array.") ? "__requires_fresh_array"
                                                      : fnName;
}

TranslateFunction::SpecialFnMapTy &
TranslateFunction::initSpecialFunctionMap(TranslateModule::SourceLanguage SL) {
  SpecialFnMapTy &SpecialFunctionMap = SpecialFunctionMaps[SL];
  if (SpecialFunctionMap.Functions.empty()) {
    auto &fns = SpecialFunctionMap.Functions;
    fns["bugle_assert"] = &TranslateFunction::handleAssert;
    fns["__assert"] = &TranslateFunction::handleAssert;
    fns["__global_assert"] = &TranslateFunction::handleGlobalAssert;
    fns["__candidate_assert"] = &TranslateFunction::handleCandidateAssert;
    fns["__candidate_global_assert"] = &TranslateFunction::handleCandidateGlobalAssert;
    fns["__invariant"] = &TranslateFunction::handleInvariant;
    fns["__global_invariant"] = &TranslateFunction::handleGlobalInvariant;
    fns["__candidate_invariant"] = &TranslateFunction::handleCandidateInvariant;
    fns["__candidate_global_invariant"] = &TranslateFunction::handleCandidateGlobalInvariant;
    fns["__non_temporal_loads_begin"] = &TranslateFunction::handleNonTemporalLoadsBegin;
    fns["__non_temporal_loads_end"] = &TranslateFunction::handleNonTemporalLoadsEnd;
    fns["bugle_assume"] = &TranslateFunction::handleAssume;
    fns["__assert_fail"] = &TranslateFunction::handleAssertFail;
    fns["bugle_requires"] = &TranslateFunction::handleRequires;
    fns["__requires"] = &TranslateFunction::handleRequires;
    fns["__global_requires"] = &TranslateFunction::handleGlobalRequires;
    fns["__requires_fresh_array"] = &TranslateFunction::handleRequiresFreshArray;
    fns["bugle_ensures"] = &TranslateFunction::handleEnsures;
    fns["__ensures"] = &TranslateFunction::handleEnsures;
    fns["__global_ensures"] = &TranslateFunction::handleGlobalEnsures;
    fns["__function_wide_invariant"] = &TranslateFunction::handleFunctionWideInvariant;
    fns["__function_wide_candidate_invariant"] = &TranslateFunction::handleFunctionWideCandidateInvariant;
    fns["__reads_from"] = &TranslateFunction::handleReadsFrom;
    fns["__reads_from_local"] = &TranslateFunction::handleReadsFrom;
    fns["__reads_from_global"] = &TranslateFunction::handleReadsFrom;
    fns["__writes_to"] = &TranslateFunction::handleWritesTo;
    fns["__writes_to_local"] = &TranslateFunction::handleWritesTo;
    fns["__writes_to_global"] = &TranslateFunction::handleWritesTo;
    fns["bugle_frexp_exp"] = &TranslateFunction::handleFrexpExp;
    fns["bugle_frexp_frac"] = &TranslateFunction::handleFrexpFrac;
    fns["__add_noovfl_char"] = &TranslateFunction::handleAddNoovflSigned;
    fns["__add_noovfl_short"] = &TranslateFunction::handleAddNoovflSigned;
    fns["__add_noovfl_int"] = &TranslateFunction::handleAddNoovflSigned;
    fns["__add_noovfl_long"] = &TranslateFunction::handleAddNoovflSigned;
    fns["__add_noovfl_unsigned_char"] = &TranslateFunction::handleAddNoovflUnsigned;
    fns["__add_noovfl_unsigned_short"] = &TranslateFunction::handleAddNoovflUnsigned;
    fns["__add_noovfl_unsigned_int"] = &TranslateFunction::handleAddNoovflUnsigned;
    fns["__add_noovfl_unsigned_long"] = &TranslateFunction::handleAddNoovflUnsigned;
    fns["__atomic_has_taken_value_local"] = &TranslateFunction::handleAtomicHasTakenValue;
    fns["__atomic_has_taken_value_global"] = &TranslateFunction::handleAtomicHasTakenValue;
    fns["__atomic_has_taken_value"] = &TranslateFunction::handleAtomicHasTakenValue;
    const unsigned NOOVFL_PREDICATE_MAX_ARITY = 20;
    const std::string tys[] = {"char", "short", "int", "long"};
    for (unsigned j = 0; j < 4; ++j) {
      std::string t = tys[j];
      for (unsigned i = 0; i <= NOOVFL_PREDICATE_MAX_ARITY; ++i) {
        std::string S = "__add_noovfl_unsigned_";
        llvm::raw_string_ostream SS(S);
        SS << t << "_" << i;
        fns[SS.str()] = &TranslateFunction::handleAddNoovflPredicate;
      }
    }
    fns["__add_char"] = &TranslateFunction::handleAdd;
    fns["__add_short"] = &TranslateFunction::handleAdd;
    fns["__add_int"] = &TranslateFunction::handleAdd;
    fns["__add_long"] = &TranslateFunction::handleAdd;
    fns["__add_unsigned_char"] = &TranslateFunction::handleAdd;
    fns["__add_unsigned_short"] = &TranslateFunction::handleAdd;
    fns["__add_unsigned_int"] = &TranslateFunction::handleAdd;
    fns["__add_unsigned_long"] = &TranslateFunction::handleAdd;
    fns["__ite_char"] = &TranslateFunction::handleIte;
    fns["__ite_short"] = &TranslateFunction::handleIte;
    fns["__ite_int"] = &TranslateFunction::handleIte;
    fns["__ite_long"] = &TranslateFunction::handleIte;
    fns["__ite_unsigned_char"] = &TranslateFunction::handleIte;
    fns["__ite_unsigned_short"] = &TranslateFunction::handleIte;
    fns["__ite_unsigned_int"] = &TranslateFunction::handleIte;
    fns["__ite_unsigned_long"] = &TranslateFunction::handleIte;
    fns["__ite_size_t"] = &TranslateFunction::handleIte;
    fns["__return_val_int"] = &TranslateFunction::handleReturnVal;
    fns["__return_val_int4"] = &TranslateFunction::handleReturnVal;
    fns["__return_val_bool"] = &TranslateFunction::handleReturnVal;
    fns["__return_val_ptr"] = &TranslateFunction::handleReturnVal;
    fns["__return_val_funptr"] = &TranslateFunction::handleReturnVal;
    fns["__old_int"] = &TranslateFunction::handleOld;
    fns["__old_bool"] = &TranslateFunction::handleOld;
    fns["__other_int"] = &TranslateFunction::handleOtherInt;
    fns["__other_bool"] = &TranslateFunction::handleOtherBool;
    fns["__other_ptr_base"] = &TranslateFunction::handleOtherPtrBase;
    fns["__implies"] = &TranslateFunction::handleImplies;
    fns["__enabled"] = &TranslateFunction::handleEnabled;
    fns["__dominator_enabled"] = &TranslateFunction::handleDominatorEnabled;
    fns["__read_local"] = &TranslateFunction::handleReadHasOccurred;
    fns["__read_global"] = &TranslateFunction::handleReadHasOccurred;
    fns["__read"] = &TranslateFunction::handleReadHasOccurred;
    fns["__write_local"] = &TranslateFunction::handleWriteHasOccurred;
    fns["__write_global"] = &TranslateFunction::handleWriteHasOccurred;
    fns["__write"] = &TranslateFunction::handleWriteHasOccurred;
    fns["__read_offset_bytes_local"] = &TranslateFunction::handleReadOffset;
    fns["__read_offset_bytes_global"] = &TranslateFunction::handleReadOffset;
    fns["__read_offset_bytes"] = &TranslateFunction::handleReadOffset;
    fns["__write_offset_bytes_local"] = &TranslateFunction::handleWriteOffset;
    fns["__write_offset_bytes_global"] = &TranslateFunction::handleWriteOffset;
    fns["__write_offset_bytes"] = &TranslateFunction::handleWriteOffset;
    fns["__ptr_base_local"] = &TranslateFunction::handlePtrBase;
    fns["__ptr_base_global"] = &TranslateFunction::handlePtrBase;
    fns["__ptr_base"] = &TranslateFunction::handlePtrBase;
    fns["__ptr_offset_bytes_local"] = &TranslateFunction::handlePtrOffset;
    fns["__ptr_offset_bytes_global"] = &TranslateFunction::handlePtrOffset;
    fns["__ptr_offset_bytes"] = &TranslateFunction::handlePtrOffset;
    fns["__array_snapshot_local"] = &TranslateFunction::handleArraySnapshot;
    fns["__array_snapshot_global"] = &TranslateFunction::handleArraySnapshot;
    fns["__array_snapshot"] = &TranslateFunction::handleArraySnapshot;

    if (SL == TranslateModule::SL_OpenCL || SL == TranslateModule::SL_CUDA) {
      const std::string opencl[] = {"__bugle_atomic_add",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_sub",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_xchg",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_float", "_global_float",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_min",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_max",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_and",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_or",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_xor",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_cmpxchg",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_inc",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    "__bugle_atomic_dec",
                                      "_local_int", "_local_uint",
                                      "_global_int", "_global_uint",
                                      "_local_long", "_local_unsigned_long",
                                      "_global_long", "_global_unsigned_long",
                                      "" ,
                                    ""};

      const std::string cuda[] = {"__atomicAdd",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int",
                                    "_float", "",
                                  "__atomicSub",
                                    "_int", "_unsigned_int", "",
                                  "__atomicExch",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int",
                                    "_float", "",
                                  "__atomicMin",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int", "",
                                  "__atomicMax",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int", "",
                                  "__atomicAnd",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int", "",
                                  "__atomicOr",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int", "",
                                  "__atomicXor",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int", "",
                                  "__atomicInc",
                                    "_unsigned_int", "",
                                  "__atomicDec",
                                    "_unsigned_int", "",
                                  "__atomicCAS",
                                    "_int", "_unsigned_int",
                                    "_unsigned_long_long_int", "",
                                  ""};

      const std::string *atomics =
          ((SL == TranslateModule::SL_OpenCL) ? opencl : cuda);

      int i = 0;
      while (atomics[i] != "") {
        std::string t = atomics[i];
        ++i;
        while (atomics[i] != "") {
          fns[t + atomics[i]] = &TranslateFunction::handleAtomic;
          ++i;
        }
        ++i;
      }
    }

    if (SL == TranslateModule::SL_OpenCL || SL == TranslateModule::SL_CUDA) {
      const unsigned BARRIER_INVARIANT_MAX_ARITY = 20;
      for (unsigned i = 0; i <= BARRIER_INVARIANT_MAX_ARITY; ++i) {
        std::string S = "__barrier_invariant_";
        llvm::raw_string_ostream SS(S);
        SS << i;
        fns[SS.str()] = &TranslateFunction::handleBarrierInvariant;
      }
      for (unsigned i = 0; i <= BARRIER_INVARIANT_MAX_ARITY; ++i) {
        std::string S = "__barrier_invariant_binary_";
        llvm::raw_string_ostream SS(S);
        SS << i;
        fns[SS.str()] = &TranslateFunction::handleBarrierInvariantBinary;
      }
    }

    if (SL == TranslateModule::SL_OpenCL) {
      fns["get_local_id"] = &TranslateFunction::handleGetLocalId;
      fns["get_group_id"] = &TranslateFunction::handleGetGroupId;
      fns["get_local_size"] = &TranslateFunction::handleGetLocalSize;
      fns["get_num_groups"] = &TranslateFunction::handleGetNumGroups;
      fns["get_global_offset"] = &TranslateFunction::handleGetGlobalOffset;
      fns["get_work_dim"] = &TranslateFunction::handleGetWorkDim;
      fns["get_image_width"] = &TranslateFunction::handleGetImageWidth;
      fns["get_image_height"] = &TranslateFunction::handleGetImageHeight;
      fns["__translate_sampler_initializer"] = &TranslateFunction::handleSamplerInitializer;
      {
        const std::string types[] = {"char", "uchar", "short", "ushort", "int",
                                     "uint", "long", "ulong", "float", "double",
                                     ""};
        for (unsigned i = 0; types[i] != ""; ++i) {
          for (unsigned width = 1; width <= 16; width *= 2) {
            std::string S;
            llvm::raw_string_ostream SS(S);
            if (width > 1) {
              SS << width;
            }
            fns["__bugle_async_work_group_copy_global_to_local_" + types[i] +
                SS.str()] = &TranslateFunction::handleAsyncWorkGroupCopy;
            fns["__bugle_async_work_group_copy_local_to_global_" + types[i] +
                SS.str()] = &TranslateFunction::handleAsyncWorkGroupCopy;
          }
        }
      }
      fns["__bugle_wait_group_events"] = &TranslateFunction::handleWaitGroupEvents;
    }

    if (SL == TranslateModule::SL_CUDA) {
      fns["cos"] = &TranslateFunction::handleCos;
      fns["sin"] = &TranslateFunction::handleSin;
      fns["sqrt"] = &TranslateFunction::handleSqrt;
      fns["sqrtf"] = &TranslateFunction::handleSqrt;
      fns["rsqrt"] = &TranslateFunction::handleRsqrt;
      fns["log2"] = &TranslateFunction::handleLog2;
      fns["exp2"] = &TranslateFunction::handleExp;
      fns["__clz"] = &TranslateFunction::handleCtlz;
    }

    auto &ints = SpecialFunctionMap.Intrinsics;
    ints[Intrinsic::ceil] = &TranslateFunction::handleCeil;
    ints[Intrinsic::ctpop] = &TranslateFunction::handleCtpop;
    ints[Intrinsic::cos] = &TranslateFunction::handleCos;
    ints[Intrinsic::ctlz] = &TranslateFunction::handleCtlz;
    ints[Intrinsic::exp] = &TranslateFunction::handleExp;
    ints[Intrinsic::exp2] = &TranslateFunction::handleExp2;
    ints[Intrinsic::fabs] = &TranslateFunction::handleFabs;
    ints[Intrinsic::fma] = &TranslateFunction::handleFma;
    ints[Intrinsic::fmuladd] = &TranslateFunction::handleFma;
    ints[Intrinsic::floor] = &TranslateFunction::handleFloor;
    ints[Intrinsic::log] = &TranslateFunction::handleLog;
    ints[Intrinsic::log10] = &TranslateFunction::handleLog10;
    ints[Intrinsic::log2] = &TranslateFunction::handleLog2;
    ints[Intrinsic::maxnum] = &TranslateFunction::handleFmax;
    ints[Intrinsic::minnum] = &TranslateFunction::handleFmin;
    ints[Intrinsic::pow] = &TranslateFunction::handlePow;
    ints[Intrinsic::powi] = &TranslateFunction::handlePowi;
    ints[Intrinsic::rint] = &TranslateFunction::handleRint;
    ints[Intrinsic::sin] = &TranslateFunction::handleSin;
    ints[Intrinsic::sqrt] = &TranslateFunction::handleSqrt;
    ints[Intrinsic::trunc] = &TranslateFunction::handleTrunc;
    ints[Intrinsic::uadd_with_overflow] = &TranslateFunction::handleUaddOvl;
    ints[Intrinsic::sadd_with_overflow] = &TranslateFunction::handleSaddOvl;
    ints[Intrinsic::usub_with_overflow] = &TranslateFunction::handleUsubOvl;
    ints[Intrinsic::ssub_with_overflow] = &TranslateFunction::handleSsubOvl;
    ints[Intrinsic::dbg_value] = &TranslateFunction::handleNoop;
    ints[Intrinsic::dbg_declare] = &TranslateFunction::handleNoop;
    ints[Intrinsic::memset] = &TranslateFunction::handleMemset;
    ints[Intrinsic::memcpy] = &TranslateFunction::handleMemcpy;
    ints[Intrinsic::trap] = &TranslateFunction::handleTrap;
    ints[Intrinsic::lifetime_start] = &TranslateFunction::handleNoop;
    ints[Intrinsic::lifetime_end] = &TranslateFunction::handleNoop;
  }
  return SpecialFunctionMap;
}

void TranslateFunction::specifyZeroDimensions(unsigned PtrArgs) {
  ArraySpec &AS = TM->GPUArraySizes[F->getName()];
  if (AS.size() != PtrArgs) {
    std::string msg; llvm::raw_string_ostream msgS(msg);
    msgS << "Expected " << PtrArgs << " array sizes for " << F->getName()
         << " got " << AS.size();
    ErrorReporter::reportParameterError(msgS.str());
  }

  auto ArraySize = AS.begin();
  for (auto &Arg : F->args()) {
    if (!Arg.getType()->isPointerTy() ||
        Arg.getType()->getPointerElementType()->isFunctionTy())
      continue;
    if (ArraySize->first) {
      GlobalArray *GA = TM->getGlobalArray(&Arg, /*IsParameter=*/true);
      auto *ElTy = Arg.getType()->getPointerElementType();
      uint64_t ElementSize = TM->TD.getTypeAllocSize(ElTy);
      uint64_t size = ArraySize->second;
      if (size % ElementSize != 0) {
        std::string msg; llvm::raw_string_ostream msgS(msg);
        msgS << "Array size " << size << " not a multiple of element size "
             << ElementSize;
        ErrorReporter::reportParameterError(msgS.str());
      }
      GA->updateZeroDimension(size / ElementSize);
    }
    ++ArraySize;
  }
}

void TranslateFunction::extractStructArrays(llvm::Value *V) {
  auto STy = cast<StructType>(V->getType());
  auto Name = (V->getName() + ".coerce").str();
  for (unsigned i = 0; i < STy->getNumElements(); ++i) {
    auto Index = ArrayRef<unsigned>(i);
    auto E = ExtractValueInst::Create(V, Index, Name + Twine(i));
    TM->StructMap[F]->push_back(E);
    if (E->getType()->isStructTy())
      extractStructArrays(E);
  }
}

void TranslateFunction::createStructArrays() {
  std::vector<llvm::Instruction *> *SV = TM->StructMap[F];
  if (!SV) {
    SV = new std::vector<llvm::Instruction *>();
    TM->StructMap[F] = SV;
    for (auto &Arg : F->args()) {
      if (Arg.getType()->isStructTy())
        extractStructArrays(&Arg);
    }
  }

  BasicBlock *BB = new BasicBlock("");
  unsigned PtrSize = TM->TD.getPointerSizeInBits();

  for (auto *I : *SV) {
    translateInstruction(BB, I);

    if (!I->getType()->isPointerTy() ||
        I->getType()->getPointerElementType()->isFunctionTy())
      continue;

    GlobalArray *GA = TM->getGlobalArray(I, /*IsParameter=*/true);
    if (TM->SL == TranslateModule::SL_CUDA)
      GA->addAttribute("global");
    auto PtrExpr = PointerExpr::create(GlobalArrayRefExpr::create(GA),
                                       BVConstExpr::createZero(PtrSize));
    BF->addRequires(EqExpr::create(ValueExprMap[I], PtrExpr), nullptr);
  }

  delete BB;
}

void TranslateFunction::translate() {
  if (isGPUEntryPoint || isStandardEntryPoint(TM->SL, F->getName()))
    BF->setEntryPoint(true);

  if (isGPUEntryPoint)
    BF->addAttribute("kernel");

  if (isBarrierFunction(TM->SL, F->getName()))
    BF->addAttribute("barrier");

  if (isGridBarrierFunction(TM->SL, F->getName()))
    BF->addAttribute("grid_barrier");

  if (isSpecificationFunction(F->getName()))
    BF->setSpecification(true);

  unsigned PtrSize = TM->TD.getPointerSizeInBits();
  unsigned PtrArgs = 0;
  for (auto &Arg : F->args()) {
    if (isGPUEntryPoint && Arg.getType()->isPointerTy() &&
        !Arg.getType()->getPointerElementType()->isFunctionTy()) {
      GlobalArray *GA = TM->getGlobalArray(&Arg, /*IsParameter=*/true);
      ++PtrArgs;
      if (TM->SL == TranslateModule::SL_CUDA)
        GA->addAttribute("global");
      ValueExprMap[&Arg] = PointerExpr::create(
          GlobalArrayRefExpr::create(GA), BVConstExpr::createZero(PtrSize));
    } else {
      Var *V = BF->addArgument(TM->getModelledType(&Arg),
                               TranslateModule::getSourceName(&Arg, F));
      ValueExprMap[&Arg] = TM->unmodelValue(&Arg, VarRefExpr::create(V));
    }
  }

  if (isGPUEntryPoint)
    createStructArrays();

  if (isGPUEntryPoint &&
      TM->GPUArraySizes.find(F->getName()) != TM->GPUArraySizes.end())
    specifyZeroDimensions(PtrArgs);

  if (BF->return_begin() != BF->return_end())
    ReturnVar = *BF->return_begin();

  std::set<llvm::BasicBlock *> BBSet;
  std::vector<llvm::BasicBlock *> BBList;

  for (auto &BB : *F) {
    AddBasicBlockInOrder(BBSet, BBList, &BB);
    BasicBlockMap[&BB] = BF->addBasicBlock(BB.getName());
  }

  for (auto *BBB : BBList) {
    Stmt *AS = AssertStmt::createBlockSourceLoc(extractSourceLocsForBlock(BBB));
    BasicBlockMap[BBB]->addStmt(AS);
    translateBasicBlock(BasicBlockMap[BBB], BBB);
  }

  // If we're modelling everything as a byte array, don't bother to compute
  // value models.
  if (TM->ModelAllAsByteArray)
    return;

  // For each phi we encountered in the function, see if we can model it.
  for (auto &Phi : PhiAssignsMap) {
    ExprVec assigns;
    std::set<llvm::PHINode *> foundPhiNodes;
    foundPhiNodes.insert(Phi.first);
    computeClosure(Phi.second, foundPhiNodes, assigns);
    TM->computeValueModel(Phi.first, PhiVarMap[Phi.first], assigns);
  }

  // See if we can model the return value. This requires the function to have
  // a body.
  if (!BBList.empty())
    TM->computeValueModel(F, nullptr, ReturnVals);
}

void TranslateFunction::computeClosure(std::vector<PhiPair> &currentAssigns,
                                       std::set<llvm::PHINode *> &foundPhiNodes,
                                       ExprVec &assigns) {
  for (auto &Pair : currentAssigns) {
    // See if this phi node is referring to another phi node, either directly
    // or through a number of getelementptr instructions. Compute the transitive
    // closure if such a phi node exists.
    auto operand = Pair.first;

    while (isa<GetElementPtrInst>(operand))
      operand = cast<GetElementPtrInst>(operand)->getPointerOperand();

    if (auto PN = dyn_cast<PHINode>(operand)) {
      if (foundPhiNodes.find(PN) == foundPhiNodes.end()) {
        foundPhiNodes.insert(PN);
        computeClosure(PhiAssignsMap[PN], foundPhiNodes, assigns);
      }
    } else {
      assigns.push_back(Pair.second);
    }
  }
}

ref<Expr> TranslateFunction::translateValue(llvm::Value *V,
                                            bugle::BasicBlock *BBB) {
  if (isa<Instruction>(V) || isa<Argument>(V)) {
    auto MI = ValueExprMap.find(V);
    assert(MI != ValueExprMap.end());
    return MI->second;
  }

  if (isa<UndefValue>(V)) {
    ref<Expr> E = HavocExpr::create(TM->translateType(V->getType()));
    BBB->addEvalStmt(E, currentSourceLocs);
    return E;
  }

  if (auto C = dyn_cast<Constant>(V))
    return TM->translateConstant(C);

  if (V->getType()->isMetadataTy()) {
    // ignore metadata values
    return nullptr;
  }

  if (isa<InlineAsm>(V))
    ErrorReporter::reportImplementationLimitation(
        "Inline assembly not supported");

  ErrorReporter::reportImplementationLimitation("Unsupported value");
}

Var *TranslateFunction::getPhiVariable(llvm::PHINode *PN) {
  auto &V = PhiVarMap[PN];
  if (V)
    return V;

  V = BF->addLocal(TM->getModelledType(PN), PN->getName());
  return V;
}

void TranslateFunction::addPhiAssigns(bugle::BasicBlock *BBB,
                                      llvm::BasicBlock *Pred,
                                      llvm::BasicBlock *Succ) {
  std::vector<Var *> Vars;
  ExprVec Exprs;
  for (auto &PN : Succ->phis()) {
    int idx = PN.getBasicBlockIndex(Pred);
    assert(idx != -1 && "No phi index?");

    Vars.push_back(getPhiVariable(&PN));
    auto Val =
        TM->modelValue(&PN, translateValue(PN.getIncomingValue(idx), BBB));
    Exprs.push_back(Val);
    PhiPair pair = std::make_pair(PN.getIncomingValue(idx), Val);
    PhiAssignsMap[&PN].push_back(pair);
  }

  if (!Vars.empty())
    BBB->addStmt(VarAssignStmt::create(Vars, Exprs));
}

SourceLocsRef
TranslateFunction::extractSourceLocsForBlock(llvm::BasicBlock *BB) {
  SourceLocsRef sourcelocs;
  for (auto &I : *BB) {
    // Skip over llvm.dbg.value, as these may point to the point of declaration
    // of a variable, which may be outside the current basic block.
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::dbg_value)
        continue;
    }
    sourcelocs = extractSourceLocs(&I);
    if (sourcelocs != nullptr)
      break;
  }
  return sourcelocs;
}

SourceLocsRef
TranslateFunction::extractSourceLocs(llvm::Instruction *I) {
  SourceLocs *sourcelocs = nullptr;
  if (MDNode *mdnode = I->getMetadata("dbg")) {
    sourcelocs = new SourceLocs();
    DILocation *Loc = cast<DILocation>(mdnode);
    do {
      sourcelocs->push_back(
          SourceLoc(Loc->getLine(), Loc->getColumn(),
                    Loc->getFilename().str(), Loc->getDirectory().str()));
      Loc = Loc->getInlinedAt();
    } while (Loc);
  }
  return SourceLocsRef(sourcelocs);
}

ref<Expr> TranslateFunction::handleNoop(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  return nullptr;
}

ref<Expr> TranslateFunction::handleAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const ExprVec &Args) {
  BBB->addStmt(AssertStmt::create(Expr::createNeZero(Args[0]), /*global=*/false,
                                  /*candidate=*/false, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleGlobalAssert(bugle::BasicBlock *BBB,
                                                llvm::CallInst *CI,
                                                const ExprVec &Args) {
  BBB->addStmt(AssertStmt::create(Expr::createNeZero(Args[0]), /*global=*/true,
                                  /*candidate=*/false, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleCandidateAssert(bugle::BasicBlock *BBB,
                                                   llvm::CallInst *CI,
                                                   const ExprVec &Args) {
  BBB->addStmt(AssertStmt::create(Expr::createNeZero(Args[0]), /*global=*/false,
                                  /*candidate=*/true, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleCandidateGlobalAssert(bugle::BasicBlock *BBB,
                                                         llvm::CallInst *CI,
                                                         const ExprVec &Args) {
  BBB->addStmt(AssertStmt::create(Expr::createNeZero(Args[0]), /*global=*/true,
                                  /*candidate=*/true, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleInvariant(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                             const ExprVec &Args) {
  BBB->addStmt(
      AssertStmt::createInvariant(Expr::createNeZero(Args[0]), /*global=*/false,
                                  /*candidate=*/false, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleGlobalInvariant(bugle::BasicBlock *BBB,
                                                   llvm::CallInst *CI,
                                                   const ExprVec &Args) {
  BBB->addStmt(
      AssertStmt::createInvariant(Expr::createNeZero(Args[0]), /*global=*/true,
                                  /*candidate=*/false, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleCandidateInvariant(bugle::BasicBlock *BBB,
                                                      llvm::CallInst *CI,
                                                      const ExprVec &Args) {
  BBB->addStmt(
      AssertStmt::createInvariant(Expr::createNeZero(Args[0]), /*global=*/false,
                                  /*candidate=*/true, currentSourceLocs));
  return nullptr;
}

ref<Expr>
TranslateFunction::handleCandidateGlobalInvariant(bugle::BasicBlock *BBB,
                                                  llvm::CallInst *CI,
                                                  const ExprVec &Args) {
  BBB->addStmt(
      AssertStmt::createInvariant(Expr::createNeZero(Args[0]), /*global=*/true,
                                  /*candidate=*/true, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleNonTemporalLoadsBegin(bugle::BasicBlock *BBB,
                                                         llvm::CallInst *CI,
                                                         const ExprVec &Args) {
  if (!LoadsAreTemporal)
    ErrorReporter::reportFatalError("Nested __non_temporal_loads_begin");
  LoadsAreTemporal = false;
  return nullptr;
}

ref<Expr> TranslateFunction::handleNonTemporalLoadsEnd(bugle::BasicBlock *BBB,
                                                       llvm::CallInst *CI,
                                                       const ExprVec &Args) {
  if (LoadsAreTemporal)
    ErrorReporter::reportFatalError(
        "__non_temporal_loads_end without __non_temporal_loads_begin");
  LoadsAreTemporal = true;
  return nullptr;
}

ref<Expr> TranslateFunction::handleAssertFail(bugle::BasicBlock *BBB,
                                              llvm::CallInst *CI,
                                              const ExprVec &Args) {
  BBB->addStmt(
      AssertStmt::create(BoolConstExpr::create(false), /*global=*/false,
                         /*candidate=*/false, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleAssume(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const ExprVec &Args) {
  BBB->addStmt(AssumeStmt::create(Expr::createNeZero(Args[0])));
  return nullptr;
}

ref<Expr> TranslateFunction::handleRequires(bugle::BasicBlock *BBB,
                                            llvm::CallInst *CI,
                                            const ExprVec &Args) {
  BF->addRequires(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return nullptr;
}

ref<Expr> TranslateFunction::handleGlobalRequires(bugle::BasicBlock *BBB,
                                                  llvm::CallInst *CI,
                                                  const ExprVec &Args) {
  BF->addGlobalRequires(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return nullptr;
}

ref<Expr> TranslateFunction::handleRequiresFreshArray(bugle::BasicBlock *BBB,
                                                      llvm::CallInst *CI,
                                                      const ExprVec &Args) {
  GlobalArray *GA = TM->getGlobalArray(CI, /*IsParameter=*/true);

  if (TM->SL == TranslateModule::SL_CUDA)
    GA->addAttribute("global");

  return PointerExpr::create(
      GlobalArrayRefExpr::create(GA),
      BVConstExpr::createZero(TM->TD.getPointerSizeInBits()));
}

ref<Expr> TranslateFunction::handleEnsures(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  BF->addEnsures(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return nullptr;
}

ref<Expr> TranslateFunction::handleGlobalEnsures(bugle::BasicBlock *BBB,
                                                 llvm::CallInst *CI,
                                                 const ExprVec &Args) {
  BF->addGlobalEnsures(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return nullptr;
}

bool TranslateFunction::isLegalFunctionWideInvariantValue(Value *V) {
  if (isa<PHINode>(V)) {
    return false;
  } else if (isa<Instruction>(V)) {
    auto MI = ValueExprMap.find(V);
    assert(MI != ValueExprMap.end());
    if (isa<CallExpr>(MI->second))
      return false;
  }

  if (auto *U = dyn_cast<User>(V)) {
    for (auto &Op : U->operands()) {
      if (!isLegalFunctionWideInvariantValue(Op))
        return false;
    }
  }
  return true;
}

void TranslateFunction::checkFunctionWideInvariant(llvm::CallInst *CI) {
  if (!isLegalFunctionWideInvariantValue(CI->getArgOperand(0)))
    ErrorReporter::reportFatalError(
        "Function-wide invariants can only be constant expressions over "
        "read-only function arguments");
  if (!isa<ReturnInst>(CI->getParent()->getTerminator()))
    ErrorReporter::reportFatalError(
        "Function-wide invariants must occur at the end of a function");
}

ref<Expr> TranslateFunction::handleFunctionWideInvariant(bugle::BasicBlock *BBB,
                                                         llvm::CallInst *CI,
                                                         const ExprVec &Args) {
  checkFunctionWideInvariant(CI);
  BF->addProcedureWideInvariant(Expr::createNeZero(Args[0]),
                                extractSourceLocs(CI));
  return nullptr;
}

ref<Expr>
TranslateFunction::handleFunctionWideCandidateInvariant(bugle::BasicBlock *BBB,
                                                        llvm::CallInst *CI,
                                                        const ExprVec &Args) {
  checkFunctionWideInvariant(CI);
  BF->addProcedureWideCandidateInvariant(Expr::createNeZero(Args[0]),
                                         extractSourceLocs(CI));
  return nullptr;
}

ref<Expr> TranslateFunction::handleReadsFrom(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                             const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  BF->addModifies(AccessHasOccurredExpr::create(arrayIdExpr, false),
                  extractSourceLocs(CI));

  if (TM->RaceInst == bugle::RaceInstrumenter::Original) {
    ref<Expr> access = AccessOffsetExpr::create(
        arrayIdExpr, TM->TD.getPointerSizeInBits(), false);
    BF->addModifies(access, extractSourceLocs(CI));
  }

  if (arrayIdExpr->getType().range().isKind(Type::Unknown))
    TM->NextModelAllAsByteArray = true;

  return nullptr;
}

ref<Expr> TranslateFunction::handleWritesTo(bugle::BasicBlock *BBB,
                                            llvm::CallInst *CI,
                                            const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  BF->addModifies(AccessHasOccurredExpr::create(arrayIdExpr, true),
                  extractSourceLocs(CI));

  if (TM->RaceInst == bugle::RaceInstrumenter::Original) {
    ref<Expr> access = AccessOffsetExpr::create(
        arrayIdExpr, TM->TD.getPointerSizeInBits(), true);
    BF->addModifies(access, extractSourceLocs(CI));
  }

  BF->addModifies(UnderlyingArrayExpr::create(arrayIdExpr),
                  extractSourceLocs(CI));

  if (arrayIdExpr->getType().range().isKind(Type::Unknown))
    TM->NextModelAllAsByteArray = true;

  return nullptr;
}

ref<Expr> TranslateFunction::handleOld(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  return OldExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleReturnVal(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                             const ExprVec &Args) {
  if (TM->getModelledType(CI).width != ReturnVar->getType().width)
    ErrorReporter::reportFatalError(
        "Type of __return_val function does not match return type");

  return TM->unmodelValue(F, VarRefExpr::create(ReturnVar));
}

ref<Expr> TranslateFunction::handleOtherInt(bugle::BasicBlock *BBB,
                                            llvm::CallInst *CI,
                                            const ExprVec &Args) {
  return OtherIntExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleOtherBool(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                             const ExprVec &Args) {
  ref<Expr> otherBool = OtherBoolExpr::create(BVToBoolExpr::create(Args[0]));
  return BoolToBVExpr::create(otherBool);
}

ref<Expr> TranslateFunction::handleOtherPtrBase(bugle::BasicBlock *BBB,
                                                llvm::CallInst *CI,
                                                const ExprVec &Args) {
  Type range = Expr::getPointerRange(Args[0], TM->defaultRange());

  if (range.isKind(Type::Unknown))
    TM->NextModelAllAsByteArray = true;

  return OtherPtrBaseExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleImplies(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  ref<Expr> implies = ImpliesExpr::create(BVToBoolExpr::create(Args[0]),
                                          BVToBoolExpr::create(Args[1]));
  return BoolToBVExpr::create(implies);
}

ref<Expr> TranslateFunction::handleEnabled(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  ref<Expr> varRef =
      SpecialVarRefExpr::create(bugle::Type(bugle::Type::Bool), "__enabled");
  return BoolToBVExpr::create(varRef);
}

ref<Expr> TranslateFunction::handleDominatorEnabled(bugle::BasicBlock *BBB,
                                                    llvm::CallInst *CI,
                                                     const ExprVec &Args) {
  ref<Expr> varRef = SpecialVarRefExpr::create(
      bugle::Type(bugle::Type::Bool), "__dominator_enabled");
  return BoolToBVExpr::create(varRef);
}

ref<Expr> TranslateFunction::handleReadHasOccurred(bugle::BasicBlock *BBB,
                                                   llvm::CallInst *CI,
                                                   const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result =
      BoolToBVExpr::create(AccessHasOccurredExpr::create(arrayIdExpr, false));

  if (arrayIdExpr->getType().range().isKind(Type::Unknown))
    TM->NextModelAllAsByteArray = true;

  return result;
}

ref<Expr> TranslateFunction::handleWriteHasOccurred(bugle::BasicBlock *BBB,
                                                    llvm::CallInst *CI,
                                                    const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result =
      BoolToBVExpr::create(AccessHasOccurredExpr::create(arrayIdExpr, true));

  if (arrayIdExpr->getType().range().isKind(Type::Unknown))
    TM->NextModelAllAsByteArray = true;

  return result;
}

ref<Expr> TranslateFunction::handleReadOffset(bugle::BasicBlock *BBB,
                                              llvm::CallInst *CI,
                                              const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result = AccessOffsetExpr::create(
      arrayIdExpr, TM->TD.getPointerSizeInBits(), false);
  Type range = arrayIdExpr->getType().range();

  if (range.isKind(Type::BV)) {
    if (range.width > 8) {
      result = BVMulExpr::create(
          BVConstExpr::create(TM->TD.getPointerSizeInBits(), range.width / 8),
          result);
    }
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (arrayIdExpr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }

  return result;
}

ref<Expr> TranslateFunction::handleWriteOffset(bugle::BasicBlock *BBB,
                                               llvm::CallInst *CI,
                                               const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result = AccessOffsetExpr::create(
      arrayIdExpr, TM->TD.getPointerSizeInBits(), true);
  Type range = arrayIdExpr->getType().range();

  if (range.isKind(Type::BV)) {
    if (range.width > 8) {
      result = BVMulExpr::create(
          BVConstExpr::create(TM->TD.getPointerSizeInBits(), range.width / 8),
          result);
    }
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (arrayIdExpr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }

  return result;
}

ref<Expr> TranslateFunction::handlePtrOffset(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                             const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result = ArrayOffsetExpr::create(Args[0]);

  if (!arrayIdExpr->getType().range().isKind(Type::BV)) {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (arrayIdExpr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }

  return result;
}

ref<Expr> TranslateFunction::handlePtrBase(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());

  if (arrayIdExpr->getType().range().isKind(Type::Unknown))
    TM->NextModelAllAsByteArray = true;

  return arrayIdExpr;
}

ref<Expr> TranslateFunction::handleArraySnapshot(bugle::BasicBlock *BBB,
                                                 llvm::CallInst *CI,
                                                 const ExprVec &Args) {
  ref<Expr> dstArrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> srcArrayIdExpr = ArrayIdExpr::create(Args[1], TM->defaultRange());
  ref<Expr> E = ArraySnapshotExpr::create(dstArrayIdExpr, srcArrayIdExpr);
  BBB->addEvalStmt(E, currentSourceLocs);
  if (dstArrayIdExpr->getType().range().isKind(Type::Unknown) ||
      srcArrayIdExpr->getType().range().isKind(Type::Unknown) ||
      dstArrayIdExpr->getType().range() != srcArrayIdExpr->getType().range())
    TM->NextModelAllAsByteArray = true;

  return nullptr;
}

ref<Expr> TranslateFunction::handleAtomic(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const ExprVec &Args) {
  assert(Args.size() > 0);
  ref<Expr> Ptr = Args[0],
            PtrArr = ArrayIdExpr::create(Ptr, TM->defaultRange()),
            PtrOfs = ArrayOffsetExpr::create(Ptr);
  Type ArrRangeTy = PtrArr->getType().range();
  Type AtomicTy = TM->translateType(CI->getType());
  assert(AtomicTy.width % 32 == 0);

  ExprVec AtomicArgs;
  for (size_t i = 1; i < Args.size(); ++i)
    AtomicArgs.push_back(Args[i]);

  ref<Expr> result;
  ref<Expr> Div;
  // If ArrRangeTy is Any, then we are using a null pointer.
  if (ArrRangeTy.isKind(Type::BV) && AtomicTy.width % ArrRangeTy.width == 0 &&
      !(Div = Expr::createExactBVSDiv(PtrOfs, ArrRangeTy.width / 8)).isNull()) {
    ExprVec Elems;
    unsigned NumElems = AtomicTy.width / ArrRangeTy.width;
    for (unsigned i = 0; i < NumElems; ++i) {
      ref<Expr> PartOfs = BVAddExpr::create(
          Div, BVConstExpr::create(Div->getType().width, i));
      ref<Expr> E = AtomicExpr::create(PtrArr, PartOfs, AtomicArgs,
                                       CI->getCalledFunction()->getName(),
                                       NumElems, i + 1);
      BBB->addEvalStmt(E, currentSourceLocs);
      Elems.push_back(E);
    }
    result = Expr::createBVConcatN(Elems);
  } else if (ArrRangeTy == Type(Type::Any)) {
    BBB->addStmt(AssertStmt::createBadAccess(currentSourceLocs));
    // The result is irrelevant, but the caller requires one.
    result = BVConstExpr::createZero(AtomicTy.width);
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (PtrArr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
    // The result is irrelevant, but the caller requires one.
    result = BVConstExpr::createZero(AtomicTy.width);
  }

  return result;
}

ref<Expr> TranslateFunction::handleBarrierInvariant(bugle::BasicBlock *BBB,
                                                    llvm::CallInst *CI,
                                                    const ExprVec &Args) {
  assert(CI->getNumArgOperands() > 1);

  auto *&BF = BarrierInvariants[CI->getNumArgOperands()];
  if (BF == nullptr) {
    llvm::Function *F = CI->getCalledFunction();
    std::string S = F->getName().str();
    llvm::raw_string_ostream SS(S);
    SS << (CI->getNumArgOperands() - 1);
    BF = TM->BM->addFunction(SS.str(), TM->getSourceFunctionName(F));

    for (size_t i = 0; i < Args.size(); ++i) {
      std::string S;
      llvm::raw_string_ostream SS(S);
      if (i == 0) {
        SS << "expr";
      } else {
        SS << "instantiation";
        SS << i;
      }
      BF->addArgument(Args[i]->getType(), SS.str());
    }

    BF->addAttribute("barrier_invariant");
  }

  BBB->addStmt(CallStmt::create(BF, Args, currentSourceLocs));
  return nullptr;
}

ref<Expr>
TranslateFunction::handleBarrierInvariantBinary(bugle::BasicBlock *BBB,
                                                llvm::CallInst *CI,
                                                const ExprVec &Args) {
  assert(CI->getNumArgOperands() > 1);
  assert(CI->getNumArgOperands() % 2 != 0 &&
         "__barrier_invariant_binary not followed by a sequence of pairs");

  auto *&BF = BinaryBarrierInvariants[CI->getNumArgOperands()];
  if (BF == nullptr) {
    llvm::Function *F = CI->getCalledFunction();
    std::string S = F->getName().str();
    llvm::raw_string_ostream SS(S);
    SS << ((CI->getNumArgOperands() - 1) / 2);
    BF = TM->BM->addFunction(SS.str(), TM->getSourceFunctionName(F));

    for (size_t i = 0; i < Args.size(); ++i) {
      std::string S;
      llvm::raw_string_ostream SS(S);
      if (i == 0) {
        SS << "expr";
      } else {
        SS << "instantiation";
        SS << (i / 2);
        SS << "_";
        SS << ((i % 2) == 0 ? 2 : 1);
      }
      BF->addArgument(Args[i]->getType(), SS.str());
    }

    BF->addAttribute("binary_barrier_invariant");
  }

  BBB->addStmt(CallStmt::create(BF, Args, currentSourceLocs));
  return nullptr;
}

ref<Expr> TranslateFunction::handleMemset(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const ExprVec &Args) {
  // Args[0] == cast<MemSetInst>(CI)->getDest()
  // Args[1] == cast<MemSetInst>(CI)->getValue()
  // Args[2] == cast<MemSetInst>(CI)->getLength()
  auto Length = dyn_cast<BVConstExpr>(Args[2]);
  if (!Length) {
    // Could emit a loop
    ErrorReporter::reportImplementationLimitation(
        "memset with non-integer constant length not supported");
  }

  auto Value = dyn_cast<BVConstExpr>(Args[1]);
  if (!Value) {
    // Could deal with expr
    ErrorReporter::reportImplementationLimitation(
        "memset with non-integer constant value not supported");
  }

  ref<Expr> Dst = Args[0],
            DstPtrArr = ArrayIdExpr::create(Dst, TM->defaultRange()),
            DstPtrOfs = ArrayOffsetExpr::create(Dst);
  unsigned Len = Length->getValue().getZExtValue();
  unsigned Val = Value->getValue().getZExtValue();
  Type DstRangeTy = DstPtrArr->getType().range();

  if (DstRangeTy == Type(Type::Any)) {
    BBB->addStmt(AssertStmt::createBadAccess(currentSourceLocs));
    return nullptr;
  }

  assert(DstRangeTy.width % 8 == 0);
  assert(DstRangeTy == Type(Type::Unknown) || DstRangeTy.width != 0);
  ref<Expr> DstDiv = Expr::createExactBVSDiv(DstPtrOfs, DstRangeTy.width / 8);
  // Handle when Len can be rewritten as an integral number of element writes
  // Special case if Val is 0
  if (DstRangeTy != Type(Type::Unknown) && !DstDiv.isNull() &&
      (Len % (DstRangeTy.width / 8) == 0) &&
      (Val == 0 || DstRangeTy.width == 8)) {
    unsigned NumElements = Len / (DstRangeTy.width / 8);
    for (unsigned i = 0; i != NumElements; ++i) {
      ref<Expr> ValExpr = BVConstExpr::create(DstRangeTy.width, Val);
      if (DstRangeTy.isKind(Type::Pointer))
        ValExpr = SafeBVToPtrExpr::create(ValExpr->getType().width, ValExpr);
      ref<Expr> StoreOfs = BVAddExpr::create(
          DstDiv, BVConstExpr::create(Dst->getType().width, i));
      BBB->addEvalStmt(ValExpr, currentSourceLocs);
      BBB->addStmt(
          StoreStmt::create(DstPtrArr, StoreOfs, ValExpr, currentSourceLocs));
    }
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (DstPtrArr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }

  return nullptr;
}

ref<Expr> TranslateFunction::handleMemcpy(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const ExprVec &Args) {
  // Args[0] == cast<MemCpyInst>(CI)->getDest()
  // Args[1] == cast<MemCpyInst>(CI)->getSource()
  // Args[2] == cast<MemCpyInst>(CI)->getLength()
  auto Length = dyn_cast<BVConstExpr>(Args[2]);
  if (!Length) {
    // Could emit a loop
    ErrorReporter::reportImplementationLimitation(
        "memcpy with non-integer constant length not supported");
  }

  ref<Expr> Src = Args[1], Dst = Args[0],
            SrcPtrArr = ArrayIdExpr::create(Src, TM->defaultRange()),
            DstPtrArr = ArrayIdExpr::create(Dst, TM->defaultRange()),
            SrcPtrOfs = ArrayOffsetExpr::create(Src),
            DstPtrOfs = ArrayOffsetExpr::create(Dst);
  unsigned Len = Length->getValue().getZExtValue();
  Type SrcRangeTy = SrcPtrArr->getType().range(),
       DstRangeTy = DstPtrArr->getType().range();

  if (DstRangeTy == Type(Type::Any)) {
    BBB->addStmt(AssertStmt::createBadAccess(currentSourceLocs));
    return nullptr;
  }

  if (SrcRangeTy == Type(Type::Any)) {
    BBB->addStmt(AssertStmt::createBadAccess(currentSourceLocs));
    return nullptr;
  }

  assert(SrcRangeTy.width % 8 == 0);
  assert(DstRangeTy.width % 8 == 0);
  assert(SrcRangeTy == Type(Type::Unknown) || SrcRangeTy.width != 0);
  ref<Expr> SrcDiv = Expr::createExactBVSDiv(SrcPtrOfs, SrcRangeTy.width / 8);
  ref<Expr> DstDiv = Expr::createExactBVSDiv(DstPtrOfs, DstRangeTy.width / 8);
  // Handle matching source and destination range types where Len can be
  // rewritten as an integral number of element read/writes
  if (SrcRangeTy == DstRangeTy && SrcRangeTy != Type(Type::Unknown) &&
      !SrcDiv.isNull() && !DstDiv.isNull() &&
      (Len % (SrcRangeTy.width / 8) == 0)) {
    unsigned NumElements = Len / (SrcRangeTy.width / 8);
    for (unsigned i = 0; i != NumElements; ++i) {
      ref<Expr> LoadOfs = BVAddExpr::create(
          SrcDiv, BVConstExpr::create(Src->getType().width, i));
      ref<Expr> Val =
          LoadExpr::create(SrcPtrArr, LoadOfs, SrcRangeTy, LoadsAreTemporal);
      ref<Expr> StoreOfs = BVAddExpr::create(
          DstDiv, BVConstExpr::create(Dst->getType().width, i));
      BBB->addEvalStmt(Val, currentSourceLocs);
      BBB->addStmt(
          StoreStmt::create(DstPtrArr, StoreOfs, Val, currentSourceLocs));
    }
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (SrcPtrArr->computeArrayCandidates(Globals) &&
        DstPtrArr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }

  return nullptr;
}

ref<Expr> TranslateFunction::handleTrap(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  BBB->addStmt(
      AssertStmt::create(BoolConstExpr::create(false), /*global=*/false,
                         /*candidate=*/false, currentSourceLocs));
  return nullptr;
}

static std::string mkDimName(const std::string &prefix, ref<Expr> dim) {
  auto CE = dyn_cast<BVConstExpr>(dim);
  if (!CE)
    ErrorReporter::reportImplementationLimitation(
        "Unsupported variable dimension");
  switch (CE->getValue().getZExtValue()) {
  case 0: return prefix + "_x";
  case 1: return prefix + "_y";
  case 2: return prefix + "_z";
  default:
    ErrorReporter::reportImplementationLimitation("Unsupported dimension");
  }
}

static ref<Expr> mkLocalId(bugle::Type t, ref<Expr> dim) {
  return SpecialVarRefExpr::create(t, mkDimName("local_id", dim));
}

static ref<Expr> mkGroupId(bugle::Type t, ref<Expr> dim) {
  return SpecialVarRefExpr::create(t, mkDimName("group_id", dim));
}

static ref<Expr> mkLocalSize(bugle::Type t, ref<Expr> dim) {
  return SpecialVarRefExpr::create(t, mkDimName("group_size", dim));
}

static ref<Expr> mkNumGroups(bugle::Type t, ref<Expr> dim) {
  return SpecialVarRefExpr::create(t, mkDimName("num_groups", dim));
}

static ref<Expr> mkGlobalOffset(bugle::Type t, ref<Expr> dim) {
  return SpecialVarRefExpr::create(t, mkDimName("global_offset", dim));
}

static ref<Expr> mkWorkDim(bugle::Type t) {
  return SpecialVarRefExpr::create(t, "work_dim");
}

ref<Expr> TranslateFunction::handleGetLocalId(bugle::BasicBlock *BBB,
                                              llvm::CallInst *CI,
                                              const ExprVec &Args) {
  Type t = TM->translateType(CI->getType());
  return mkLocalId(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetGroupId(bugle::BasicBlock *BBB,
                                              llvm::CallInst *CI,
                                              const ExprVec &Args) {
  Type t = TM->translateType(CI->getType());
  return mkGroupId(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetLocalSize(bugle::BasicBlock *BBB,
                                                llvm::CallInst *CI,
                                                const ExprVec &Args) {
  Type t = TM->translateType(CI->getType());
  return mkLocalSize(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetNumGroups(bugle::BasicBlock *BBB,
                                                llvm::CallInst *CI,
                                                const ExprVec &Args) {
  Type t = TM->translateType(CI->getType());
  return mkNumGroups(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetGlobalOffset(bugle::BasicBlock *BBB,
                                                   llvm::CallInst *CI,
                                                   const ExprVec &Args) {
  Type t = TM->translateType(CI->getType());
  return mkGlobalOffset(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetWorkDim(bugle::BasicBlock *BBB,
                                              llvm::CallInst *CI,
                                              const ExprVec &Args) {
  Type t = TM->translateType(CI->getType());
  return mkWorkDim(t);
}

ref<Expr> TranslateFunction::handleGetImageWidth(bugle::BasicBlock *BBB,
                                                 llvm::CallInst *CI,
                                                 const ExprVec &Args) {
  return GetImageWidthExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleGetImageHeight(bugle::BasicBlock *BBB,
                                                  llvm::CallInst *CI,
                                                  const ExprVec &Args) {
  return GetImageHeightExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleSamplerInitializer(bugle::BasicBlock *BBB,
                                                      llvm::CallInst *CI,
                                                      const ExprVec &Args) {
  GlobalArray *GA = TM->getGlobalArray(CI);

  if (auto *C = dyn_cast<Constant>(CI->getArgOperand(0))) {
    TM->translateGlobalInit(GA, 0, C);
  } else {
    ErrorReporter::reportImplementationLimitation(
        "Non-constant samplers not supported");
  }

  return PointerExpr::create(
      GlobalArrayRefExpr::create(GA),
      BVConstExpr::createZero(TM->TD.getPointerSizeInBits()));
}

ref<Expr> TranslateFunction::handleAsyncWorkGroupCopy(bugle::BasicBlock *BBB,
                                                      llvm::CallInst *CI,
                                                      const ExprVec &Args) {
  ref<Expr> Src = Args[1], Dst = Args[0],
            SrcArr = ArrayIdExpr::create(Src, TM->defaultRange()),
            DstArr = ArrayIdExpr::create(Dst, TM->defaultRange()),
            SrcOfs = ArrayOffsetExpr::create(Src),
            DstOfs = ArrayOffsetExpr::create(Dst);
  Type SrcRangeTy = SrcArr->getType().range(),
       DstRangeTy = DstArr->getType().range();

  if (DstRangeTy == Type(Type::Any)) {
    BBB->addStmt(AssertStmt::createBadAccess(currentSourceLocs));
    // The result is irrelevant, but the caller requires one.
    Type HandleTy = TM->translateType(CI->getType());
    return BVConstExpr::createZero(HandleTy.width);
  }

  if (SrcRangeTy == Type(Type::Any)) {
    BBB->addStmt(AssertStmt::createBadAccess(currentSourceLocs));
    // The result is irrelevant, but the caller requires one.
    Type HandleTy = TM->translateType(CI->getType());
    return BVConstExpr::createZero(HandleTy.width);
  }

  assert(SrcRangeTy.width % 8 == 0);
  assert(DstRangeTy.width % 8 == 0);

  Type SrcArgRangeTy =
      TM->translateType(CI->getOperand(1)->getType()->getPointerElementType());
#ifndef NDEBUG
  Type DstArgRangeTy =
      TM->translateType(CI->getOperand(0)->getType()->getPointerElementType());
  assert(DstArgRangeTy == SrcArgRangeTy);
#endif

  ref<Expr> result;
  ref<Expr> SrcDiv = Expr::createExactBVSDiv(SrcOfs, SrcRangeTy.width / 8);
  ref<Expr> DstDiv = Expr::createExactBVSDiv(DstOfs, DstRangeTy.width / 8);
  if (SrcRangeTy == DstRangeTy && SrcRangeTy != Type(Type::Unknown) &&
      SrcRangeTy.width <= SrcArgRangeTy.width && !SrcDiv.isNull() &&
      !DstDiv.isNull()) {
    // Compensate for the modelled width being smaller than the width expected
    // by the function. In case the modelled width is larger, we model both
    // the source and destinations as byte-arrays.
    ref<Expr> NumElements = Args[2];
    if (SrcRangeTy.width < SrcArgRangeTy.width) {
      ref<Expr> NumElementsFactor = BVConstExpr::create(
          NumElements->getType().width, SrcArgRangeTy.width / SrcRangeTy.width);
      NumElements = BVMulExpr::create(NumElements, NumElementsFactor);
    }
    result = AsyncWorkGroupCopyExpr::create(DstArr, DstDiv, SrcArr, SrcDiv,
                                            NumElements, Args[3]);
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (SrcArr->computeArrayCandidates(Globals) &&
        DstArr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
    // The result is irrelevant, but the caller requires one.
    Type HandleTy = TM->translateType(CI->getType());
    result = BVConstExpr::createZero(HandleTy.width);
  }

  return result;
}

ref<Expr> TranslateFunction::handleWaitGroupEvents(bugle::BasicBlock *BBB,
                                                   llvm::CallInst *CI,
                                                   const ExprVec &Args) {
  auto NumEvents = dyn_cast<BVConstExpr>(Args[0]);
  if (!NumEvents) {
    // Could emit loop
    ErrorReporter::reportImplementationLimitation(
        "wait_group_events with a variable-sized set of events not supported");
  }

  Type EventsArgRangeTy =
      TM->translateType(CI->getOperand(1)->getType()->getPointerElementType());

  ref<Expr> Events = Args[1],
            EventsPtrArr = ArrayIdExpr::create(Events, EventsArgRangeTy),
            EventsPtrOfs = ArrayOffsetExpr::create(Events);
  Type EventsRangeTy = EventsPtrArr->getType().range();

  if (EventsRangeTy != EventsArgRangeTy) {
    // Could recombine by concatenating
    ErrorReporter::reportImplementationLimitation(
        "wait_group_events with cast set of events not supported");
  }

  for (unsigned i = 0; i < NumEvents->getValue().getZExtValue(); ++i) {
    auto Off = BVAddExpr::create(
        EventsPtrOfs, BVConstExpr::create(TM->BM->getPointerWidth(), i));
    auto LE =
        LoadExpr::create(EventsPtrArr, Off, EventsRangeTy, LoadsAreTemporal);
    BBB->addEvalStmt(LE, currentSourceLocs);
    BBB->addStmt(WaitGroupEventStmt::create(LE, currentSourceLocs));
  }

  return nullptr;
}

ref<Expr> TranslateFunction::handleCeil(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FCeilExpr::create(E); });
}

ref<Expr> TranslateFunction::handleCtpop(bugle::BasicBlock *BBB,
                                         llvm::CallInst *CI,
                                         const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return BVCtpopExpr::create(E); });
}

ref<Expr> TranslateFunction::handleCos(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FCosExpr::create(E); });
}

ref<Expr> TranslateFunction::handleCtlz(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  ref<Expr> isZeroUndef;

  if (TM->SL == TranslateModule::SL_CUDA)
    isZeroUndef = BoolConstExpr::create(false);
  else
    isZeroUndef = BVToBoolExpr::create(Args[1]);

  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return BVCtlzExpr::create(E, isZeroUndef);
  });
}

ref<Expr> TranslateFunction::handleExp(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FExpExpr::create(E); });
}

ref<Expr> TranslateFunction::handleExp2(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FExp2Expr::create(E); });
}

ref<Expr> TranslateFunction::handleFabs(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FAbsExpr::create(E); });
}

ref<Expr> TranslateFunction::handleFmax(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0], Args[1],
                                FMaxExpr::create);
}

ref<Expr> TranslateFunction::handleFmin(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0], Args[1],
                                FMinExpr::create);
}

ref<Expr> TranslateFunction::handleFloor(bugle::BasicBlock *BBB,
                                         llvm::CallInst *CI,
                                         const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FFloorExpr::create(E); });
}

ref<Expr> TranslateFunction::handleFma(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  ref<Expr> M =
      maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0], Args[1], FMulExpr::create);
  return maybeTranslateSIMDInst(BBB, Ty, Ty, M, Args[2], FAddExpr::create);
}

ref<Expr> TranslateFunction::handleFrexpExp(bugle::BasicBlock *BBB,
                                            llvm::CallInst *CI,
                                            const ExprVec &Args) {
  return FrexpExpExpr::create(cast<IntegerType>(CI->getType())->getBitWidth(),
                              Args[0]);
}

ref<Expr> TranslateFunction::handleFrexpFrac(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                             const ExprVec &Args) {
  return FrexpFracExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleLog(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FLogExpr::create(E); });
}

ref<Expr> TranslateFunction::handleLog10(bugle::BasicBlock *BBB,
                                         llvm::CallInst *CI,
                                         const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FLog10Expr::create(E); });
}

ref<Expr> TranslateFunction::handleLog2(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FLog2Expr::create(E); });
}


ref<Expr> TranslateFunction::handlePow(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0], Args[1],
      [&](ref<Expr> LHS, ref<Expr> RHS) { return FPowExpr::create(LHS, RHS); });
}

ref<Expr> TranslateFunction::handlePowi(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *Ty, ref<Expr> Op) {
    return FPowiExpr::create(Op, Args[1]);
  });
}

ref<Expr> TranslateFunction::handleRint(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FRintExpr::create(E); });
}

ref<Expr> TranslateFunction::handleSin(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FSinExpr::create(E); });
}

ref<Expr> TranslateFunction::handleRsqrt(bugle::BasicBlock *BBB,
                                         llvm::CallInst *CI,
                                         const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FRsqrtExpr::create(E); });
}

ref<Expr> TranslateFunction::handleSqrt(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FSqrtExpr::create(E); });
}

ref<Expr> TranslateFunction::handleTrunc(bugle::BasicBlock *BBB,
                                         llvm::CallInst *CI,
                                         const ExprVec &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(
      BBB, Ty, Ty, Args[0],
      [&](llvm::Type *T, ref<Expr> E) { return FTruncExpr::create(E); });
}

ref<Expr> TranslateFunction::handleSaddOvl(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  llvm::StructType *STy = cast<StructType>(CI->getType());
  llvm::Type *AddTy = STy->getElementType(0), *OvlTy = STy->getElementType(1);
  unsigned BitWidth = cast<IntegerType>(AddTy)->getBitWidth();

  ref<Expr> AddResult = BVAddExpr::create(Args[0], Args[1]),
            Arg0Sign = BVExtractExpr::create(Args[0], BitWidth - 1, 1),
            Arg1Sign = BVExtractExpr::create(Args[1], BitWidth - 1, 1),
            AddSign = BVExtractExpr::create(AddResult, BitWidth - 1, 1);

  // Overflow if the sign bits of the arguments are identical but differ from
  // the sign bit of the result.
  ref<Expr> OvlResult = BoolToBVExpr::create(AndExpr::create(
      EqExpr::create(Arg0Sign, Arg1Sign), NeExpr::create(Arg0Sign, AddSign)));

  if (TM->TD.getTypeAllocSize(AddTy) * 8 != BitWidth) {
    ref<Expr> Pad =
        BVConstExpr::createZero(TM->TD.getTypeAllocSize(AddTy) * 8 - BitWidth);
    AddResult = BVConcatExpr::create(Pad, AddResult);
  }

  assert(cast<IntegerType>(OvlTy)->getBitWidth() == 1);
  assert(TM->TD.getTypeAllocSize(OvlTy) > 0);
  ref<Expr> Pad =
      BVConstExpr::createZero(TM->TD.getTypeAllocSize(OvlTy) * 8 - 1);
  OvlResult = BVConcatExpr::create(Pad, OvlResult);

  return BVConcatExpr::create(OvlResult, AddResult);
}

ref<Expr> TranslateFunction::handleUaddOvl(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  llvm::StructType *STy = cast<StructType>(CI->getType());
  llvm::Type *AddTy = STy->getElementType(0), *OvlTy = STy->getElementType(1);
  unsigned BitWidth = cast<IntegerType>(AddTy)->getBitWidth();

  ref<Expr> AddExpr =
                BVAddExpr::create(BVZExtExpr::create(BitWidth + 1, Args[0]),
                                  BVZExtExpr::create(BitWidth + 1, Args[1])),
            AddResult = BVExtractExpr::create(AddExpr, 0, BitWidth);

  // Overflow if the most significant bit is non-zero.
  ref<Expr> OvlResult = BVExtractExpr::create(AddExpr, BitWidth, 1);

  if (TM->TD.getTypeAllocSize(AddTy) * 8 != BitWidth) {
    ref<Expr> Pad =
        BVConstExpr::createZero(TM->TD.getTypeAllocSize(AddTy) * 8 - BitWidth);
    AddResult = BVConcatExpr::create(Pad, AddResult);
  }

  assert(cast<IntegerType>(OvlTy)->getBitWidth() == 1);
  assert(TM->TD.getTypeAllocSize(OvlTy) > 0);
  ref<Expr> Pad =
      BVConstExpr::createZero(TM->TD.getTypeAllocSize(OvlTy) * 8 - 1);
  OvlResult = BVConcatExpr::create(Pad, OvlResult);

  return BVConcatExpr::create(OvlResult, AddResult);
}

ref<Expr> TranslateFunction::handleSsubOvl(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  llvm::StructType *STy = cast<StructType>(CI->getType());
  llvm::Type *SubTy = STy->getElementType(0), *OvlTy = STy->getElementType(1);
  unsigned BitWidth = cast<IntegerType>(SubTy)->getBitWidth();

  ref<Expr> SubResult = BVSubExpr::create(Args[0], Args[1]),
            Arg0Sign = BVExtractExpr::create(Args[0], BitWidth - 1, 1),
            Arg1Sign = BVExtractExpr::create(Args[1], BitWidth - 1, 1),
            AddSign = BVExtractExpr::create(SubResult, BitWidth - 1, 1);

  // Overflow if the sign bit of the first argument differs from both the sign
  // bit of the second argument and the sign bit of the result.
  ref<Expr> OvlResult = BoolToBVExpr::create(AndExpr::create(
      NeExpr::create(Arg0Sign, Arg1Sign), NeExpr::create(Arg0Sign, AddSign)));

  if (TM->TD.getTypeAllocSize(SubTy) * 8 != BitWidth) {
    ref<Expr> Pad =
        BVConstExpr::createZero(TM->TD.getTypeAllocSize(SubTy) * 8 - BitWidth);
    SubResult = BVConcatExpr::create(Pad, SubResult);
  }

  assert(cast<IntegerType>(OvlTy)->getBitWidth() == 1);
  assert(TM->TD.getTypeAllocSize(OvlTy) > 0);
  ref<Expr> Pad =
      BVConstExpr::createZero(TM->TD.getTypeAllocSize(OvlTy) * 8 - 1);
  OvlResult = BVConcatExpr::create(Pad, OvlResult);

  return BVConcatExpr::create(OvlResult, SubResult);
}

ref<Expr> TranslateFunction::handleUsubOvl(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const ExprVec &Args) {
  llvm::StructType *STy = cast<StructType>(CI->getType());
  llvm::Type *SubTy = STy->getElementType(0), *OvlTy = STy->getElementType(1);
  unsigned BitWidth = cast<IntegerType>(SubTy)->getBitWidth();

  ref<Expr> SubExpr =
                BVSubExpr::create(BVZExtExpr::create(BitWidth + 1, Args[0]),
                                  BVZExtExpr::create(BitWidth + 1, Args[1])),
            SubResult = BVExtractExpr::create(SubExpr, 0, BitWidth);

  // Overflow if the most significant bit is non-zero.
  ref<Expr> OvlResult = BVExtractExpr::create(SubExpr, BitWidth, 1);

  if (TM->TD.getTypeAllocSize(SubTy) * 8 != BitWidth) {
    ref<Expr> Pad =
        BVConstExpr::createZero(TM->TD.getTypeAllocSize(SubTy) * 8 - BitWidth);
    SubResult = BVConcatExpr::create(Pad, SubResult);
  }

  assert(cast<IntegerType>(OvlTy)->getBitWidth() == 1);
  assert(TM->TD.getTypeAllocSize(OvlTy) > 0);
  ref<Expr> Pad =
      BVConstExpr::createZero(TM->TD.getTypeAllocSize(OvlTy) * 8 - 1);
  OvlResult = BVConcatExpr::create(Pad, OvlResult);

  return BVConcatExpr::create(OvlResult, SubResult);
}

ref<Expr> TranslateFunction::handleAddNoovflUnsigned(bugle::BasicBlock *BBB,
                                                     llvm::CallInst *CI,
                                                     const ExprVec &Args) {
  ref<Expr> E = AddNoovflExpr::create(Args[0], Args[1], /*isSigned=*/false);
  BBB->addEvalStmt(E, currentSourceLocs);
  return E;
}

ref<Expr> TranslateFunction::handleAddNoovflSigned(bugle::BasicBlock *BBB,
                                                   llvm::CallInst *CI,
                                                   const ExprVec &Args) {
  ref<Expr> E = AddNoovflExpr::create(Args[0], Args[1], /*isSigned=*/true);
  BBB->addEvalStmt(E, currentSourceLocs);
  return E;
}

ref<Expr> TranslateFunction::handleAddNoovflPredicate(bugle::BasicBlock *BBB,
                                                      llvm::CallInst *CI,
                                                      const ExprVec &Args) {
  return AddNoovflPredicateExpr::create(Args);
}

ref<Expr> TranslateFunction::handleAdd(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  return BVAddExpr::create(Args[0], Args[1]);
}

ref<Expr> TranslateFunction::handleUninterpretedFunction(bugle::BasicBlock *BBB,
                                                         llvm::CallInst *CI,
                                                         const ExprVec &Args) {
  std::string name = CI->getCalledFunction()->getName();
  std::string strippedName = name.substr(strlen("__uninterpreted_function_"));
  Type RT = TM->translateType(CI->getCalledFunction()->getReturnType());
  return UninterpretedFunctionExpr::create(strippedName, RT, Args);
}

ref<Expr> TranslateFunction::handleIte(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const ExprVec &Args) {
  return IfThenElseExpr::create(BVToBoolExpr::create(Args[0]), Args[1],
                                Args[2]);
}

ref<Expr> TranslateFunction::handleAtomicHasTakenValue(bugle::BasicBlock *BBB,
                                                       llvm::CallInst *CI,
                                                       const ExprVec &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result = BoolToBVExpr::create(
      AtomicHasTakenValueExpr::create(arrayIdExpr, Args[1], Args[2]));

  if (!arrayIdExpr->getType().range().isKind(Type::BV)) {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (arrayIdExpr->computeArrayCandidates(Globals)) {
      std::transform(
          Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }
  return result;
}

ref<Expr> TranslateFunction::maybeTranslateSIMDInst(
    bugle::BasicBlock *BBB, llvm::Type *Ty, llvm::Type *OpTy, ref<Expr> Op,
    std::function<ref<Expr>(llvm::Type *, ref<Expr>)> F) {
  if (!isa<VectorType>(Ty))
    return F(Ty, Op);

  auto *VT = cast<VectorType>(Ty);
  unsigned NumElems = VT->getNumElements();
  assert(cast<VectorType>(OpTy)->getNumElements() == NumElems);
  unsigned ElemWidth = Op->getType().width / NumElems;
  ExprVec Elems;
  for (unsigned i = 0; i < NumElems; ++i) {
    ref<Expr> Opi = BVExtractExpr::create(Op, i * ElemWidth, ElemWidth);
    ref<Expr> Elem = F(VT->getElementType(), Opi);
    BBB->addEvalStmt(Elem, currentSourceLocs);
    Elems.push_back(Elem);
  }

  return Expr::createBVConcatN(Elems);
}

ref<Expr> TranslateFunction::maybeTranslateSIMDInst(
    bugle::BasicBlock *BBB, llvm::Type *Ty, llvm::Type *OpTy, ref<Expr> LHS,
    ref<Expr> RHS, std::function<ref<Expr>(ref<Expr>, ref<Expr>)> F) {
  if (!isa<VectorType>(Ty))
    return F(LHS, RHS);

  auto *VT = cast<VectorType>(Ty);
  unsigned NumElems = VT->getNumElements();
  assert(cast<VectorType>(OpTy)->getNumElements() == NumElems);
  unsigned ElemWidth = LHS->getType().width / NumElems;
  ExprVec Elems;
  for (unsigned i = 0; i < NumElems; ++i) {
    ref<Expr> LHSi = BVExtractExpr::create(LHS, i * ElemWidth, ElemWidth);
    ref<Expr> RHSi = BVExtractExpr::create(RHS, i * ElemWidth, ElemWidth);
    ref<Expr> Elem = F(LHSi, RHSi);
    BBB->addEvalStmt(Elem, currentSourceLocs);
    Elems.push_back(Elem);
  }

  return Expr::createBVConcatN(Elems);
}

void TranslateFunction::translateInstruction(bugle::BasicBlock *BBB,
                                             Instruction *I) {
  SourceLocsRef SLI = extractSourceLocs(I);
  if (SLI != nullptr)
    currentSourceLocs = SLI;

  ref<Expr> E;
  if (auto *BO = dyn_cast<BinaryOperator>(I)) {
    ref<Expr> LHS = translateValue(BO->getOperand(0), BBB),
              RHS = translateValue(BO->getOperand(1), BBB);
    ref<Expr>(*F)(ref<Expr>, ref<Expr>);
    switch (BO->getOpcode()) {
    case BinaryOperator::Add:  F = BVAddExpr::create;  break;
    case BinaryOperator::FAdd: F = FAddExpr::create;   break;
    case BinaryOperator::Sub:  F = BVSubExpr::create;  break;
    case BinaryOperator::FSub: F = FSubExpr::create;   break;
    case BinaryOperator::Mul:  F = BVMulExpr::create;  break;
    case BinaryOperator::FMul: F = FMulExpr::create;   break;
    case BinaryOperator::SDiv: F = BVSDivExpr::create; break;
    case BinaryOperator::UDiv: F = BVUDivExpr::create; break;
    case BinaryOperator::FDiv: F = FDivExpr::create;   break;
    case BinaryOperator::SRem: F = BVSRemExpr::create; break;
    case BinaryOperator::URem: F = BVURemExpr::create; break;
    case BinaryOperator::FRem: F = FRemExpr::create;   break;
    case BinaryOperator::Shl:  F = BVShlExpr::create;  break;
    case BinaryOperator::AShr: F = BVAShrExpr::create; break;
    case BinaryOperator::LShr: F = BVLShrExpr::create; break;
    case BinaryOperator::And:  F = BVAndExpr::create;  break;
    case BinaryOperator::Or:   F = BVOrExpr::create;   break;
    case BinaryOperator::Xor:  F = BVXorExpr::create;  break;
    default:
      ErrorReporter::reportImplementationLimitation(
          "Unsupported binary operator");
    }
    E = maybeTranslateSIMDInst(BBB, BO->getType(), BO->getType(), LHS, RHS, F);
  } else if (auto *GEPI = dyn_cast<GetElementPtrInst>(I)) {
    ref<Expr> Ptr = translateValue(GEPI->getPointerOperand(), BBB);
    E = TM->translateGEP(Ptr, klee::gep_type_begin(GEPI),
                         klee::gep_type_end(GEPI),
                         [&](Value *V) { return translateValue(V, BBB); });
  } else if (auto *EV = dyn_cast<ExtractValueInst>(I)) {
    ref<Expr> Agg = translateValue(EV->getAggregateOperand(), BBB);
    E = TM->translateEV(Agg, klee::ev_type_begin(EV), klee::ev_type_end(EV),
                        [&](Value *V) { return translateValue(V, BBB); });
  } else if (auto *IV = dyn_cast<InsertValueInst>(I)) {
    ref<Expr> Agg = translateValue(IV->getAggregateOperand(), BBB);
    ref<Expr> Val = translateValue(IV->getInsertedValueOperand(), BBB);
    E = TM->translateIV(Agg, Val, klee::iv_type_begin(IV),
                        klee::iv_type_end(IV),
                        [&](Value *V) { return translateValue(V, BBB); });
  } else if (auto *AI = dyn_cast<AllocaInst>(I)) {
    auto *AS = dyn_cast<Constant>(AI->getArraySize());
    if (AS == nullptr)
      ErrorReporter::reportImplementationLimitation(
          "Variable length arrays not supported");
    auto *NE = dyn_cast<BVConstExpr>(TM->translateConstant(AS));
    if (NE == nullptr || NE->getValue().getZExtValue() != 1)
      ErrorReporter::reportImplementationLimitation(
          "Only alloca with one element supported");
    GlobalArray *GA = TM->getGlobalArray(AI);
    E = PointerExpr::create(
        GlobalArrayRefExpr::create(GA),
        BVConstExpr::createZero(TM->TD.getPointerSizeInBits()));
  } else if (auto *LI = dyn_cast<LoadInst>(I)) {
    ref<Expr> Ptr = translateValue(LI->getPointerOperand(), BBB),
              PtrArr = ArrayIdExpr::create(Ptr, TM->defaultRange()),
              PtrOfs = ArrayOffsetExpr::create(Ptr);
    Type ArrRangeTy = PtrArr->getType().range();
    Type LoadTy = TM->translateType(LI->getType());
    Type LoadElTy = TM->translateArrayRangeType(LI->getType());
    bool VectorLoad = false;
    unsigned VectorElemsCount = 1;
    if (LoadTy != LoadElTy) {
      VectorLoad = true;
      VectorElemsCount = LoadTy.width / LoadElTy.width;
    }
    assert(LoadTy.width % 8 == 0);
    ref<Expr> Div;
    if ((ArrRangeTy == LoadElTy || ArrRangeTy == Type(Type::Any)) &&
        !(Div = Expr::createExactBVSDiv(PtrOfs, LoadElTy.width / 8)).isNull()) {
      if (VectorLoad) {
        ExprVec ElemsLoaded;
        for (unsigned i = 0; i != VectorElemsCount; ++i) {
          ref<Expr> ElemOfs = BVAddExpr::create(
              Div, BVConstExpr::create(Div->getType().width, i));
          ref<Expr> ValElem =
              LoadExpr::create(PtrArr, ElemOfs, LoadElTy, LoadsAreTemporal);
          BBB->addEvalStmt(ValElem, currentSourceLocs);
          if (LoadElTy.isKind(Type::Pointer))
            ValElem =
                SafePtrToBVExpr::create(ValElem->getType().width, ValElem);
          else if (LoadElTy.isKind(Type::FunctionPointer))
            ValElem =
                FuncPtrToBVExpr::create(ValElem->getType().width, ValElem);
          ElemsLoaded.push_back(ValElem);
        }
        E = Expr::createBVConcatN(ElemsLoaded);
      } else {
        E = LoadExpr::create(PtrArr, Div, LoadElTy, LoadsAreTemporal);
      }
    } else if (ArrRangeTy.kind == Type::BV &&
               LoadTy.width % ArrRangeTy.width == 0 &&
               !(Div = Expr::createExactBVSDiv(PtrOfs, ArrRangeTy.width / 8))
                    .isNull()) {
      ExprVec PartsLoaded;
      for (unsigned i = 0; i != LoadTy.width / ArrRangeTy.width; ++i) {
        ref<Expr> PartOfs = BVAddExpr::create(
            Div, BVConstExpr::create(Div->getType().width, i));
        ref<Expr> PartVal =
            LoadExpr::create(PtrArr, PartOfs, ArrRangeTy, LoadsAreTemporal);
        PartsLoaded.push_back(PartVal);
        BBB->addEvalStmt(PartVal, currentSourceLocs);
      }
      E = Expr::createBVConcatN(PartsLoaded);
      if (LoadTy.isKind(Type::Pointer))
        E = SafeBVToPtrExpr::create(E->getType().width, E);
      else if (LoadTy.isKind(Type::FunctionPointer))
        E = BVToFuncPtrExpr::create(E->getType().width, E);
    } else {
      TM->NeedAdditionalByteArrayModels = true;
      std::set<GlobalArray *> Globals;
      if (PtrArr->computeArrayCandidates(Globals)) {
        std::transform(
            Globals.begin(), Globals.end(),
            std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
            [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
      } else {
        TM->NextModelAllAsByteArray = true;
      }
      E = TM->translateArbitrary(LoadTy);
    }
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    ref<Expr> Ptr = translateValue(SI->getPointerOperand(), BBB),
              Val = translateValue(SI->getValueOperand(), BBB),
              PtrArr = ArrayIdExpr::create(Ptr, TM->defaultRange()),
              PtrOfs = ArrayOffsetExpr::create(Ptr);
    Type ArrRangeTy = PtrArr->getType().range();
    Type StoreTy = Val->getType();
    Type StoreElTy =
        TM->translateArrayRangeType(SI->getValueOperand()->getType());
    bool VectorStore = false;
    unsigned VectorElemsCount = 1;
    if (StoreTy != StoreElTy) {
      VectorStore = true;
      VectorElemsCount = StoreTy.width / StoreElTy.width;
    }
    assert(StoreTy.width % 8 == 0);
    ref<Expr> Div;
    // If ArrRangeTy is Any, then we are using a null pointer for storing
    if ((ArrRangeTy == StoreElTy || ArrRangeTy == Type(Type::Any)) &&
        !(Div = Expr::createExactBVSDiv(PtrOfs, StoreElTy.width / 8))
             .isNull()) {
      if (VectorStore) {
        for (unsigned i = 0; i != VectorElemsCount; ++i) {
          ref<Expr> ElemOfs = BVAddExpr::create(
              Div, BVConstExpr::create(Div->getType().width, i));
          ref<Expr> ValElem =
              BVExtractExpr::create(Val, i * StoreElTy.width, StoreElTy.width);
          if (StoreElTy.isKind(Type::Pointer))
            ValElem =
                SafeBVToPtrExpr::create(ValElem->getType().width, ValElem);
          else if (StoreElTy.isKind(Type::FunctionPointer))
            ValElem =
                BVToFuncPtrExpr::create(ValElem->getType().width, ValElem);
          BBB->addStmt(
              StoreStmt::create(PtrArr, ElemOfs, ValElem, currentSourceLocs));
        }
      } else {
        BBB->addStmt(StoreStmt::create(PtrArr, Div, Val, currentSourceLocs));
      }
    } else if (ArrRangeTy.kind == Type::BV &&
               StoreTy.width % ArrRangeTy.width == 0 &&
               !(Div = Expr::createExactBVSDiv(PtrOfs, ArrRangeTy.width / 8))
                    .isNull()) {
      if (StoreTy.isKind(Type::Pointer)) {
        Val = SafePtrToBVExpr::create(Val->getType().width, Val);
        BBB->addEvalStmt(Val, currentSourceLocs);
      } else if (StoreTy.isKind(Type::FunctionPointer)) {
        Val = FuncPtrToBVExpr::create(Val->getType().width, Val);
        BBB->addEvalStmt(Val, currentSourceLocs);
      }
      for (unsigned i = 0; i != Val->getType().width / ArrRangeTy.width; ++i) {
        ref<Expr> PartOfs = BVAddExpr::create(
            Div, BVConstExpr::create(Div->getType().width, i));
        ref<Expr> PartVal =
            BVExtractExpr::create(Val, i * ArrRangeTy.width,
                                  ArrRangeTy.width); // Assumes little endian
        BBB->addStmt(
            StoreStmt::create(PtrArr, PartOfs, PartVal, currentSourceLocs));
      }
    } else {
      TM->NeedAdditionalByteArrayModels = true;
      std::set<GlobalArray *> Globals;
      if (PtrArr->computeArrayCandidates(Globals)) {
        std::transform(
            Globals.begin(), Globals.end(),
            std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
            [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
      } else {
        TM->NextModelAllAsByteArray = true;
      }
    }
    return;
  } else if (auto *II = dyn_cast<ICmpInst>(I)) {
    ref<Expr> LHS = translateValue(II->getOperand(0), BBB),
              RHS = translateValue(II->getOperand(1), BBB);
    E = maybeTranslateSIMDInst(BBB, II->getType(), II->getOperand(0)->getType(),
                               LHS, RHS,
                               [&](ref<Expr> LHS, ref<Expr> RHS) -> ref<Expr> {
      ref<Expr> E = TM->translateICmp(II->getPredicate(), LHS, RHS);
      BBB->addEvalStmt(E, currentSourceLocs);
      return BoolToBVExpr::create(E);
    });
  } else if (auto *FI = dyn_cast<FCmpInst>(I)) {
    ref<Expr> LHS = translateValue(FI->getOperand(0), BBB),
              RHS = translateValue(FI->getOperand(1), BBB);
    E = maybeTranslateSIMDInst(BBB, FI->getType(), FI->getOperand(0)->getType(),
                               LHS, RHS,
                               [&](ref<Expr> LHS, ref<Expr> RHS) -> ref<Expr> {
      ref<Expr> E = BoolConstExpr::create(false);
      if (FI->getPredicate() & FCmpInst::FCMP_OEQ)
        E = OrExpr::create(E, FEqExpr::create(LHS, RHS));
      if (FI->getPredicate() & FCmpInst::FCMP_OGT)
        E = OrExpr::create(E, FLtExpr::create(RHS, LHS));
      if (FI->getPredicate() & FCmpInst::FCMP_OLT)
        E = OrExpr::create(E, FLtExpr::create(LHS, RHS));
      if (FI->getPredicate() & FCmpInst::FCMP_UNO)
        E = OrExpr::create(E, FUnoExpr::create(LHS, RHS));
      BBB->addEvalStmt(E, currentSourceLocs);
      return BoolToBVExpr::create(E);
    });
  } else if (auto *ZEI = dyn_cast<ZExtInst>(I)) {
    ref<Expr> Op = translateValue(ZEI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, ZEI->getType(),
                               ZEI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return BVZExtExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto *SEI = dyn_cast<SExtInst>(I)) {
    ref<Expr> Op = translateValue(SEI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, SEI->getType(),
                               SEI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return BVSExtExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto *FPSII = dyn_cast<FPToSIInst>(I)) {
    ref<Expr> Op = translateValue(FPSII->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, FPSII->getType(),
                               FPSII->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return FPToSIExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto *FPUII = dyn_cast<FPToUIInst>(I)) {
    ref<Expr> Op = translateValue(FPUII->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, FPUII->getType(),
                               FPUII->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return FPToUIExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto *SIFPI = dyn_cast<SIToFPInst>(I)) {
    ref<Expr> Op = translateValue(SIFPI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, SIFPI->getType(),
                               SIFPI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return SIToFPExpr::create(TM->TD.getTypeSizeInBits(Ty), Op);
    });
  } else if (auto *UIFPI = dyn_cast<UIToFPInst>(I)) {
    ref<Expr> Op = translateValue(UIFPI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, UIFPI->getType(),
                               UIFPI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return UIToFPExpr::create(TM->TD.getTypeSizeInBits(Ty), Op);
    });
  } else if (isa<FPExtInst>(I) || isa<FPTruncInst>(I)) {
    auto *CI = cast<CastInst>(I);
    ref<Expr> Op = translateValue(CI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, CI->getType(), CI->getOperand(0)->getType(),
                               Op, [&](llvm::Type *Ty, ref<Expr> Op) {
      return FPConvExpr::create(TM->TD.getTypeSizeInBits(Ty), Op);
    });
  } else if (auto *TI = dyn_cast<TruncInst>(I)) {
    ref<Expr> Op = translateValue(TI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, TI->getType(), TI->getOperand(0)->getType(),
                               Op, [&](llvm::Type *Ty, ref<Expr> Op) {
      return BVExtractExpr::create(Op, 0, cast<IntegerType>(Ty)->getBitWidth());
    });
  } else if (auto *I2PI = dyn_cast<IntToPtrInst>(I)) {
    ref<Expr> Op = translateValue(I2PI->getOperand(0), BBB);
    assert(I2PI->getType()->isPointerTy());
    if (I2PI->getType()->getPointerElementType()->isFunctionTy())
      E = BVToFuncPtrExpr::create(TM->TD.getPointerSizeInBits(), Op);
    else
      E = BVToPtrExpr::create(TM->TD.getPointerSizeInBits(), Op);
  } else if (auto *P2II = dyn_cast<PtrToIntInst>(I)) {
    ref<Expr> Op = translateValue(P2II->getOperand(0), BBB);
    Type OpTy = Op->getType();
    if (OpTy.isKind(Type::Pointer))
      E = PtrToBVExpr::create(TM->TD.getTypeSizeInBits(I->getType()), Op);
    else if (OpTy.isKind(Type::FunctionPointer))
      E = FuncPtrToBVExpr::create(TM->TD.getTypeSizeInBits(I->getType()), Op);
  } else if (auto *BCI = dyn_cast<BitCastInst>(I)) {
    ref<Expr> Op = translateValue(BCI->getOperand(0), BBB);
    E = TM->translateBitCast(BCI->getSrcTy(), BCI->getDestTy(), Op);
    if (Op.get() == E.get()) {
      ValueExprMap[I] = Op;
      return;
    }
  } else if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(I)) {
    ValueExprMap[I] = translateValue(ASCI->getOperand(0), BBB);
    return;
  } else if (auto *SI = dyn_cast<SelectInst>(I)) {
    ref<Expr> Cond = translateValue(SI->getCondition(), BBB),
              TrueVal = translateValue(SI->getTrueValue(), BBB),
              FalseVal = translateValue(SI->getFalseValue(), BBB);
    if (auto *VT = dyn_cast<VectorType>(SI->getCondition()->getType())) {
      unsigned elementBitWidth =
          TrueVal->getType().width / VT->getNumElements();
      E = nullptr;
      for (unsigned i = 0; i < VT->getNumElements(); ++i) {
        ref<Expr> Ite = IfThenElseExpr::create(
            BVToBoolExpr::create(BVExtractExpr::create(Cond, i, 1)),
            BVExtractExpr::create(TrueVal, i * elementBitWidth,
                                  elementBitWidth),
            BVExtractExpr::create(FalseVal, i * elementBitWidth,
                                  elementBitWidth));
        E = (i == 0) ? Ite : BVConcatExpr::create(Ite, E);
      }
    } else {
      Cond = BVToBoolExpr::create(Cond);
      E = IfThenElseExpr::create(Cond, TrueVal, FalseVal);
    }
  } else if (auto *EEI = dyn_cast<ExtractElementInst>(I)) {
    ref<Expr> Vec = translateValue(EEI->getVectorOperand(), BBB),
              Idx = translateValue(EEI->getIndexOperand(), BBB);
    unsigned EltBits = TM->TD.getTypeSizeInBits(EEI->getType());
    unsigned ElemCount =
        EEI->getVectorOperand()->getType()->getVectorNumElements();
    E = BVExtractExpr::create(Vec, EltBits * 0, EltBits);
    for (unsigned i = 1; i < ElemCount; ++i) {
      ref<Expr> Cmp =
          EqExpr::create(BVConstExpr::create(Idx->getType().width, i), Idx);
      ref<Expr> Extract = BVExtractExpr::create(Vec, EltBits * i, EltBits);
      E = IfThenElseExpr::create(Cmp, Extract, E);
    }
  } else if (auto *IEI = dyn_cast<InsertElementInst>(I)) {
    ref<Expr> Vec = translateValue(IEI->getOperand(0), BBB),
              NewElt = translateValue(IEI->getOperand(1), BBB),
              Idx = translateValue(IEI->getOperand(2), BBB);
    llvm::Type *EltType = IEI->getType()->getElementType();
    unsigned EltBits = TM->TD.getTypeSizeInBits(EltType);
    unsigned ElemCount = IEI->getType()->getNumElements();
    BVConstExpr *CEIdx = cast<BVConstExpr>(Idx);
    unsigned UIdx = CEIdx->getValue().getZExtValue();
    ExprVec Elems;
    for (unsigned i = 0; i != ElemCount; ++i) {
      Elems.push_back(i == UIdx ? NewElt : BVExtractExpr::create(
                                               Vec, EltBits * i, EltBits));
    }
    E = Expr::createBVConcatN(Elems);
  } else if (auto *SVI = dyn_cast<ShuffleVectorInst>(I)) {
    ref<Expr> Vec1 = translateValue(SVI->getOperand(0), BBB),
              Vec2 = translateValue(SVI->getOperand(1), BBB);
    unsigned EltBits =
        TM->TD.getTypeSizeInBits(SVI->getType()->getElementType());
    unsigned VecElemCount =
        cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements();
    unsigned ResElemCount = SVI->getType()->getNumElements();
    ExprVec Elems;
    for (unsigned i = 0; i != ResElemCount; ++i) {
      ref<Expr> L;
      int MaskValI = SVI->getMaskValue(i);
      if (MaskValI < 0)
        L = BVConstExpr::create(EltBits, 0);
      else {
        unsigned MaskVal = (unsigned)MaskValI;
        if (MaskVal < VecElemCount)
          L = BVExtractExpr::create(Vec1, EltBits * MaskVal, EltBits);
        else
          L = BVExtractExpr::create(Vec2, EltBits * (MaskVal - VecElemCount),
                                    EltBits);
      }
      Elems.push_back(L);
    }
    E = Expr::createBVConcatN(Elems);
  } else if (auto *CI = dyn_cast<CallInst>(I)) {
    CallSite CS(CI);
    ExprVec Args;
    std::transform(CS.arg_begin(), CS.arg_end(), std::back_inserter(Args),
                   [&](Value *V) { return translateValue(V, BBB); });

    if (auto *II = dyn_cast<IntrinsicInst>(CI)) {
      auto ID = II->getIntrinsicID();
      auto SFII = SpecialFunctionMap.Intrinsics.find(ID);
      if (SFII != SpecialFunctionMap.Intrinsics.end()) {
        E = (this->*SFII->second)(BBB, CI, Args);
        assert(E.isNull() == CI->getType()->isVoidTy());
        if (E.isNull())
          return;
      } else {
        std::string name = Intrinsic::getName(ID, {});
        std::string msg = "Intrinsic '" + name + "' not supported";
        ErrorReporter::reportImplementationLimitation(msg);
      }
    } else {
      auto *F = CI->getCalledFunction();
      auto SFI = SpecialFunctionMap.Functions.end();
      if (F) {
        SFI = SpecialFunctionMap.Functions.find(
            trimForRequiresFreshArrayFunction(F->getName()));
      }
      if (SFI != SpecialFunctionMap.Functions.end()) {
        E = (this->*SFI->second)(BBB, CI, Args);
        assert(E.isNull() == CI->getType()->isVoidTy());
        if (E.isNull())
          return;
      } else {
        if (CI->getType()->isVoidTy()) {
          auto *V = CI->getCalledValue();
          BBB->addStmt(TM->modelCallStmt(V->getType(), CI->getCalledFunction(),
                                         translateValue(V, BBB), Args,
                                         currentSourceLocs));
          return;
        } else {
          auto *V = CI->getCalledValue();
          E = TM->modelCallExpr(V->getType(), CI->getCalledFunction(),
                                translateValue(V, BBB), Args);
          BBB->addEvalStmt(E, currentSourceLocs);
          ValueExprMap[I] = TM->unmodelValue(F, E);
          return;
        }
      }
    }
  } else if (auto *RI = dyn_cast<ReturnInst>(I)) {
    if (auto *V = RI->getReturnValue()) {
      assert(ReturnVar && "Returning value without return variable?");
      ref<Expr> Val = TM->modelValue(F, translateValue(V, BBB));
      BBB->addStmt(VarAssignStmt::create(ReturnVar, Val));
      ReturnVals.push_back(Val);
    }
    BBB->addStmt(ReturnStmt::create());
    return;
  } else if (auto *BI = dyn_cast<BranchInst>(I)) {
    if (BI->isConditional()) {
      ref<Expr> Cond =
          BVToBoolExpr::create(translateValue(BI->getCondition(), BBB));
      BBB->addEvalStmt(Cond, currentSourceLocs);

      bugle::BasicBlock *TrueBB = BF->addBasicBlock("truebb");
      TrueBB->addStmt(AssumeStmt::createPartition(Cond));
      TrueBB->addStmt(AssertStmt::createBlockSourceLoc(currentSourceLocs));
      addPhiAssigns(TrueBB, I->getParent(), BI->getSuccessor(0));
      TrueBB->addStmt(GotoStmt::create(BasicBlockMap[BI->getSuccessor(0)]));

      bugle::BasicBlock *FalseBB = BF->addBasicBlock("falsebb");
      FalseBB->addStmt(AssumeStmt::createPartition(NotExpr::create(Cond)));
      FalseBB->addStmt(AssertStmt::createBlockSourceLoc(currentSourceLocs));
      addPhiAssigns(FalseBB, I->getParent(), BI->getSuccessor(1));
      FalseBB->addStmt(GotoStmt::create(BasicBlockMap[BI->getSuccessor(1)]));

      std::vector<bugle::BasicBlock *> BBs;
      BBs.push_back(TrueBB);
      BBs.push_back(FalseBB);
      BBB->addStmt(GotoStmt::create(BBs));
    } else {
      addPhiAssigns(BBB, I->getParent(), BI->getSuccessor(0));
      BBB->addStmt(GotoStmt::create(BasicBlockMap[BI->getSuccessor(0)]));
    }
    return;
  } else if (auto *SI = dyn_cast<SwitchInst>(I)) {
    ref<Expr> Cond = translateValue(SI->getCondition(), BBB);
    ref<Expr> DefaultExpr = BoolConstExpr::create(true);
    std::vector<bugle::BasicBlock *> Succs;

    for (auto &Case : SI->cases()) {
      ref<Expr> Val = TM->translateConstant(Case.getCaseValue());
      bugle::BasicBlock *BB = BF->addBasicBlock("casebb");
      Succs.push_back(BB);
      BB->addStmt(AssumeStmt::createPartition(EqExpr::create(Cond, Val)));
      BB->addStmt(AssertStmt::createBlockSourceLoc(currentSourceLocs));
      addPhiAssigns(BB, SI->getParent(), Case.getCaseSuccessor());
      BB->addStmt(GotoStmt::create(BasicBlockMap[Case.getCaseSuccessor()]));
      DefaultExpr = AndExpr::create(DefaultExpr, NeExpr::create(Cond, Val));
    }

    bugle::BasicBlock *DefaultBB = BF->addBasicBlock("defaultbb");
    Succs.push_back(DefaultBB);
    DefaultBB->addStmt(AssertStmt::createBlockSourceLoc(currentSourceLocs));
    DefaultBB->addStmt(AssumeStmt::createPartition(DefaultExpr));
    addPhiAssigns(DefaultBB, SI->getParent(),
                  SI->case_default()->getCaseSuccessor());
    DefaultBB->addStmt(GotoStmt::create(
        BasicBlockMap[SI->case_default()->getCaseSuccessor()]));

    BBB->addStmt(GotoStmt::create(Succs));
    return;
  } else if (auto *PN = dyn_cast<PHINode>(I)) {
    ValueExprMap[I] =
        TM->unmodelValue(PN, VarRefExpr::create(getPhiVariable(PN)));
    return;
  } else if (isa<UnreachableInst>(I)) {
    BBB->addStmt(
        AssertStmt::create(BoolConstExpr::create(false), /*global=*/false,
                           /*candidate=*/false, currentSourceLocs));
    return;
  } else {
    std::string name = I->getOpcodeName();
    std::string msg = "Instruction '" + name + "' not supported";
    ErrorReporter::reportImplementationLimitation(msg);
  }
  ValueExprMap[I] = E;
  if (LoadsAreTemporal)
    BBB->addEvalStmt(E, currentSourceLocs);
  return;
}

void TranslateFunction::translateBasicBlock(bugle::BasicBlock *BBB,
                                            llvm::BasicBlock *BB) {
  for (auto &I : *BB)
    translateInstruction(BBB, &I);
}
