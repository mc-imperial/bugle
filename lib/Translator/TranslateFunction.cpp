#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/BasicBlock.h"
#include "bugle/Expr.h"
#include "bugle/GlobalArray.h"
#include "bugle/Module.h"
#include "bugle/util/ErrorReporter.h"
#include "bugle/util/Functional.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "klee/util/GetElementPtrTypeIterator.h"

using namespace bugle;
using namespace llvm;

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

void TranslateFunction::addUninterpretedFunction(TranslateModule::SourceLanguage SL,
                                                 const std::string &fnName) {
  SpecialFnMapTy &SpecialFunctionMap = initSpecialFunctionMap(SL);
  SpecialFunctionMap.Functions[fnName] = &TranslateFunction::handleUninterpretedFunction;
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

bool TranslateFunction::isPreOrPostCondition(llvm::StringRef fnName) {
  if (fnName == "bugle_requires") return true;
  if (fnName == "__requires") return true;
  if (fnName == "__global_requires") return true;
  if (fnName == "bugle_ensures") return true;
  if (fnName == "__ensures") return true;
  if (fnName == "__global_ensures") return true;
  return false;
}

bool TranslateFunction::isBarrierFunction(TranslateModule::SourceLanguage SL,
                                          StringRef fnName)
{
  return (SL == TranslateModule::SL_OpenCL || SL == TranslateModule::SL_CUDA) &&
         fnName == "bugle_barrier";
}

bool TranslateFunction::isNormalFunction(TranslateModule::SourceLanguage SL,
                                         llvm::Function *F) {
  if (F->isIntrinsic()) return false;
  if (isAxiomFunction(F->getName())) return false;
  if (isUninterpretedFunction(F->getName())) return false;
  if (isSpecialFunction(SL, F->getName())) return false;
  if (isSpecificationFunction(F->getName())) return false;
  if (isBarrierFunction(SL, F->getName())) return false;
  return true;
}

bool TranslateFunction::isStandardEntryPoint(TranslateModule::SourceLanguage SL,
                                             StringRef fnName) {
  return SL == TranslateModule::SL_C && fnName == "main";
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
    fns["bugle_ensures"] = &TranslateFunction::handleEnsures;
    fns["__ensures"] = &TranslateFunction::handleEnsures;
    fns["__global_ensures"] = &TranslateFunction::handleGlobalEnsures;
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
    const std::string tys[] = { "char", "short", "int", "long" };
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
    fns["__return_val_int"] = &TranslateFunction::handleReturnVal;
    fns["__return_val_int4"] = &TranslateFunction::handleReturnVal;
    fns["__return_val_bool"] = &TranslateFunction::handleReturnVal;
    fns["__return_val_ptr"] = &TranslateFunction::handleReturnVal;
    fns["__old_int"] = &TranslateFunction::handleOld;
    fns["__old_bool"] = &TranslateFunction::handleOld;
    fns["__other_int"] = &TranslateFunction::handleOtherInt;
    fns["__other_bool"] = &TranslateFunction::handleOtherBool;
    fns["__other_ptr_base"] = &TranslateFunction::handleOtherPtrBase;
    fns["__implies"] = &TranslateFunction::handleImplies;
    fns["__enabled"] = &TranslateFunction::handleEnabled;
    fns["__read_local"] = &TranslateFunction::handleReadHasOccurred;
    fns["__read_global"] = &TranslateFunction::handleReadHasOccurred;
    fns["__read"] = &TranslateFunction::handleReadHasOccurred;
    fns["__write_local"] = &TranslateFunction::handleWriteHasOccurred;
    fns["__write_global"] = &TranslateFunction::handleWriteHasOccurred;
    fns["__write"] = &TranslateFunction::handleWriteHasOccurred;
    fns["__read_offset_local"] = &TranslateFunction::handleReadOffset;
    fns["__read_offset_global"] = &TranslateFunction::handleReadOffset;
    fns["__read_offset"] = &TranslateFunction::handleReadOffset;
    fns["__write_offset_local"] = &TranslateFunction::handleWriteOffset;
    fns["__write_offset_global"] = &TranslateFunction::handleWriteOffset;
    fns["__write_offset"] = &TranslateFunction::handleWriteOffset;
    fns["__not_accessed_local"] = &TranslateFunction::handleNotAccessed;
    fns["__not_accessed_global"] = &TranslateFunction::handleNotAccessed;
    fns["__not_accessed"] = &TranslateFunction::handleNotAccessed;
    fns["__ptr_base_local"] = &TranslateFunction::handlePtrBase;
    fns["__ptr_base_global"] = &TranslateFunction::handlePtrBase;
    fns["__ptr_base"] = &TranslateFunction::handlePtrBase;
    fns["__ptr_offset_local"] = &TranslateFunction::handlePtrOffset;
    fns["__ptr_offset_global"] = &TranslateFunction::handlePtrOffset;
    fns["__ptr_offset"] = &TranslateFunction::handlePtrOffset;
    fns["__array_snapshot_local"] = &TranslateFunction::handleArraySnapshot;
    fns["__array_snapshot_global"] = &TranslateFunction::handleArraySnapshot;
    fns["__array_snapshot"] = &TranslateFunction::handleArraySnapshot;

    if (SL == TranslateModule::SL_OpenCL || SL == TranslateModule::SL_CUDA)
    {
      const std::string opencl[] = { "__atomic_add",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_sub",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_xchg",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_float", "_global_float",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_min",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_max",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_and",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_or",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_xor",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_cmpxchg",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_inc",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
                                     "__atomic_dec",
                                       "_local_int", "_local_unsigned_int",
                                       "_global_int", "_global_unsigned_int",
                                       "_local_long", "_local_unsigned_long",
                                       "_global_long", "_global_unsigned_long",
                                       "" ,
         ""
      };

      const std::string cuda[] = { "__atomicAdd",
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
         ""
      };

      const std::string* atomics =
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

    if (SL == TranslateModule::SL_OpenCL ||
        SL == TranslateModule::SL_CUDA) {
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
      fns["get_image_width"] = &TranslateFunction::handleGetImageWidth;
      fns["get_image_height"] = &TranslateFunction::handleGetImageHeight;
    }

    if (SL == TranslateModule::SL_CUDA) {
      fns["cos"] = &TranslateFunction::handleCos;
      fns["sin"] = &TranslateFunction::handleSin;
      fns["sqrt"] = &TranslateFunction::handleSqrt;
      fns["sqrtf"] = &TranslateFunction::handleSqrt;
      fns["rsqrt"] = &TranslateFunction::handleRsqrt;
      fns["log"] = &TranslateFunction::handleLog;
      fns["exp"] = &TranslateFunction::handleExp;
    }

    auto &ints = SpecialFunctionMap.Intrinsics;
    ints[Intrinsic::cos] = &TranslateFunction::handleCos;
    ints[Intrinsic::exp2] = &TranslateFunction::handleExp;
    ints[Intrinsic::fabs] = &TranslateFunction::handleFabs;
    ints[Intrinsic::fma] = &TranslateFunction::handleFma;
    ints[Intrinsic::fmuladd] = &TranslateFunction::handleFma;
    ints[Intrinsic::floor] = &TranslateFunction::handleFloor;
    ints[Intrinsic::log2] = &TranslateFunction::handleLog;
    ints[Intrinsic::pow] = &TranslateFunction::handlePow;
    ints[Intrinsic::sin] = &TranslateFunction::handleSin;
    ints[Intrinsic::sqrt] = &TranslateFunction::handleSqrt;
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

void TranslateFunction::translate() {
  initSpecialFunctionMap(TM->SL);

  if (isGPUEntryPoint || isStandardEntryPoint(TM->SL, F->getName()))
    BF->setEntryPoint(true);

  if (isGPUEntryPoint)
    BF->addAttribute("kernel");

  if (isBarrierFunction(TM->SL, F->getName()))
    BF->addAttribute("barrier");

  if (isSpecificationFunction(F->getName()))
    BF->setSpecification(true);

  unsigned PtrSize = TM->TD.getPointerSizeInBits();
  for (auto i = F->arg_begin(), e = F->arg_end(); i != e; ++i) {
    if (isGPUEntryPoint && i->getType()->isPointerTy()) {
      GlobalArray *GA = TM->getGlobalArray(&*i);
      if (TM->SL == TranslateModule::SL_CUDA)
        GA->addAttribute("global");
      ValueExprMap[&*i] = PointerExpr::create(GlobalArrayRefExpr::create(GA),
                        BVConstExpr::createZero(PtrSize));
    } else {
      Var *V = BF->addArgument(TM->getModelledType(&*i), i->getName());
      ValueExprMap[&*i] = TM->unmodelValue(&*i, VarRefExpr::create(V));
    }
  }

  if (BF->return_begin() != BF->return_end())
    ReturnVar = *BF->return_begin();

  std::set<llvm::BasicBlock *> BBSet;
  std::vector<llvm::BasicBlock *> BBList;

  for (auto i = F->begin(), e = F->end(); i != e; ++i) {
    AddBasicBlockInOrder(BBSet, BBList, &*i);
    BasicBlockMap[&*i] = BF->addBasicBlock(i->getName());
  }

  for (auto i = BBList.begin(), e = BBList.end(); i != e; ++i)
    translateBasicBlock(BasicBlockMap[*i], *i);

  // If we're modelling everything as a byte array, don't bother to compute
  // value models.
  if (TM->ModelAllAsByteArray)
    return;

  // For each phi we encountered in the function, see if we can model it.
  for (auto i = PhiAssignsMap.begin(), e = PhiAssignsMap.end(); i != e; ++i) {
    TM->computeValueModel(i->first, PhiVarMap[i->first], i->second);
  }

  // See if we can model the return value. This requires the function to have
  // a body.
  if (BBList.begin() != BBList.end())
    TM->computeValueModel(F, 0, ReturnVals);
}

ref<Expr> TranslateFunction::translateValue(llvm::Value *V,  bugle::BasicBlock *BBB) {
  if (isa<Instruction>(V) || isa<Argument>(V)) {
    auto MI = ValueExprMap.find(V);
    assert(MI != ValueExprMap.end());
    return MI->second;
  }

  if (isa<UndefValue>(V)) {
    ref<Expr> E = HavocExpr::create(TM->translateType(V->getType()));
    BBB->addStmt(new EvalStmt(E));
    return E;
  }

  if (auto C = dyn_cast<Constant>(V))
    return TM->translateConstant(C);

  if (isa<MDNode>(V)) {
    // ignore metadata values
    return 0;
  }
  ErrorReporter::reportImplementationLimitation("Unsupported value");
}

Var *TranslateFunction::getPhiVariable(llvm::PHINode *PN) {
  auto &i = PhiVarMap[PN];
  if (i)
    return i;

  i = BF->addLocal(TM->getModelledType(PN), PN->getName());
  return i;
}

void TranslateFunction::addPhiAssigns(bugle::BasicBlock *BBB,
                                      llvm::BasicBlock *Pred,
                                      llvm::BasicBlock *Succ) {
  std::vector<Var *> Vars;
  std::vector<ref<Expr>> Exprs;
  for (auto i = Succ->begin(), e = Succ->end(); i != e && isa<PHINode>(i); ++i){
    PHINode *PN = cast<PHINode>(i);
    int idx = PN->getBasicBlockIndex(Pred);
    assert(idx != -1 && "No phi index?");

    Vars.push_back(getPhiVariable(PN));
    auto Val = TM->modelValue(PN, translateValue(PN->getIncomingValue(idx), BBB));
    Exprs.push_back(Val);
    PhiAssignsMap[PN].push_back(Val);
  }

  if (!Vars.empty())
    BBB->addStmt(new VarAssignStmt(Vars, Exprs));
}

SourceLocsRef TranslateFunction::extractSourceLocs(llvm::Instruction *I) {
  SourceLocs *sourcelocs = 0;
  if (llvm::MDNode *mdnode = I->getMetadata("dbg")) {
    sourcelocs = new SourceLocs();
    llvm::DILocation Loc(mdnode);
    do {
      sourcelocs->push_back(SourceLoc(Loc.getLineNumber(), Loc.getColumnNumber(),
                                      Loc.getFilename().str(),
                                      Loc.getDirectory().str()));
      Loc = Loc.getOrigLocation();
    } while (Loc != llvm::DILocation(0));
  }
  return SourceLocsRef(sourcelocs);
}


ref<Expr> TranslateFunction::handleNoop(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const std::vector<ref<Expr>> &Args) {
  return 0;
}

void TranslateFunction::addAssertStmt(bugle::BasicBlock *BBB,
                                      const ref<Expr> &Arg, bool isGlobal,
                                      bool isCandidate, bool isInvariant) {
  Stmt *assertstmt = new AssertStmt(Expr::createNeZero(Arg), isGlobal,
                                    isCandidate, isInvariant);
  assertstmt->setSourceLocs(currentSourceLocs);
  BBB->addStmt(assertstmt);
}

ref<Expr> TranslateFunction::handleAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], false, false, false);
  return 0;
}

ref<Expr> TranslateFunction::handleGlobalAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], true, false, false);
  return 0;
}

ref<Expr> TranslateFunction::handleCandidateAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], false, true, false);
  return 0;
}

ref<Expr> TranslateFunction::handleCandidateGlobalAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], true, true, false);
  return 0;
}

ref<Expr> TranslateFunction::handleInvariant(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], false, false, true);
  return 0;
}

ref<Expr> TranslateFunction::handleGlobalInvariant(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], true, false, true);
  return 0;
}

ref<Expr> TranslateFunction::handleCandidateInvariant(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], false, true, true);
  return 0;
}

ref<Expr> TranslateFunction::handleCandidateGlobalInvariant(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  addAssertStmt(BBB, Args[0], true, true, true);
  return 0;
}

ref<Expr> TranslateFunction::handleNonTemporalLoadsBegin(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  assert(LoadsAreTemporal && "Nested __non_temporal_loads_begin");
  LoadsAreTemporal = false;
  return 0;
}

ref<Expr> TranslateFunction::handleNonTemporalLoadsEnd(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  assert(!LoadsAreTemporal && "__non_temporal_loads_end without __non_temporal_loads_begin");
  LoadsAreTemporal = true;
  return 0;
}

ref<Expr> TranslateFunction::handleAssertFail(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  Stmt *assertStmt = new AssertStmt(BoolConstExpr::create(false), false, false, false);
  assertStmt->setSourceLocs(currentSourceLocs);
  BBB->addStmt(assertStmt);
  return 0;
}

ref<Expr> TranslateFunction::handleAssume(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  BBB->addStmt(new AssumeStmt(Expr::createNeZero(Args[0])));
  return 0;
}

ref<Expr> TranslateFunction::handleRequires(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addRequires(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleGlobalRequires(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addGlobalRequires(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleEnsures(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addEnsures(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleGlobalEnsures(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addGlobalEnsures(Expr::createNeZero(Args[0]), extractSourceLocs(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleReadsFrom(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addModifies(AccessHasOccurredExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), false),
        extractSourceLocs(CI));
  BF->addModifies(AccessOffsetExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), false),
        extractSourceLocs(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleWritesTo(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addModifies(AccessHasOccurredExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), true),
        extractSourceLocs(CI));
  BF->addModifies(AccessOffsetExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), true),
        extractSourceLocs(CI));
  BF->addModifies(UnderlyingArrayExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange())),
        extractSourceLocs(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleOld(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return OldExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleReturnVal(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return TM->unmodelValue(F, VarRefExpr::create(ReturnVar));
}

ref<Expr> TranslateFunction::handleOtherInt(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return OtherIntExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleOtherBool(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return BoolToBVExpr::create(OtherBoolExpr::create(BVToBoolExpr::create(Args[0])));
}

ref<Expr> TranslateFunction::handleOtherPtrBase(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return OtherPtrBaseExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleImplies(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return BoolToBVExpr::create(ImpliesExpr::create(BVToBoolExpr::create(Args[0]),
                                                  BVToBoolExpr::create(Args[1])));
}

ref<Expr> TranslateFunction::handleEnabled(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return BoolToBVExpr::create(SpecialVarRefExpr::create(bugle::Type(bugle::Type::Bool),
                                                        "__enabled"));
}

ref<Expr> TranslateFunction::handleReadHasOccurred(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return BoolToBVExpr::create(AccessHasOccurredExpr::create(
                              ArrayIdExpr::create(Args[0], TM->defaultRange()),
                              false));
}

ref<Expr> TranslateFunction::handleWriteHasOccurred(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return BoolToBVExpr::create(AccessHasOccurredExpr::create(
                              ArrayIdExpr::create(Args[0], TM->defaultRange()),
                              true));
}

ref<Expr> TranslateFunction::handleReadOffset(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result = AccessOffsetExpr::create(arrayIdExpr, false);
  Type range = arrayIdExpr->getType().range();

  if (range.isKind(Type::BV)) {
    if (range.width > 8) {
      result = BVMulExpr::create(BVConstExpr::create(
                                 TM->TD.getPointerSizeInBits(),
                                 range.width/8), result);
    }
  }
  return result;
}

ref<Expr> TranslateFunction::handleWriteOffset(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result = AccessOffsetExpr::create(arrayIdExpr, true);
  Type range = arrayIdExpr->getType().range();

  if (range.isKind(Type::BV)) {
    if (range.width > 8) {
      result = BVMulExpr::create(BVConstExpr::create(
                                 TM->TD.getPointerSizeInBits(),
                                 range.width/8), result);
    }
  }
  return result;
}

ref<Expr> TranslateFunction::handlePtrOffset(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return ArrayOffsetExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handlePtrBase(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return ArrayIdExpr::create(Args[0], TM->defaultRange());
}

ref<Expr> TranslateFunction::handleNotAccessed(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  ref<Expr> arrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> result = NotAccessedExpr::create(arrayIdExpr);
  return result;
}

ref<Expr> TranslateFunction::handleArraySnapshot(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  ref<Expr> dstArrayIdExpr = ArrayIdExpr::create(Args[0], TM->defaultRange());
  ref<Expr> srcArrayIdExpr = ArrayIdExpr::create(Args[1], TM->defaultRange());
  ref<Expr> E = ArraySnapshotExpr::create(dstArrayIdExpr, srcArrayIdExpr);
  BBB->addStmt(new EvalStmt(E));
  return 0;
}

ref<Expr> TranslateFunction::handleAtomic(bugle::BasicBlock *BBB,
    llvm::CallInst *CI, const std::vector<ref<Expr>> &Args) {
  assert(Args.size() > 0);
  ref<Expr> Ptr = Args[0],
            PtrArr = ArrayIdExpr::create(Ptr, TM->translateType(CI->getType())),
            PtrOfs = ArrayOffsetExpr::create(Ptr);
  Type ArrRangeTy = PtrArr->getType().range();
  Type AtomicTy = TM->translateType(CI->getType());
  assert(AtomicTy.width%32 == 0);
  if (ArrRangeTy.width > 8)
    PtrOfs = BVSDivExpr::create(PtrOfs,
                                BVConstExpr::create(TM->TD.getPointerSizeInBits(),
                                                    ArrRangeTy.width/8));

  std::vector<ref<Expr>> AtomicArgs;
  for (unsigned i = 1, e = Args.size(); i < e; ++i)
    AtomicArgs.push_back(Args[i]);

  ref<Expr> result;
  // If ArrRangeTy is Any, then we are using a null pointer.
  if (ArrRangeTy != Type(Type::Any)) {
    std::vector<ref<Expr>> Elems;
    unsigned NumElems = AtomicTy.width/ArrRangeTy.width;
    for (unsigned i = 0; i < NumElems; ++i) {
      ref<Expr> E = AtomicExpr::create(PtrArr, PtrOfs, AtomicArgs,
                                       CI->getCalledFunction()->getName(),
                                       NumElems, i+1);
      EvalStmt* ES = new EvalStmt(E);
      ES->setSourceLocs(currentSourceLocs);
      BBB->addStmt(ES);
      Elems.push_back(E);
      PtrOfs = BVAddExpr::create(PtrOfs,
                                 BVConstExpr::create(TM->TD.getPointerSizeInBits(), 1));
    }
    result = Expr::createBVConcatN(Elems);
  } else {
    // If we have a null pointer, add a fake load expression.
    ref<Expr> Div = BVConstExpr::createZero(AtomicTy.width/32);
    result = LoadExpr::create(PtrArr, Div, AtomicTy, LoadsAreTemporal);
  }

  return result;
}

ref<Expr> TranslateFunction::handleBarrierInvariant(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  assert (CI->getNumArgOperands() > 1);

  auto BF = BarrierInvariants[CI->getNumArgOperands()];
  if (!BF) {
    std::string S = CI->getCalledFunction()->getName().str();
    llvm::raw_string_ostream SS(S);
    SS << (CI->getNumArgOperands() - 1);
    BF = TM->BM->addFunction(SS.str());
    BarrierInvariants[CI->getNumArgOperands()] = BF;

    int count = 0;
    for (auto i = Args.begin(), e = Args.end(); i != e; ++i, ++count) {
      std::string S;
      llvm::raw_string_ostream SS(S);
      if (count == 0) {
        SS << "expr";
      } else {
        SS << "instantiation";
        SS << count;
      }
      BF->addArgument((*i)->getType(), SS.str());
    }

    BF->addAttribute("barrier_invariant");

  }

  auto CS = new CallStmt(BF, Args);
  CS->setSourceLocs(currentSourceLocs);
  BBB->addStmt(CS);
  return 0;
}


ref<Expr> TranslateFunction::handleBarrierInvariantBinary(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  assert (CI->getNumArgOperands() > 1);
  assert ((CI->getNumArgOperands() % 2) &&
    "Arguments to __barrier_invariant_binary should consist of barrier invariant"
    " followed by a sequence of *pairs* of instantiation arguments");

  auto BF = BinaryBarrierInvariants[CI->getNumArgOperands()];
  if (!BF) {
    std::string S = CI->getCalledFunction()->getName().str();
    llvm::raw_string_ostream SS(S);
    SS << ((CI->getNumArgOperands() - 1)/2);
    BF = TM->BM->addFunction(SS.str());
    BinaryBarrierInvariants[CI->getNumArgOperands()] = BF;

    int count = 0;
    for (auto i = Args.begin(), e = Args.end(); i != e; ++i, ++count) {
      std::string S;
      llvm::raw_string_ostream SS(S);
      if (count == 0) {
        SS << "expr";
      } else {
        SS << "instantiation";
        SS << (count/2);
        SS << "_";
        SS << ((count % 2) == 0 ? 2 : 1);
      }
      BF->addArgument((*i)->getType(), SS.str());
    }

    BF->addAttribute("binary_barrier_invariant");

  }

  auto CS = new CallStmt(BF, Args);
  CS->setSourceLocs(currentSourceLocs);
  BBB->addStmt(CS);
  return 0;
}

ref<Expr> TranslateFunction::handleMemset(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  auto MSI = dyn_cast<MemSetInst>(CI);
  auto Const = dyn_cast<ConstantInt>(MSI->getLength());

  if (!Const) {
    // Could emit a loop
    ErrorReporter::reportImplementationLimitation(
                       "memset with non-integer constant length not supported");
  }

  auto ConstValue = dyn_cast<ConstantInt>(MSI->getValue());

  if (!ConstValue) {
    // Could deal with expr
    ErrorReporter::reportImplementationLimitation(
                        "memset with non-integer constant value not supported");
  }

  ref<Expr> Dst = translateValue(MSI->getDest(), BBB),
            DstPtrArr = ArrayIdExpr::create(Dst, TM->defaultRange()),
            DstPtrOfs = ArrayOffsetExpr::create(Dst);
  unsigned Len = Const->getZExtValue();
  unsigned Val = ConstValue->getZExtValue();
  Type DstRangeTy = DstPtrArr->getType().range();

  if (DstRangeTy == Type(Type::Any) || DstRangeTy == Type(Type::Unknown)) {
    ErrorReporter::reportImplementationLimitation(
                                  "memset with null pointer destination or"
                                  " destinations of mixed types not supported");
  }

  assert(DstRangeTy.width % 8 == 0);
  unsigned NumElements = Len / (DstRangeTy.width/8);
  ref<Expr> DstDiv = Expr::createExactBVUDiv(DstPtrOfs, DstRangeTy.width/8);
  // Handle when Len can be rewritten as an integral number of element writes
  // Special case if Val is 0
  if (!DstDiv.isNull() &&
      (Len % (DstRangeTy.width/8) == 0) && 0 < NumElements &&
      (Val == 0 || DstRangeTy.width == 8)
     ) {
    for (unsigned i = 0; i != NumElements; ++i) {
      ref<Expr> ValExpr = BVConstExpr::create(DstRangeTy.width, Val);
      ref<Expr> StoreOfs =
        BVAddExpr::create(DstDiv, BVConstExpr::create(Dst->getType().width, i));
      addEvalStmt(BBB, CI, ValExpr);
      StoreStmt* SS = new StoreStmt(DstPtrArr, StoreOfs, ValExpr);
      SS->setSourceLocs(currentSourceLocs);
      BBB->addStmt(SS);
    }
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (DstPtrArr->computeArrayCandidates(Globals)) {
      std::transform(Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }
  return 0;
}

ref<Expr> TranslateFunction::handleMemcpy(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  auto MCI = dyn_cast<MemCpyInst>(CI);
  auto Const = dyn_cast<ConstantInt>(MCI->getLength());

  if (!Const) {
    // Could emit a loop
    ErrorReporter::reportImplementationLimitation(
                       "memcpy with non-integer constant length not supported");
  }

  ref<Expr> Src = translateValue(MCI->getSource(), BBB),
            Dst = translateValue(MCI->getDest(), BBB),
            SrcPtrArr = ArrayIdExpr::create(Src, TM->defaultRange()),
            DstPtrArr = ArrayIdExpr::create(Dst, TM->defaultRange()),
            SrcPtrOfs = ArrayOffsetExpr::create(Src),
            DstPtrOfs = ArrayOffsetExpr::create(Dst);
  unsigned Len = Const->getZExtValue();
  Type SrcRangeTy = SrcPtrArr->getType().range(),
       DstRangeTy = DstPtrArr->getType().range();

  if (DstRangeTy == Type(Type::Any) || DstRangeTy == Type(Type::Unknown)) {
    ErrorReporter::reportImplementationLimitation(
                                  "memset with null pointer destination or"
                                  " destinations of mixed types not supported");
  }

  if (SrcRangeTy == Type(Type::Any) || SrcRangeTy == Type(Type::Unknown)) {
    ErrorReporter::reportImplementationLimitation(
                                       "memset with null pointer source or"
                                       " sources of mixed types not supported");
  }

  assert(SrcRangeTy.width % 8 == 0);
  assert(DstRangeTy.width % 8 == 0);
  unsigned NumElements = Len / (SrcRangeTy.width/8);
  ref<Expr> SrcDiv = Expr::createExactBVUDiv(SrcPtrOfs, SrcRangeTy.width/8);
  ref<Expr> DstDiv = Expr::createExactBVUDiv(DstPtrOfs, DstRangeTy.width/8);
  // Handle matching source and destination range types where Len can be
  // rewritten as an integral number of element read/writes
  if (SrcRangeTy == DstRangeTy &&
      !SrcDiv.isNull() && !DstDiv.isNull() &&
      (Len % (SrcRangeTy.width/8) == 0) && 0 < NumElements) {
    for (unsigned i = 0; i != NumElements; ++i) {
      ref<Expr> LoadOfs =
        BVAddExpr::create(SrcDiv, BVConstExpr::create(Src->getType().width, i));
      ref<Expr> Val = LoadExpr::create(SrcPtrArr, LoadOfs, SrcRangeTy, LoadsAreTemporal);
      ref<Expr> StoreOfs =
        BVAddExpr::create(DstDiv, BVConstExpr::create(Dst->getType().width, i));
      addEvalStmt(BBB, CI, Val);
      StoreStmt* SS = new StoreStmt(DstPtrArr, StoreOfs, Val);
      SS->setSourceLocs(currentSourceLocs);
      BBB->addStmt(SS);
    }
  } else {
    TM->NeedAdditionalByteArrayModels = true;
    std::set<GlobalArray *> Globals;
    if (SrcPtrArr->computeArrayCandidates(Globals) &&
        DstPtrArr->computeArrayCandidates(Globals)) {
      std::transform(Globals.begin(), Globals.end(),
          std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
          [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
    } else {
      TM->NextModelAllAsByteArray = true;
    }
  }
  return 0;
}

ref<Expr> TranslateFunction::handleTrap(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const std::vector<ref<Expr>> &Args) {
  Stmt *assertStmt = new AssertStmt(BoolConstExpr::create(false),
                                    false, false, false);
  assertStmt->setSourceLocs(currentSourceLocs);
  BBB->addStmt(assertStmt);
  return 0;
}

static std::string mkDimName(const std::string &prefix, ref<Expr> dim) {
  auto CE = dyn_cast<BVConstExpr>(dim);
  if (!CE)
    ErrorReporter::reportImplementationLimitation("Unsupported variable dimension");
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

ref<Expr> TranslateFunction::handleGetLocalId(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Type t = TM->translateType(CI->getType());
  return mkLocalId(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetGroupId(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Type t = TM->translateType(CI->getType());
  return mkGroupId(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetLocalSize(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Type t = TM->translateType(CI->getType());
  return mkLocalSize(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetNumGroups(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Type t = TM->translateType(CI->getType());
  return mkNumGroups(t, Args[0]);
}

ref<Expr> TranslateFunction::handleGetImageWidth(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return GetImageWidthExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleGetImageHeight(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return GetImageHeightExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleCos(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FCosExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handleExp(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FExpExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handleFabs(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FAbsExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handleFloor(bugle::BasicBlock *BBB,
                                         llvm::CallInst *CI,
                                         const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FFloorExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handleFma(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  ref<Expr> M =
    maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0], Args[1], FMulExpr::create);
  return
    maybeTranslateSIMDInst(BBB, Ty, Ty, M, Args[2], FAddExpr::create);
}

ref<Expr> TranslateFunction::handleFrexpExp(bugle::BasicBlock *BBB,
                                            llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  return FrexpExpExpr::create(cast<IntegerType>(CI->getType())->getBitWidth(),
                              Args[0]);
}

ref<Expr> TranslateFunction::handleFrexpFrac(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  return FrexpFracExpr::create(Args[0]);
}

ref<Expr> TranslateFunction::handleLog(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FLogExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handlePow(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0], Args[1],
                                [&](ref<Expr> LHS, ref<Expr> RHS) {
    return FPowExpr::create(LHS, RHS);
  });
}

ref<Expr> TranslateFunction::handleSin(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FSinExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handleSqrt(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FSqrtExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handleRsqrt(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const std::vector<ref<Expr>> &Args) {
  llvm::Type *Ty = CI->getType();
  return maybeTranslateSIMDInst(BBB, Ty, Ty, Args[0],
                                [&](llvm::Type *T, ref<Expr> E) {
    return FRsqrtExpr::create(E);
  });
}

ref<Expr> TranslateFunction::handleAddNoovflUnsigned(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  ref<Expr> E = AddNoovflExpr::create(Args[0], Args[1], false);
  BBB->addStmt(new EvalStmt(E));
  return E;
}

ref<Expr> TranslateFunction::handleAddNoovflSigned(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  ref<Expr> E = AddNoovflExpr::create(Args[0], Args[1], true);
  BBB->addStmt(new EvalStmt(E));
  return E;
}

ref<Expr> TranslateFunction::handleAddNoovflPredicate(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  return AddNoovflPredicateExpr::create(Args);
}

ref<Expr> TranslateFunction::handleAdd(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  return BVAddExpr::create(Args[0], Args[1]);
}

ref<Expr> TranslateFunction::handleUninterpretedFunction(bugle::BasicBlock *BBB,
                                             llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  return UninterpretedFunctionExpr::create(CI->getCalledFunction()->getName().
                                      substr(strlen("__uninterpreted_function_")),
                                   TM->translateType(CI->getCalledFunction()->getReturnType()),
                                   Args);
}

ref<Expr> TranslateFunction::handleIte(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const std::vector<ref<Expr>> &Args) {
  return IfThenElseExpr::create(BVToBoolExpr::create(Args[0]), Args[1], Args[2]);
}

ref<Expr> TranslateFunction::handleAtomicHasTakenValue(bugle::BasicBlock *BBB,
                                       llvm::CallInst *CI,
                                       const std::vector<ref<Expr>> &Args) {
  return BoolToBVExpr::create(
    AtomicHasTakenValueExpr::create(Args[0], Args[1], Args[2]));
}

void TranslateFunction::addEvalStmt(bugle::BasicBlock *BBB,
                                    llvm::Instruction *I, ref<Expr> E) {
  auto ES = BBB->addEvalStmt(E);
  if (ES)
    ES->setSourceLocs(currentSourceLocs);
}

ref<Expr> TranslateFunction::maybeTranslateSIMDInst(bugle::BasicBlock *BBB,
                             llvm::Type *Ty, llvm::Type *OpTy,
                             ref<Expr> Op,
                          std::function<ref<Expr>(llvm::Type *, ref<Expr>)> F) {
  if (!isa<VectorType>(Ty))
    return F(Ty, Op);

  auto VT = cast<VectorType>(Ty);
  unsigned NumElems = VT->getNumElements();
  assert(cast<VectorType>(OpTy)->getNumElements() == NumElems);
  unsigned ElemWidth = Op->getType().width / NumElems;
  std::vector<ref<Expr>> Elems;
  for (unsigned i = 0; i < NumElems; ++i) {
    ref<Expr> Opi = BVExtractExpr::create(Op, i*ElemWidth, ElemWidth);
    ref<Expr> Elem = F(VT->getElementType(), Opi);
    BBB->addEvalStmt(Elem);
    Elems.push_back(Elem);
  }
  return Expr::createBVConcatN(Elems);
}

ref<Expr> TranslateFunction::maybeTranslateSIMDInst(bugle::BasicBlock *BBB,
                             llvm::Type *Ty, llvm::Type *OpTy,
                             ref<Expr> LHS, ref<Expr> RHS,
                             std::function<ref<Expr>(ref<Expr>, ref<Expr>)> F) {
  if (!isa<VectorType>(Ty))
    return F(LHS, RHS);

  auto VT = cast<VectorType>(Ty);
  unsigned NumElems = VT->getNumElements();
  assert(cast<VectorType>(OpTy)->getNumElements() == NumElems);
  unsigned ElemWidth = LHS->getType().width / NumElems;
  std::vector<ref<Expr>> Elems;
  for (unsigned i = 0; i < NumElems; ++i) {
    ref<Expr> LHSi = BVExtractExpr::create(LHS, i*ElemWidth, ElemWidth);
    ref<Expr> RHSi = BVExtractExpr::create(RHS, i*ElemWidth, ElemWidth);
    ref<Expr> Elem = F(LHSi, RHSi);
    BBB->addEvalStmt(Elem);
    Elems.push_back(Elem);
  }
  return Expr::createBVConcatN(Elems);
}

void TranslateFunction::translateInstruction(bugle::BasicBlock *BBB,
                                             Instruction *I) {

  SourceLocsRef SLI = extractSourceLocs(I);
  if (SLI.get() != 0)
    currentSourceLocs = SLI;

  ref<Expr> E;
  if (auto BO = dyn_cast<BinaryOperator>(I)) {
    ref<Expr> LHS = translateValue(BO->getOperand(0), BBB),
              RHS = translateValue(BO->getOperand(1), BBB);
    ref<Expr> (*F)(ref<Expr>, ref<Expr>);
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
    case BinaryOperator::Shl:  F = BVShlExpr::create;  break;
    case BinaryOperator::AShr: F = BVAShrExpr::create; break;
    case BinaryOperator::LShr: F = BVLShrExpr::create; break;
    case BinaryOperator::And:  F = BVAndExpr::create;  break;
    case BinaryOperator::Or:   F = BVOrExpr::create;   break;
    case BinaryOperator::Xor:  F = BVXorExpr::create;  break;
    default:
      ErrorReporter::reportImplementationLimitation("Unsupported binary operator");
    }
    E = maybeTranslateSIMDInst(BBB, BO->getType(), BO->getType(), LHS, RHS, F);
  } else if (auto GEPI = dyn_cast<GetElementPtrInst>(I)) {
    ref<Expr> Ptr = translateValue(GEPI->getPointerOperand(), BBB);
    E = TM->translateGEP(Ptr, klee::gep_type_begin(GEPI),
                         klee::gep_type_end(GEPI),
                         [&](Value *V) { return translateValue(V, BBB); });
  } else if (auto AI = dyn_cast<AllocaInst>(I)) {
    GlobalArray *GA = TM->getGlobalArray(AI);
    E = PointerExpr::create(GlobalArrayRefExpr::create(GA),
                        BVConstExpr::createZero(TM->TD.getPointerSizeInBits()));
  } else if (auto LI = dyn_cast<LoadInst>(I)) {
    ref<Expr> Ptr = translateValue(LI->getPointerOperand(), BBB),
              PtrArr = ArrayIdExpr::create(Ptr, TM->defaultRange()),
              PtrOfs = ArrayOffsetExpr::create(Ptr);
    Type ArrRangeTy = PtrArr->getType().range();
    Type LoadTy = TM->translateType(LI->getType()), LoadElTy = LoadTy;
    auto VT = dyn_cast<VectorType>(LI->getType());
    if (VT)
      LoadElTy = TM->translateType(VT->getElementType());
    assert(LoadTy.width % 8 == 0);
    ref<Expr> Div;
    if ((ArrRangeTy == LoadElTy || ArrRangeTy == Type(Type::Any)) &&
        !(Div = Expr::createExactBVUDiv(PtrOfs, LoadElTy.width/8)).isNull()) {
      if (VT) {
        std::vector<ref<Expr>> ElemsLoaded;
        for (unsigned i = 0; i != VT->getNumElements(); ++i) {
          ref<Expr> ElemOfs =
            BVAddExpr::create(Div,
                              BVConstExpr::create(Div->getType().width, i));
          ref<Expr> ValElem = LoadExpr::create(PtrArr, ElemOfs, LoadElTy, LoadsAreTemporal);
          addEvalStmt(BBB, I, ValElem);
          if (LoadElTy.isKind(Type::Pointer))
            ValElem = PtrToBVExpr::create(ValElem);
          ElemsLoaded.push_back(ValElem);
        }
        E = Expr::createBVConcatN(ElemsLoaded);
      } else {
        E = LoadExpr::create(PtrArr, Div, LoadElTy, LoadsAreTemporal);
      }
    } else if (ArrRangeTy == Type(Type::BV, 8)) {
      std::vector<ref<Expr> > BytesLoaded;
      for (unsigned i = 0; i != LoadTy.width / 8; ++i) {
        ref<Expr> PtrByteOfs =
          BVAddExpr::create(PtrOfs,
                            BVConstExpr::create(PtrOfs->getType().width, i));
        ref<Expr> ValByte = LoadExpr::create(PtrArr, PtrByteOfs, ArrRangeTy, LoadsAreTemporal);
        BytesLoaded.push_back(ValByte);
        addEvalStmt(BBB, I, ValByte);
      }
      E = Expr::createBVConcatN(BytesLoaded);
      if (LoadTy.isKind(Type::Pointer))
        E = BVToPtrExpr::create(E);
    } else {
      TM->NeedAdditionalByteArrayModels = true;
      std::set<GlobalArray *> Globals;
      if (PtrArr->computeArrayCandidates(Globals)) {
        std::transform(Globals.begin(), Globals.end(),
            std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
            [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
      } else {
        TM->NextModelAllAsByteArray = true;
      }
      E = TM->translateArbitrary(LoadTy);
    }
  } else if (auto SI = dyn_cast<StoreInst>(I)) {
    ref<Expr> Ptr = translateValue(SI->getPointerOperand(), BBB),
              Val = translateValue(SI->getValueOperand(), BBB),
              PtrArr = ArrayIdExpr::create(Ptr, TM->defaultRange()),
              PtrOfs = ArrayOffsetExpr::create(Ptr);
    Type ArrRangeTy = PtrArr->getType().range();
    Type StoreTy = Val->getType(), StoreElTy = StoreTy;
    auto VT = dyn_cast<VectorType>(SI->getValueOperand()->getType());
    if (VT)
      StoreElTy = TM->translateType(VT->getElementType());
    assert(StoreTy.width % 8 == 0);
    ref<Expr> Div;
    // If ArrRangeTy is Any, then we are using a null pointer for storing
    if ((ArrRangeTy == StoreElTy || ArrRangeTy == Type(Type::Any)) &&
        !(Div = Expr::createExactBVUDiv(PtrOfs, StoreElTy.width/8)).isNull()) {
      if (VT) {
        for (unsigned i = 0; i != VT->getNumElements(); ++i) {
          ref<Expr> ElemOfs =
            BVAddExpr::create(Div,
                              BVConstExpr::create(Div->getType().width, i));
          ref<Expr> ValElem =
            BVExtractExpr::create(Val, i*StoreElTy.width, StoreElTy.width);
          if (StoreElTy.isKind(Type::Pointer))
            ValElem = BVToPtrExpr::create(ValElem);
          StoreStmt* SS = new StoreStmt(PtrArr, ElemOfs, ValElem);
          SS->setSourceLocs(currentSourceLocs);
          BBB->addStmt(SS);
        }
      } else {
        StoreStmt* SS = new StoreStmt(PtrArr, Div, Val);
        SS->setSourceLocs(currentSourceLocs);
        BBB->addStmt(SS);
      }
    } else if (ArrRangeTy == Type(Type::BV, 8)) {
      if (StoreTy.isKind(Type::Pointer)) {
        Val = PtrToBVExpr::create(Val);
        addEvalStmt(BBB, I, Val);
      }
      for (unsigned i = 0; i != Val->getType().width / 8; ++i) {
        ref<Expr> PtrByteOfs =
          BVAddExpr::create(PtrOfs,
                            BVConstExpr::create(PtrOfs->getType().width, i));
        ref<Expr> ValByte =
          BVExtractExpr::create(Val, i*8, 8); // Assumes little endian
        StoreStmt* SS = new StoreStmt(PtrArr, PtrByteOfs, ValByte);
        SS->setSourceLocs(currentSourceLocs);
        BBB->addStmt(SS);
      }
    } else {
      TM->NeedAdditionalByteArrayModels = true;
      std::set<GlobalArray *> Globals;
      if (PtrArr->computeArrayCandidates(Globals)) {
        std::transform(Globals.begin(), Globals.end(),
            std::inserter(TM->ModelAsByteArray, TM->ModelAsByteArray.begin()),
            [&](GlobalArray *A) { return TM->GlobalValueMap[A]; });
      } else {
        TM->NextModelAllAsByteArray = true;
      }
    }
    return;
  } else if (auto II = dyn_cast<ICmpInst>(I)) {
    ref<Expr> LHS = translateValue(II->getOperand(0), BBB),
              RHS = translateValue(II->getOperand(1), BBB);
    E = maybeTranslateSIMDInst(BBB, II->getType(), II->getOperand(0)->getType(),
                               LHS, RHS,
                               [&](ref<Expr> LHS, ref<Expr> RHS) -> ref<Expr> {
      ref<Expr> E;
      if (II->getPredicate() == ICmpInst::ICMP_EQ)
        E = EqExpr::create(LHS, RHS);
      else if (II->getPredicate() == ICmpInst::ICMP_NE)
        E = NeExpr::create(LHS, RHS);
      else if (LHS->getType().isKind(Type::Pointer)) {
        assert(RHS->getType().isKind(Type::Pointer));
        switch (II->getPredicate()) {
        case ICmpInst::ICMP_ULT:
        case ICmpInst::ICMP_SLT: E = Expr::createPtrLt(LHS, RHS); break;
        case ICmpInst::ICMP_ULE:
        case ICmpInst::ICMP_SLE: E = Expr::createPtrLe(LHS, RHS); break;
        case ICmpInst::ICMP_UGT:
        case ICmpInst::ICMP_SGT: E = Expr::createPtrLt(RHS, LHS); break;
        case ICmpInst::ICMP_UGE:
        case ICmpInst::ICMP_SGE: E = Expr::createPtrLe(RHS, LHS); break;
        default:
          ErrorReporter::reportImplementationLimitation("Unsupported ptr icmp");
        }
      } else {
        assert(RHS->getType().isKind(Type::BV));
        switch (II->getPredicate()) {
        case ICmpInst::ICMP_UGT: E = BVUgtExpr::create(LHS, RHS); break;
        case ICmpInst::ICMP_UGE: E = BVUgeExpr::create(LHS, RHS); break;
        case ICmpInst::ICMP_ULT: E = BVUltExpr::create(LHS, RHS); break;
        case ICmpInst::ICMP_ULE: E = BVUleExpr::create(LHS, RHS); break;
        case ICmpInst::ICMP_SGT: E = BVSgtExpr::create(LHS, RHS); break;
        case ICmpInst::ICMP_SGE: E = BVSgeExpr::create(LHS, RHS); break;
        case ICmpInst::ICMP_SLT: E = BVSltExpr::create(LHS, RHS); break;
        case ICmpInst::ICMP_SLE: E = BVSleExpr::create(LHS, RHS); break;
        default:
          ErrorReporter::reportImplementationLimitation("Unsupported icmp");
        }
      }
      addEvalStmt(BBB, I, E);
      return BoolToBVExpr::create(E);
    });
  } else if (auto FI = dyn_cast<FCmpInst>(I)) {
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
      addEvalStmt(BBB, I, E);
      return BoolToBVExpr::create(E);
    });
  } else if (auto ZEI = dyn_cast<ZExtInst>(I)) {
    ref<Expr> Op = translateValue(ZEI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, ZEI->getType(),
                               ZEI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return BVZExtExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto SEI = dyn_cast<SExtInst>(I)) {
    ref<Expr> Op = translateValue(SEI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, SEI->getType(),
                               SEI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return BVSExtExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto FPSII = dyn_cast<FPToSIInst>(I)) {
    ref<Expr> Op = translateValue(FPSII->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, FPSII->getType(),
                               FPSII->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return FPToSIExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto FPUII = dyn_cast<FPToUIInst>(I)) {
    ref<Expr> Op = translateValue(FPUII->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, FPUII->getType(),
                               FPUII->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return FPToUIExpr::create(cast<IntegerType>(Ty)->getBitWidth(), Op);
    });
  } else if (auto SIFPI = dyn_cast<SIToFPInst>(I)) {
    ref<Expr> Op = translateValue(SIFPI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, SIFPI->getType(),
                               SIFPI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return SIToFPExpr::create(TM->TD.getTypeSizeInBits(Ty), Op);
    });
  } else if (auto UIFPI = dyn_cast<UIToFPInst>(I)) {
    ref<Expr> Op = translateValue(UIFPI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, UIFPI->getType(),
                               UIFPI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return UIToFPExpr::create(TM->TD.getTypeSizeInBits(Ty), Op);
    });
  } else if (isa<FPExtInst>(I) || isa<FPTruncInst>(I)) {
    auto CI = cast<CastInst>(I);
    ref<Expr> Op = translateValue(CI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, CI->getType(),
                               CI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return FPConvExpr::create(TM->TD.getTypeSizeInBits(Ty), Op);
    });
  } else if (auto TI = dyn_cast<TruncInst>(I)) {
    ref<Expr> Op = translateValue(TI->getOperand(0), BBB);
    E = maybeTranslateSIMDInst(BBB, TI->getType(),
                               TI->getOperand(0)->getType(), Op,
                               [&](llvm::Type *Ty, ref<Expr> Op) {
      return BVExtractExpr::create(Op, 0, cast<IntegerType>(Ty)->getBitWidth());
    });
  } else if (auto I2PI = dyn_cast<IntToPtrInst>(I)) {
    ref<Expr> Op = translateValue(I2PI->getOperand(0), BBB);
    E = BVToPtrExpr::create(Op);
  } else if (auto P2II = dyn_cast<PtrToIntInst>(I)) {
    ref<Expr> Op = translateValue(P2II->getOperand(0), BBB);
    E = PtrToBVExpr::create(Op);
  } else if (auto BCI = dyn_cast<BitCastInst>(I)) {
    ref<Expr> Op = translateValue(BCI->getOperand(0), BBB);
    E = TM->translateBitCast(BCI->getSrcTy(), BCI->getDestTy(), Op);
    if (Op.get() == E.get()) {
      ValueExprMap[I] = Op;
      return;
    }
  } else if (auto SI = dyn_cast<SelectInst>(I)) {
    ref<Expr> Cond = translateValue(SI->getCondition(), BBB),
              TrueVal = translateValue(SI->getTrueValue(), BBB),
              FalseVal = translateValue(SI->getFalseValue(), BBB);
    if (auto VT = dyn_cast<VectorType>(SI->getCondition()->getType())) {
      unsigned elementBitWidth = TrueVal->getType().width / VT->getNumElements();
      E = 0;
      for (unsigned i = 0; i < VT->getNumElements(); ++i) {
        ref<Expr> Ite = IfThenElseExpr::create(BVToBoolExpr::create(BVExtractExpr::create(Cond, i, 1)),
                                                                    BVExtractExpr::create(TrueVal, i*elementBitWidth, elementBitWidth),
                                                                    BVExtractExpr::create(FalseVal, i*elementBitWidth, elementBitWidth));
        E = (i == 0) ? Ite : BVConcatExpr::create(Ite, E);
      }
    } else {
      Cond = BVToBoolExpr::create(Cond);
      E = IfThenElseExpr::create(Cond, TrueVal, FalseVal);
    }
  } else if (auto EEI = dyn_cast<ExtractElementInst>(I)) {
    ref<Expr> Vec = translateValue(EEI->getVectorOperand(), BBB),
              Idx = translateValue(EEI->getIndexOperand(), BBB);
    unsigned EltBits = TM->TD.getTypeSizeInBits(EEI->getType());
    BVConstExpr *CEIdx = cast<BVConstExpr>(Idx);
    unsigned UIdx = CEIdx->getValue().getZExtValue();
    E = BVExtractExpr::create(Vec, EltBits*UIdx, EltBits);
  } else if (auto IEI = dyn_cast<InsertElementInst>(I)) {
    ref<Expr> Vec = translateValue(IEI->getOperand(0), BBB),
              NewElt = translateValue(IEI->getOperand(1), BBB),
              Idx = translateValue(IEI->getOperand(2), BBB);
    llvm::Type *EltType = IEI->getType()->getElementType();
    unsigned EltBits = TM->TD.getTypeSizeInBits(EltType);
    unsigned ElemCount = IEI->getType()->getNumElements();
    BVConstExpr *CEIdx = cast<BVConstExpr>(Idx);
    unsigned UIdx = CEIdx->getValue().getZExtValue();
    std::vector<ref<Expr>> Elems;
    for (unsigned i = 0; i != ElemCount; ++i) {
      Elems.push_back(i == UIdx ? NewElt
                              : BVExtractExpr::create(Vec, EltBits*i, EltBits));
    }
    E = Expr::createBVConcatN(Elems);
  } else if (auto SVI = dyn_cast<ShuffleVectorInst>(I)) {
    ref<Expr> Vec1 = translateValue(SVI->getOperand(0), BBB),
              Vec2 = translateValue(SVI->getOperand(1), BBB);
    unsigned EltBits =
      TM->TD.getTypeSizeInBits(SVI->getType()->getElementType());
    unsigned VecElemCount = cast<VectorType>(SVI->getOperand(0)->getType())
                              ->getNumElements();
    unsigned ResElemCount = SVI->getType()->getNumElements();
    std::vector<ref<Expr>> Elems;
    for (unsigned i = 0; i != ResElemCount; ++i) {
      ref<Expr> L;
      int MaskValI = SVI->getMaskValue(i);
      if (MaskValI < 0)
        L = BVConstExpr::create(EltBits, 0);
      else {
        unsigned MaskVal = (unsigned) MaskValI;
        if (MaskVal < VecElemCount)
          L = BVExtractExpr::create(Vec1, EltBits*MaskVal, EltBits);
        else
          L = BVExtractExpr::create(Vec2, EltBits*(MaskVal-VecElemCount),
                                    EltBits);
      }
      Elems.push_back(L);
    }
    E = Expr::createBVConcatN(Elems);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    auto F = CI->getCalledFunction();

    if (!F)
      ErrorReporter::reportImplementationLimitation("Only direct calls supported");

    CallSite CS(CI);
    std::vector<ref<Expr>> Args;
    std::transform(CS.arg_begin(), CS.arg_end(), std::back_inserter(Args),
                   [&](Value *V) { return translateValue(V, BBB); });

    if (auto II = dyn_cast<IntrinsicInst>(CI)) {
      auto ID = II->getIntrinsicID();
      auto SFII = SpecialFunctionMap.Intrinsics.find(ID);
      if (SFII != SpecialFunctionMap.Intrinsics.end()) {
        E = (this->*SFII->second)(BBB, CI, Args);
        assert(E.isNull() == CI->getType()->isVoidTy());
        if (E.isNull())
          return;
      } else {
        std::string msg = "Intrinsic '" + Intrinsic::getName(ID) + "' not supported";
        ErrorReporter::reportImplementationLimitation(msg);
      }
    } else {
      auto SFI = SpecialFunctionMap.Functions.find(F->getName());
      if (SFI != SpecialFunctionMap.Functions.end()) {
        E = (this->*SFI->second)(BBB, CI, Args);
        assert(E.isNull() == CI->getType()->isVoidTy());
        if (E.isNull())
          return;
      } else {
        std::transform(Args.begin(), Args.end(), F->arg_begin(), Args.begin(),
                       [&](ref<Expr> E, Argument &Arg) {
                         return TM->modelValue(&Arg, E);
                       });

        auto FI = TM->FunctionMap.find(F);
        assert(FI != TM->FunctionMap.end() && "Couldn't find function in map!");
        if (CI->getType()->isVoidTy()) {
          auto CS = new CallStmt(FI->second, Args);
          CS->setSourceLocs(currentSourceLocs);
          BBB->addStmt(CS);
          TM->CallSites[F].push_back(&CS->getArgs());
          return;
        } else {
          E = CallExpr::create(FI->second, Args);
          addEvalStmt(BBB, I, E);
          ValueExprMap[I] = TM->unmodelValue(F, E);
          if (auto CE = dyn_cast<CallExpr>(E))
            TM->CallSites[F].push_back(&CE->getArgs());
          return;
        }
      }
    }
  } else if (auto RI = dyn_cast<ReturnInst>(I)) {
    if (auto V = RI->getReturnValue()) {
      assert(ReturnVar && "Returning value without return variable?");
      ref<Expr> Val = TM->modelValue(F, translateValue(V, BBB));
      BBB->addStmt(new VarAssignStmt(ReturnVar, Val));
      ReturnVals.push_back(Val);
    }
    BBB->addStmt(new ReturnStmt);
    return;
  } else if (auto BI = dyn_cast<BranchInst>(I)) {
    if (BI->isConditional()) {
      ref<Expr> Cond = BVToBoolExpr::create(translateValue(BI->getCondition(), BBB));

      bugle::BasicBlock *TrueBB = BF->addBasicBlock("truebb");
      TrueBB->addStmt(new AssumeStmt(Cond, /*partition=*/true));
      addPhiAssigns(TrueBB, I->getParent(), BI->getSuccessor(0));
      TrueBB->addStmt(new GotoStmt(BasicBlockMap[BI->getSuccessor(0)]));

      bugle::BasicBlock *FalseBB = BF->addBasicBlock("falsebb");
      FalseBB->addStmt(new AssumeStmt(NotExpr::create(Cond),
                                      /*partition=*/true));
      addPhiAssigns(FalseBB, I->getParent(), BI->getSuccessor(1));
      FalseBB->addStmt(new GotoStmt(BasicBlockMap[BI->getSuccessor(1)]));

      std::vector<bugle::BasicBlock *> BBs;
      BBs.push_back(TrueBB);
      BBs.push_back(FalseBB);
      BBB->addStmt(new GotoStmt(BBs));
    } else {
      addPhiAssigns(BBB, I->getParent(), BI->getSuccessor(0));
      BBB->addStmt(new GotoStmt(BasicBlockMap[BI->getSuccessor(0)]));
    }
    return;
  } else if (auto SI = dyn_cast<SwitchInst>(I)) {
    ref<Expr> Cond = translateValue(SI->getCondition(), BBB);
    ref<Expr> DefaultExpr = BoolConstExpr::create(true);
    std::vector<bugle::BasicBlock *> Succs;

    for (auto i = SI->case_begin(), e = SI->case_end(); i != e; ++i) {
      ref<Expr> Val = TM->translateConstant(i.getCaseValue());
      bugle::BasicBlock *BB = BF->addBasicBlock("casebb");
      Succs.push_back(BB);
      BB->addStmt(new AssumeStmt(EqExpr::create(Cond, Val),/*partition=*/true));
      addPhiAssigns(BB, SI->getParent(), i.getCaseSuccessor());
      BB->addStmt(new GotoStmt(BasicBlockMap[i.getCaseSuccessor()]));
      DefaultExpr = AndExpr::create(DefaultExpr, NeExpr::create(Cond, Val));
    }

    bugle::BasicBlock *DefaultBB = BF->addBasicBlock("defaultbb");
    Succs.push_back(DefaultBB);
    DefaultBB->addStmt(new AssumeStmt(DefaultExpr, /*partition=*/true));
    addPhiAssigns(DefaultBB, SI->getParent(),
                  SI->case_default().getCaseSuccessor());
    DefaultBB->addStmt(
      new GotoStmt(BasicBlockMap[SI->case_default().getCaseSuccessor()]));

    BBB->addStmt(new GotoStmt(Succs));
    return;
  } else if (auto PN = dyn_cast<PHINode>(I)) {
    ValueExprMap[I] =
      TM->unmodelValue(PN, VarRefExpr::create(getPhiVariable(PN)));
    return;
  } else if (dyn_cast<UnreachableInst>(I)) {
    Stmt *assertStmt = new AssertStmt(BoolConstExpr::create(false), false, false, false);
    assertStmt->setSourceLocs(currentSourceLocs);
    BBB->addStmt(assertStmt);
    return;
  } else {
    std::string name = I->getOpcodeName();
    std::string msg = "Instruction '" + name + "' not supported";
    ErrorReporter::reportImplementationLimitation(msg);
  }
  ValueExprMap[I] = E;
  addEvalStmt(BBB, I, E);
  return;
}

void TranslateFunction::translateBasicBlock(bugle::BasicBlock *BBB,
                                            llvm::BasicBlock *BB) {
  for (auto i = BB->begin(), e = BB->end(); i != e; ++i)
    translateInstruction(BBB, &*i);
}
