#include "bugle/Translator/TranslateFunction.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/BasicBlock.h"
#include "bugle/Expr.h"
#include "bugle/GlobalArray.h"
#include "bugle/Module.h"
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

static cl::opt<bool>
DumpTranslatedExprs("dump-translated-exprs", cl::Hidden, cl::init(false),
  cl::desc("Dump each translated expression below the instruction which "
           "generated it"));

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

TranslateFunction::SpecialFnMapTy &
TranslateFunction::initSpecialFunctionMap(TranslateModule::SourceLanguage SL) {
  SpecialFnMapTy &SpecialFunctionMap = SpecialFunctionMaps[SL];
  if (SpecialFunctionMap.Functions.empty()) {
    auto &fns = SpecialFunctionMap.Functions;
    fns["llvm.lifetime.start"] = &TranslateFunction::handleNoop;
    fns["llvm.lifetime.end"] = &TranslateFunction::handleNoop;
    fns["bugle_assert"] = &TranslateFunction::handleAssert;
    fns["__assert"] = &TranslateFunction::handleAssert;
    fns["__invariant"] = &TranslateFunction::handleAssert;
    fns["__global_assert"] = &TranslateFunction::handleGlobalAssert;
    fns["__global_invariant"] = &TranslateFunction::handleGlobalAssert;
    fns["__candidate_invariant"] = &TranslateFunction::handleCandidateAssert;
    fns["__candidate_global_invariant"] = &TranslateFunction::handleCandidateGlobalAssert;
    fns["__candidate_assert"] = &TranslateFunction::handleCandidateAssert;
    fns["__candidate_global_assert"] = &TranslateFunction::handleCandidateGlobalAssert;
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
    const unsigned NOOVFL_PREDICATE_MAX_ARITY = 20;
    const std::string tys[] = { "char", "short", "int", "long" };
    for(unsigned j = 0; j < 4; ++j) {
      std::string t = tys[j];
      for(unsigned i = 0; i <= NOOVFL_PREDICATE_MAX_ARITY; ++i) {
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

    {
      const std::string atomics[] = { "__atomic_add" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_sub" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_xchg" ,     "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_float", "_global_float", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_min" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_max" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_and" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_or" ,       "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_xor" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_cmpxchg" ,  "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_inc" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,
                                      "__atomic_dec" ,      "_local_int", "_global_int", "_local_unsigned_int", "_global_unsigned_int", "_local_long", "_global_long", "_local_unsigned_long", "_global_unsigned_long", "" ,

                                      "__atomicAdd" ,   "_int", "_unsigned_int", "_unsigned_long_long_int", "_float", "",
                                      "__atomicSub" ,   "_int", "_unsigned_int", "",
                                      "__atomicExch" ,  "_int", "_unsigned_int", "_unsigned_long_long_int", "_float", "",
                                      "__atomicMin" ,   "_int", "_unsigned_int", "_unsigned_long_long_int", "",
                                      "__atomicMax" ,   "_int", "_unsigned_int", "_unsigned_long_long_int", "",
                                      "__atomicAnd" ,   "_int", "_unsigned_int", "_unsigned_long_long_int", "",
                                      "__atomicOr" ,    "_int", "_unsigned_int", "_unsigned_long_long_int", "",
                                      "__atomicXor" ,   "_int", "_unsigned_int", "_unsigned_long_long_int", "",
                                      "__atomicInc",    "_unsigned_int", "",
                                      "__atomicDec",    "_unsigned_int", "",
                                      "__atomicCAS",    "_int", "_unsigned_int", "_unsigned_long_long_int", "",
         ""
      };
      int i = 0;
      while (atomics[i] != "") {
        std::string t = atomics[i];
        i++;
        while (atomics[i] != "") {
          fns[t + atomics[i]] = &TranslateFunction::handleAtomic;
          i++;
        }
        i++;
      }
    }

    if (SL == TranslateModule::SL_OpenCL ||
        SL == TranslateModule::SL_CUDA) {
      const unsigned BARRIER_INVARIANT_MAX_ARITY = 20;
      for(unsigned i = 0; i <= BARRIER_INVARIANT_MAX_ARITY; ++i) {
        std::string S = "__barrier_invariant_";
        llvm::raw_string_ostream SS(S);
        SS << i;
        fns[SS.str()] = &TranslateFunction::handleBarrierInvariant;
      }
      for(unsigned i = 0; i <= BARRIER_INVARIANT_MAX_ARITY; ++i) {
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
  }
  return SpecialFunctionMap;
}

void TranslateFunction::translate() {
  initSpecialFunctionMap(TM->SL);

  if (isGPUEntryPoint || F->getName() == "main")
    BF->setEntryPoint(true);

  if (isGPUEntryPoint)
    BF->addAttribute("kernel");

  if ((TM->SL == TranslateModule::SL_OpenCL || 
       TM->SL == TranslateModule::SL_CUDA)
       && F->getName() == "bugle_barrier")
    BF->addAttribute("barrier");

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

  // See if we can model the return value.
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
  assert(0 && "Unsupported value");
  return 0;
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

void TranslateFunction::addLocToStmt(Stmt *stmt) {
  if(0 != currentSourceLoc.get()) {
    stmt->setSourceLoc(new SourceLoc(*currentSourceLoc));
  }
}

SourceLoc *TranslateFunction::extractSourceLoc(llvm::Instruction *I) {
  SourceLoc* sourceloc = 0;
  if (llvm::MDNode *mdnode = I->getMetadata("dbg")) {
    llvm::DILocation Loc(mdnode);
    sourceloc = new SourceLoc(Loc.getLineNumber(),
                              Loc.getColumnNumber(),
                              Loc.getFilename().str(),
                              Loc.getDirectory().str());
  }
  return sourceloc;
}


ref<Expr> TranslateFunction::handleNoop(bugle::BasicBlock *BBB,
                                        llvm::CallInst *CI,
                                        const std::vector<ref<Expr>> &Args) {
  return 0;
}

ref<Expr> TranslateFunction::handleAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Stmt *assertstmt = new AssertStmt(Expr::createNeZero(Args[0]));
  addLocToStmt(assertstmt);
  BBB->addStmt(assertstmt);
  return 0;
}

ref<Expr> TranslateFunction::handleCandidateAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Stmt *candidateAssertstmt = new AssertStmt(Expr::createNeZero(Args[0]), true);
  addLocToStmt(candidateAssertstmt);
  BBB->addStmt(candidateAssertstmt);
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
  Stmt *assertStmt = new AssertStmt(BoolConstExpr::create(false));
  addLocToStmt(assertStmt);
  BBB->addStmt(assertStmt);
  return 0;
}

ref<Expr> TranslateFunction::handleAssume(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  BBB->addStmt(new AssumeStmt(Expr::createNeZero(Args[0])));
  return 0;
}

ref<Expr> TranslateFunction::handleGlobalAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Stmt *globalAssertStmt = new GlobalAssertStmt(Expr::createNeZero(Args[0]));
  addLocToStmt(globalAssertStmt);
  BBB->addStmt(globalAssertStmt);
  return 0;
}

ref<Expr> TranslateFunction::handleCandidateGlobalAssert(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  Stmt *candidateGlobalAssertStmt = new GlobalAssertStmt(
                                        Expr::createNeZero(Args[0]), true);
  addLocToStmt(candidateGlobalAssertStmt);
  BBB->addStmt(candidateGlobalAssertStmt);
  return 0;
}

ref<Expr> TranslateFunction::handleRequires(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addRequires(Expr::createNeZero(Args[0]), extractSourceLoc(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleGlobalRequires(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addGlobalRequires(Expr::createNeZero(Args[0]), extractSourceLoc(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleEnsures(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addEnsures(Expr::createNeZero(Args[0]), extractSourceLoc(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleGlobalEnsures(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addGlobalEnsures(Expr::createNeZero(Args[0]), extractSourceLoc(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleReadsFrom(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addModifies(AccessHasOccurredExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), false),
        extractSourceLoc(CI));
  BF->addModifies(AccessOffsetExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), false),
        extractSourceLoc(CI));
  return 0;
}

ref<Expr> TranslateFunction::handleWritesTo(bugle::BasicBlock *BBB,
                                           llvm::CallInst *CI,
                                           const std::vector<ref<Expr>> &Args) {
  BF->addModifies(AccessHasOccurredExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), true),
        extractSourceLoc(CI));
  BF->addModifies(AccessOffsetExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange()), true),
        extractSourceLoc(CI));
  BF->addModifies(UnderlyingArrayExpr::create(
        ArrayIdExpr::create(Args[0], TM->defaultRange())),
        extractSourceLoc(CI));
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
  return BoolToBVExpr::create(ImpliesExpr::create(BVToBoolExpr::create(Args[0]), BVToBoolExpr::create(Args[1])));
}

ref<Expr> TranslateFunction::handleEnabled(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  return BoolToBVExpr::create(SpecialVarRefExpr::create(bugle::Type(bugle::Type::Bool), "__enabled"));
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

  if(range.isKind(Type::BV)) {
    if(range.width > 8) {
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

  if(range.isKind(Type::BV)) {
    if(range.width > 8) {
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
  ref<Expr> Ptr = translateValue(CI->getArgOperand(0), BBB),
            PtrArr = ArrayIdExpr::create(Ptr, TM->translateType(CI->getType()) ),
            PtrOfs = ArrayOffsetExpr::create(Ptr);
  Type range = PtrArr->getType().range();
  if (range.width > 8)
    PtrOfs = BVSDivExpr::create(PtrOfs,BVConstExpr::create(TM->TD.getPointerSizeInBits(), range.width/8));

  std::vector<ref<Expr>> args;
  for (unsigned i = 1; i < CI->getNumArgOperands(); i++)
    args.push_back(translateValue(CI->getArgOperand(i),BBB));

  ref<Expr> E = AtomicExpr::create(PtrArr, PtrOfs, args, CI->getCalledFunction()->getName());

  auto S = new EvalStmt(E);
  addLocToStmt(S);
  BBB->addStmt(S);
  return E;
}

ref<Expr> TranslateFunction::handleBarrierInvariant(bugle::BasicBlock *BBB,
                                          llvm::CallInst *CI,
                                          const std::vector<ref<Expr>> &Args) {
  assert (CI->getNumArgOperands() > 1);

  auto BF = BarrierInvariants[CI->getNumArgOperands()];
  if(!BF) {
    std::string S = CI->getCalledFunction()->getName().str();
    llvm::raw_string_ostream SS(S);
    SS << (CI->getNumArgOperands() - 1);
    BF = TM->BM->addFunction(SS.str());
    BarrierInvariants[CI->getNumArgOperands()] = BF;

    int count = 0;
    for (auto i = Args.begin(), e = Args.end(); i != e; ++i, ++count) {
      std::string S;
      llvm::raw_string_ostream SS(S);
      if(count == 0) {
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
  addLocToStmt(CS);
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
  if(!BF) {
    std::string S = CI->getCalledFunction()->getName().str();
    llvm::raw_string_ostream SS(S);
    SS << ((CI->getNumArgOperands() - 1)/2);
    BF = TM->BM->addFunction(SS.str());
    BinaryBarrierInvariants[CI->getNumArgOperands()] = BF;

    int count = 0;
    for (auto i = Args.begin(), e = Args.end(); i != e; ++i, ++count) {
      std::string S;
      llvm::raw_string_ostream SS(S);
      if(count == 0) {
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
  addLocToStmt(CS);
  BBB->addStmt(CS);
  return 0;
}




static std::string mkDimName(const std::string &prefix, ref<Expr> dim) {
  auto CE = dyn_cast<BVConstExpr>(dim);
  switch (CE->getValue().getZExtValue()) {
  case 0: return prefix + "_x";
  case 1: return prefix + "_y";
  case 2: return prefix + "_z";
  default: assert(0 && "Unsupported dimension!"); return 0;
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


void TranslateFunction::addEvalStmt(bugle::BasicBlock *BBB,
                                    llvm::Instruction *I, ref<Expr> E) {
  auto ES = BBB->addEvalStmt(E);
  if (ES)
    addLocToStmt(ES);
}

ref<Expr> TranslateFunction::maybeTranslateSIMDInst(bugle::BasicBlock *BBB,
                             llvm::Type *Ty, llvm::Type *OpTy,
                             ref<Expr> Op,
                          std::function<ref<Expr>(llvm::Type *, ref<Expr>)> F) {
  if (!isa<VectorType>(Ty))
    return F(Ty, Op);

  auto VT = cast<VectorType>(Ty), OpVT = cast<VectorType>(OpTy);
  unsigned NumElems = VT->getNumElements();
  assert(OpVT->getNumElements() == NumElems);
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

  auto VT = cast<VectorType>(Ty), OpVT = cast<VectorType>(OpTy);
  unsigned NumElems = VT->getNumElements();
  assert(OpVT->getNumElements() == NumElems);
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

  if (auto SourceLocI = extractSourceLoc(I)) {
    currentSourceLoc.reset(SourceLocI);
  }

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
      assert(0 && "Unsupported binary operator");
      return;
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
          ref<Expr> ValElem = LoadExpr::create(PtrArr, ElemOfs, LoadsAreTemporal);
          addEvalStmt(BBB, I, ValElem);
          if (LoadElTy.isKind(Type::Pointer))
            ValElem = PtrToBVExpr::create(ValElem);
          ElemsLoaded.push_back(ValElem);
        }
        E = Expr::createBVConcatN(ElemsLoaded);
      } else {
        E = LoadExpr::create(PtrArr, Div, LoadsAreTemporal);
      }
    } else if (ArrRangeTy == Type(Type::BV, 8)) {
      std::vector<ref<Expr> > BytesLoaded;
      for (unsigned i = 0; i != LoadTy.width / 8; ++i) {
        ref<Expr> PtrByteOfs =
          BVAddExpr::create(PtrOfs,
                            BVConstExpr::create(PtrOfs->getType().width, i));
        ref<Expr> ValByte = LoadExpr::create(PtrArr, PtrByteOfs, LoadsAreTemporal);
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
    if (ArrRangeTy == StoreElTy &&
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
          addLocToStmt(SS);
          BBB->addStmt(SS);
        }
      } else {
        StoreStmt* SS = new StoreStmt(PtrArr, Div, Val);
        addLocToStmt(SS);
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
        addLocToStmt(SS);
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
          assert(0 && "Unsupported ptr icmp");
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
          assert(0 && "Unsupported icmp");
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
    Cond = BVToBoolExpr::create(Cond);
    E = IfThenElseExpr::create(Cond, TrueVal, FalseVal);
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
    assert(F && "Only direct calls for now");

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
        assert(CI->getType()->isVoidTy() && "Intrinsic unsupported, can't no-op");
        llvm::errs() << "Warning: intrinsic " << Intrinsic::getName(ID)
                     << " not supported, treating as no-op\n";
        return;
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
          addLocToStmt(CS);
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
  } else {
    assert(0 && "Unsupported instruction");
  }
  if (DumpTranslatedExprs) {
    I->dump();
    E->dump();
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
