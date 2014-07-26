#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/BasicBlock.h"
#include "bugle/Casting.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/IntegerRepresentation.h"
#include "bugle/Module.h"
#include "bugle/SourceLoc.h"
#include "bugle/SourceLocWriter.h"
#include "bugle/Stmt.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace bugle;

void BPLFunctionWriter::maybeWriteCaseSplit(
    llvm::raw_ostream &OS, Expr *PtrArr, const SourceLocsRef &SLocs,
    std::function<void(GlobalArray *, unsigned int)> F) {
  if (isa<NullArrayRefExpr>(PtrArr) ||
      MW->M->global_begin() == MW->M->global_end()) {
    OS << "  assert {:bad_pointer_access} ";
    writeSourceLocs(OS, SLocs);
    OS << "false;\n";
  } else {
    std::set<GlobalArray *> Globals;
    if (!PtrArr->computeArrayCandidates(Globals)) {
      // If we could not compute any candidates, then we take all arrays
      // and the null pointer as candidates.
      Globals.insert(MW->M->global_begin(), MW->M->global_end());
      Globals.insert((bugle::GlobalArray *)0);
    }

    if (Globals.size() == 1 && *Globals.begin() != (bugle::GlobalArray *)0) {
      F(*Globals.begin(), 2);
      OS << "\n";
    } else {
      MW->UsesPointers = true;
      OS << "  ";
      for (auto i = Globals.begin(), e = Globals.end(); i != e; ++i) {
        if (*i == (bugle::GlobalArray *)0)
          continue; // Null pointer; dealt with as last case
        OS << "if (";
        writeExpr(OS, PtrArr);
        OS << " == $arrayId$$" << (*i)->getName() << ") {\n";
        F(*i, 4);
        OS << "\n  } else ";
      }
      OS << "{\n    assert {:bad_pointer_access} ";
      writeSourceLocs(OS, SLocs);
      OS << "false;\n  }\n";
    }
  }
}

void BPLFunctionWriter::writeExpr(llvm::raw_ostream &OS, Expr *E,
                                  unsigned Depth) {
  auto id = SSAVarIds.find(E);
  if (id != SSAVarIds.end()) {
    OS << "v" << id->second;
    return;
  }

  return BPLExprWriter::writeExpr(OS, E, Depth);
}

void BPLFunctionWriter::writeCallStmt(llvm::raw_ostream &OS, CallStmt *CS) {
  OS << "$" << CS->getCallee()->getName() << "(";
  for (auto b = CS->getArgs().begin(), i = b, e = CS->getArgs().end(); i != e;
       ++i) {
    if (i != b)
      OS << ", ";
    writeExpr(OS, i->get());
  }
  OS << ")";
}

void BPLFunctionWriter::writeStmt(llvm::raw_ostream &OS, Stmt *S) {
  if (auto ES = dyn_cast<EvalStmt>(S)) {
    assert(!ES->getExpr()->preventEvalStmt);
    assert(SSAVarIds.find(ES->getExpr().get()) == SSAVarIds.end());
    unsigned id = SSAVarIds.size();
    if (auto ASE = dyn_cast<ArraySnapshotExpr>(ES->getExpr())) {
      auto DstArray = ASE->getDst().get();
      auto SrcArray = ASE->getSrc().get();

      assert(!(isa<NullArrayRefExpr>(DstArray) ||
               isa<NullArrayRefExpr>(SrcArray) ||
               MW->M->global_begin() == MW->M->global_end()));

      std::set<GlobalArray *> GlobalsDst;
      if (!DstArray->computeArrayCandidates(GlobalsDst)) {
        GlobalsDst.insert(MW->M->global_begin(), MW->M->global_end());
      }

      std::set<GlobalArray *> GlobalsSrc;
      if (!SrcArray->computeArrayCandidates(GlobalsSrc)) {
        GlobalsSrc.insert(MW->M->global_begin(), MW->M->global_end());
      }

      if (GlobalsDst.size() == 1 && GlobalsSrc.size() == 1) {
        OS << "  $$" << (*GlobalsDst.begin())->getName() << " := "
           << "$$" << (*GlobalsSrc.begin())->getName() << ";\n";
      } else {
        ErrorReporter::reportImplementationLimitation(
            "Array snapshots on pointers not supported");
      }
      return;
    }
    if (isa<CallExpr>(ES->getExpr())) {
      OS << "  call ";
      writeSourceLocs(OS, ES->getSourceLocs());
    }
    if (isa<AddNoovflExpr>(ES->getExpr())) {
      OS << "  call ";
    }
    if (isa<HavocExpr>(ES->getExpr())) {
      OS << "  havoc v" << id << ";\n";
    } else if (auto CMOE = dyn_cast<CallMemberOfExpr>(ES->getExpr())) {
      auto CES = CMOE->getCallExprs();
      auto SL = ES->getSourceLocs();
      auto F = CMOE->getFunc();
      OS << "  ";
      for (auto i = CES.begin(), e = CES.end(); i != e; ++i) {
        auto CE = cast<CallExpr>(i->get());
        OS << "if (";
        writeExpr(OS, F.get());
        OS << " == $functionId$$" << CE->getCallee()->getName() << ") {\n";
        OS << "    call ";
        writeSourceLocs(OS, SL);
        OS << "v" << id << " := ";
        writeExpr(OS, CE);
        OS << ";\n  } else ";
      }
      OS << "{\n    assert {:bad_pointer_access} ";
      writeSourceLocs(OS, SL);
      OS << "false;\n  }\n";
    } else if (auto LE = dyn_cast<LoadExpr>(ES->getExpr())) {
      maybeWriteCaseSplit(OS, LE->getArray().get(), ES->getSourceLocs(),
                          [&](GlobalArray *GA, unsigned int indent) {
        if (GA->isGlobalOrGroupShared()) {
          writeSourceLocsMarker(OS, ES->getSourceLocs(), indent);
        }
        assert(LE->getType() == GA->getRangeType());
        OS << std::string(indent, ' ');
        OS << "v" << id << " := $$" << GA->getName() << "[";
        writeExpr(OS, LE->getOffset().get());
        OS << "];";
      });
    } else if (auto AE = dyn_cast<AtomicExpr>(ES->getExpr())) {
      maybeWriteCaseSplit(OS, AE->getArray().get(), ES->getSourceLocs(),
                          [&](GlobalArray *GA, unsigned int indent) {
        if (GA->isGlobalOrGroupShared()) {
          writeSourceLocsMarker(OS, ES->getSourceLocs(), indent);
        }
        assert(AE->getType() == GA->getRangeType());
        OS << std::string(indent, ' ');
        OS << "call {:atomic} ";
        OS << "{:atomic_function \"" << AE->getFunction() << "\"} ";
        for (unsigned int i = 0; i < AE->getArgs().size(); i++) {
          OS << "{:arg" << (i + 1) << " ";
          writeExpr(OS, AE->getArgs()[i].get());
          OS << "} ";
        }
        OS << "{:parts " << AE->getParts() << "} ";
        OS << "{:part " << AE->getPart() << "} ";
        OS << "v" << id << ", $$" << GA->getName();
        OS << " := _ATOMIC_OP" << GA->getRangeType().width;
        OS << "($$" << GA->getName() << ", ";
        writeExpr(OS, AE->getOffset().get());
        OS << ");";
      });
    } else if (auto AWGCE = dyn_cast<AsyncWorkGroupCopyExpr>(ES->getExpr())) {
      auto DstArray = AWGCE->getDst().get();
      auto DstOffset = AWGCE->getDstOffset().get();
      auto SrcArray = AWGCE->getSrc().get();
      auto SrcOffset = AWGCE->getSrcOffset().get();

      std::set<GlobalArray *> GlobalsDst;
      if (!DstArray->computeArrayCandidates(GlobalsDst)) {
        GlobalsDst.insert(MW->M->global_begin(), MW->M->global_end());
      }

      std::set<GlobalArray *> GlobalsSrc;
      if (!SrcArray->computeArrayCandidates(GlobalsSrc)) {
        GlobalsSrc.insert(MW->M->global_begin(), MW->M->global_end());
      }

      if (GlobalsDst.size() != 1 || GlobalsSrc.size() != 1) {
        ErrorReporter::reportImplementationLimitation(
            "Async work group copies on pointers not supported");
      }

      auto dst = *GlobalsDst.begin();
      auto src = *GlobalsSrc.begin();
      assert(dst->getRangeType() == src->getRangeType());

      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "procedure {:async_work_group_copy} _ASYNC_WORK_GROUP_COPY_"
           << dst->getRangeType().width
           << "(dstOffset : bv" << DstOffset->getType().width
           << ", src : [bv"
           << MW->M->getPointerWidth() << "]bv" << src->getRangeType().width
           << ", srcOffset : bv" << SrcOffset->getType().width
           << ", size : bv" << MW->M->getPointerWidth()
           << ", handle : bv" << MW->M->getPointerWidth()
           << ") returns (handle' : bv" << MW->M->getPointerWidth()
           << ", dst : [bv" << MW->M->getPointerWidth() << "]bv"
           << dst->getRangeType().width << ")";
      });
      writeSourceLocsMarker(OS, ES->getSourceLocs(), 2);
      OS << "  ";
      OS << "call {:async_work_group_copy} v" << id << ", $$" << dst->getName()
         << " := _ASYNC_WORK_GROUP_COPY_" << dst->getRangeType().width << "(";
      writeExpr(OS, DstOffset);
      OS << ", "
         << "$$" << src->getName() << ", ";
      writeExpr(OS, SrcOffset);
      OS << ", ";
      writeExpr(OS, AWGCE->getSize().get());
      OS << ", ";
      writeExpr(OS, AWGCE->getHandle().get());
      OS << ");\n";
    } else if (auto CE = dyn_cast<BVCtlzExpr>(ES->getExpr())) {
      unsigned Width = CE->getVal()->getType().width;

      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
                           OS << MW->IntRep->getArithmeticBinary(
                               "LSHR", Expr::BVLShr, Width);
                         },
                         false);

      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
                           OS << MW->IntRep->getCtlz(Width);
                         },
                         false);

      OS << "  call v" << id << " := BV" << Width << "_CTLZ(";
      writeExpr(OS, CE->getVal().get());
      OS << ", ";
      writeExpr(OS, CE->getIsZeroUndef().get());
      OS << ");\n";
    } else {
      OS << "  v" << id << " := ";
      writeExpr(OS, ES->getExpr().get());
      OS << ";\n";
    }
    SSAVarIds[ES->getExpr().get()] = id;
  } else if (auto CS = dyn_cast<CallStmt>(S)) {
    OS << "  call ";
    writeSourceLocs(OS, CS->getSourceLocs());
    writeCallStmt(OS, CS);
    OS << ";\n";
  } else if (auto CMOS = dyn_cast<CallMemberOfStmt>(S)) {
    auto CSS = CMOS->getCallStmts();
    auto SL = S->getSourceLocs();
    auto F = CMOS->getFunc();
    OS << "  ";
    for (auto i = CSS.begin(), e = CSS.end(); i != e; ++i) {
      CS = cast<CallStmt>(*i);
      OS << "if (";
      writeExpr(OS, F.get());
      OS << " == $functionId$$" << CS->getCallee()->getName() << ") {\n";
      OS << "    call ";
      writeSourceLocs(OS, SL);
      writeCallStmt(OS, CS);
      OS << ";\n  } else ";
    }
    OS << "{\n    assert {:bad_pointer_access} ";
    writeSourceLocs(OS, SL);
    OS << "false;\n  }\n";
  } else if (auto SS = dyn_cast<StoreStmt>(S)) {
    maybeWriteCaseSplit(OS, SS->getArray().get(), SS->getSourceLocs(),
                        [&](GlobalArray *GA, unsigned int indent) {
      if (GA->isGlobalOrGroupSharedOrConstant()) {
        writeSourceLocsMarker(OS, SS->getSourceLocs(), indent);
      }
      assert(SS->getValue()->getType() == GA->getRangeType());
      OS << std::string(indent, ' ');
      OS << "$$" << GA->getName() << "[";
      writeExpr(OS, SS->getOffset().get());
      OS << "] := ";
      writeExpr(OS, SS->getValue().get());
      OS << ";";
    });
  } else if (auto VAS = dyn_cast<VarAssignStmt>(S)) {
    OS << "  ";
    for (auto b = VAS->getVars().begin(), i = b, e = VAS->getVars().end();
         i != e; ++i) {
      if (i != b)
        OS << ", ";
      OS << "$" << (*i)->getName();
    }
    OS << " := ";
    for (auto b = VAS->getValues().begin(), i = b, e = VAS->getValues().end();
         i != e; ++i) {
      if (i != b)
        OS << ", ";
      writeExpr(OS, i->get());
    }
    OS << ";\n";
  } else if (auto GS = dyn_cast<GotoStmt>(S)) {
    OS << "  goto ";
    for (auto b = GS->getBlocks().begin(), i = b, e = GS->getBlocks().end();
         i != e; ++i) {
      if (i != b)
        OS << ", ";
      OS << "$" << (*i)->getName();
    }
    OS << ";\n";
  } else if (auto AS = dyn_cast<AssumeStmt>(S)) {
    OS << "  assume ";
    if (AS->isPartition())
      OS << "{:partition} ";
    writeExpr(OS, AS->getPredicate().get());
    OS << ";\n";
  } else if (auto AtS = dyn_cast<AssertStmt>(S)) {
    OS << "  assert ";
    if (AtS->isGlobal())
      OS << "{:do_not_predicate} ";
    if (AtS->isCandidate())
      OS << "{:tag \"user\"} ";
    if (AtS->isInvariant())
      OS << "{:originated_from_invariant} ";
    if (AtS->isBadAccess())
      OS << "{:bad_pointer_access} ";
    if (AtS->isBlockSourceLoc())
      OS << "{:block_sourceloc} ";
    writeSourceLocs(OS, AtS->getSourceLocs());
    if (AtS->isCandidate()) {
      unsigned candidateNumber = MW->nextCandidateNumber();
      OS << "_c" << candidateNumber << " ==> ";
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
                           OS << "const {:existential true} _c"
                              << candidateNumber << " : bool";
                         },
                         true);
    }
    writeExpr(OS, AtS->getPredicate().get());
    OS << ";\n";
  } else if (isa<ReturnStmt>(S)) {
    OS << "  return;\n";
  } else if (auto WGES = dyn_cast<WaitGroupEventStmt>(S)) {
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "procedure {:wait_group_events} _WAIT_GROUP_EVENTS(handle : bv"
         << MW->M->getPointerWidth() << ")";
    });
    OS << "  ";
    OS << "call {:wait_group_events} ";
    writeSourceLocs(OS, S->getSourceLocs());
    OS << "_WAIT_GROUP_EVENTS(";
    writeExpr(OS, WGES->getHandle().get());
    OS << ");\n";
  } else {
    llvm_unreachable("Unsupported statement");
  }
}

void BPLFunctionWriter::writeBasicBlock(llvm::raw_ostream &OS, BasicBlock *BB) {
  OS << "$" << BB->getName() << ":\n";
  for (auto i = BB->begin(), e = BB->end(); i != e; ++i)
    writeStmt(OS, *i);
}

void BPLFunctionWriter::writeSourceLocs(llvm::raw_ostream &OS,
                                        const SourceLocsRef &sourcelocs) {
  if (sourcelocs.get() == 0 || sourcelocs->size() == 0)
    return;
  unsigned locnum = MW->SLW->writeSourceLocs(sourcelocs);
  OS << "{:sourceloc_num " << locnum << "}";
  OS << " ";
}

void BPLFunctionWriter::writeSourceLocsMarker(llvm::raw_ostream &OS,
                                              const SourceLocsRef &sourcelocs,
                                              const unsigned int indentLevel) {
  if (sourcelocs.get() == 0 || sourcelocs->size() == 0)
    return;
  OS << std::string(indentLevel, ' ') << "assert {:sourceloc} ";
  writeSourceLocs(OS, sourcelocs);
  OS << "true;\n";
}

void BPLFunctionWriter::writeVar(llvm::raw_ostream &OS, Var *V) {
  OS << "$" << V->getName() << ":";
  MW->writeType(OS, V->getType());
}

void BPLFunctionWriter::write() {
  OS << "procedure ";
  OS << "{:source_name \"" << F->getSourceName() << "\"} ";
  for (auto i = F->attrib_begin(), e = F->attrib_end(); i != e; ++i) {
    OS << "{:" << *i << "} ";
  }

  OS << "$" << F->getName() << "(";
  for (auto b = F->arg_begin(), i = b, e = F->arg_end(); i != e; ++i) {
    if (i != b)
      OS << ", ";
    writeVar(OS, *i);
  }
  OS << ")";

  if (F->return_begin() != F->return_end()) {
    OS << " returns (";
    for (auto b = F->return_begin(), i = b, e = F->return_end(); i != e; ++i) {
      if (i != b)
        OS << ", ";
      writeVar(OS, *i);
    }
    OS << ")";
  }

  if (F->begin() == F->end()) {
    OS << ";\n";
  } else {
    if (F->isSpecification()) {
      OS << ";";
    }
    OS << "\n";

    if (F->isEntryPoint()) {
      OS << MW->getGlobalInitRequires();
    };

    for (auto i = F->requires_begin(), e = F->requires_end(); i != e; ++i) {
      OS << "requires ";
      writeSourceLocs(OS, (*i)->getSourceLocs());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->globalRequires_begin(), e = F->globalRequires_end();
         i != e; ++i) {
      OS << "requires {:do_not_predicate} ";
      writeSourceLocs(OS, (*i)->getSourceLocs());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->procedureWideInvariant_begin(),
              e = F->procedureWideInvariant_end();
         i != e; ++i) {
      OS << "requires {:procedure_wide_invariant} {:do_not_predicate} ";
      writeSourceLocs(OS, (*i)->getSourceLocs());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->ensures_begin(), e = F->ensures_end(); i != e; ++i) {
      OS << "ensures ";
      writeSourceLocs(OS, (*i)->getSourceLocs());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->globalEnsures_begin(), e = F->globalEnsures_end(); i != e;
         ++i) {
      OS << "ensures {:do_not_predicate} ";
      writeSourceLocs(OS, (*i)->getSourceLocs());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->modifies_begin(), e = F->modifies_end(); i != e; ++i) {
      OS << "modifies ";
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    if (F->isSpecification()) {
      OS << "\n";
      return;
    }

    std::string Body;
    llvm::raw_string_ostream BodyOS(Body);
    std::for_each(F->begin(), F->end(),
                  [&](BasicBlock *BB) { writeBasicBlock(BodyOS, BB); });

    OS << "{\n";

    for (auto i = F->local_begin(), e = F->local_end(); i != e; ++i) {
      OS << "  var ";
      writeVar(OS, *i);
      OS << ";\n";
    }

    for (auto i = SSAVarIds.begin(), e = SSAVarIds.end(); i != e; ++i) {
      OS << "  var v" << i->second << ":";
      MW->writeType(OS, i->first->getType());
      OS << ";\n";
    }

    OS << BodyOS.str();
    OS << "}\n";
  }
}
