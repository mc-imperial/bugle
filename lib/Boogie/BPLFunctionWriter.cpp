#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "bugle/BasicBlock.h"
#include "bugle/Casting.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/Module.h"
#include "bugle/SourceLoc.h"
#include "bugle/Stmt.h"
#include "llvm/ADT/StringRef.h"

using namespace bugle;

void BPLFunctionWriter::maybeWriteCaseSplit(llvm::raw_ostream &OS,
                                            Expr *PtrArr,
                                            SourceLoc *SLoc,
                                            std::function<void(GlobalArray *, unsigned int)> F) {
  if (isa<NullArrayRefExpr>(PtrArr) ||
      MW->M->global_begin() == MW->M->global_end()) {
    OS << "  assert {:bad_pointer_access} ";
    writeSourceLoc(OS, SLoc);
    OS << "false;\n";
  } else {
    std::set<GlobalArray *> Globals;
    if (!PtrArr->computeArrayCandidates(Globals)) {
      // If we could not compute any candidates, then we take all arrays
      // and the null pointer as candidates.
      Globals.insert(MW->M->global_begin(), MW->M->global_end());
      Globals.insert((bugle::GlobalArray*)0);
    }

    if (Globals.size() == 1) {
      F(*Globals.begin(), 2);
      OS << "\n";
    } else {
      MW->UsesPointers = true;
      OS << "  ";
      for (auto i = Globals.begin(), e = Globals.end(); i != e; ++i) {
        if (*i == (bugle::GlobalArray*)0)
          continue; // Null pointer; dealt with as last case
        OS << "if (";
        writeExpr(OS, PtrArr);
        OS << " == $arrayId$$" << (*i)->getName() << ") {\n";
        F(*i, 4);
        OS << "\n  } else ";
      }
      OS << "{\n    assert {:bad_pointer_access} ";
      writeSourceLoc(OS, SLoc);
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
        OS << "  $$" << (*GlobalsDst.begin())->getName() << " := " <<
          "$$" << (*GlobalsSrc.begin())->getName() << ";\n";
      } else {
        llvm::errs() << "Error: array snapshots on pointers not supported\n";
        std::exit(1);
      }
      return;
    }
    if (isa<CallExpr>(ES->getExpr())) {
      OS << "  call ";
      writeSourceLoc(OS, ES->getSourceLoc());
    }
    if (isa<AddNoovflExpr>(ES->getExpr())) {
      OS << "  call ";
    }
    if (isa<HavocExpr>(ES->getExpr())) {
      OS << "  havoc v" << id << ";\n";
    } else if (auto LE = dyn_cast<LoadExpr>(ES->getExpr())) {
      maybeWriteCaseSplit(OS, LE->getArray().get(), ES->getSourceLoc(),
          [&](GlobalArray *GA, unsigned int indent) {
        if (GA->isGlobalOrGroupShared()) {
          writeSourceLocMarker(OS, ES->getSourceLoc(), indent);
        }
        assert(LE->getType() == GA->getRangeType());
        OS << std::string(indent, ' ');
        OS << "v" << id << " := $$" << GA->getName() << "[";
        writeExpr(OS, LE->getOffset().get());
        OS << "];";
      });
    } else if (auto AE = dyn_cast<AtomicExpr>(ES->getExpr())) {
      maybeWriteCaseSplit(OS, AE->getArray().get(), ES->getSourceLoc(),
          [&](GlobalArray *GA, unsigned int indent) {
        if (GA->isGlobalOrGroupShared()) {
          writeSourceLocMarker(OS, ES->getSourceLoc(), indent);
        }
        assert(AE->getType() == GA->getRangeType());
        OS << std::string(indent, ' ');
        OS << "call {:atomic} ";
        OS << "{:atomic_function \"" << AE->getFunction() << "\"}";
        for (unsigned int i = 0; i < AE->getArgs().size(); i++) {
          OS << "{:arg" << (i+1) << " ";
          writeExpr(OS,AE->getArgs()[i].get());
          OS << "}";
        }
        OS << "{:parts " << AE->getParts() << "}";
        OS << "{:part " << AE->getPart() << "} ";
        OS << "v" << id << " := _ATOMIC_OP" << GA->getRangeType().width;
        OS << "($$" << GA->getName() << ", ";
        writeExpr(OS, AE->getOffset().get());
        OS << ");";
      });
    } else {
      OS << "  v" << id << " := ";
      writeExpr(OS, ES->getExpr().get());
      OS << ";\n";
    }
    SSAVarIds[ES->getExpr().get()] = id;
  } else if (auto CS = dyn_cast<CallStmt>(S)) {
    OS << "  call ";
    writeSourceLoc(OS, CS->getSourceLoc());
    OS << "$" << CS->getCallee()->getName() << "(";
    for (auto b = CS->getArgs().begin(), i = b, e = CS->getArgs().end();
         i != e; ++i) {
      if (i != b)
        OS << ", ";
      writeExpr(OS, i->get());
    }
    OS << ");\n";
  } else if (auto SS = dyn_cast<StoreStmt>(S)) {
    maybeWriteCaseSplit(OS, SS->getArray().get(), SS->getSourceLoc(),
        [&](GlobalArray *GA, unsigned int indent) {
      if(GA->isGlobalOrGroupSharedOrConstant()) {
        writeSourceLocMarker(OS, SS->getSourceLoc(), indent);
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
    if(AtS->isCandidate()) {
      OS << "{:tag \"user\"} ";
    }
    writeSourceLoc(OS, AtS->getSourceLoc());
    if(AtS->isCandidate()) {
      unsigned candidateNumber = MW->nextCandidateNumber();
      OS << "_c" << candidateNumber << " ==> ";
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "const {:existential true} _c" << candidateNumber << " : bool";
      }, true);
    }
    writeExpr(OS, AtS->getPredicate().get());
    OS << ";\n";
  } else if (auto AtS = dyn_cast<GlobalAssertStmt>(S)) {
    OS << "  assert {:do_not_predicate} ";
    if(AtS->isCandidate()) {
      OS << "{:tag \"user\"} ";
    }
    writeSourceLoc(OS, AtS->getSourceLoc());
    if(AtS->isCandidate()) {
      unsigned candidateNumber = MW->nextCandidateNumber();
      OS << "_c" << candidateNumber << " ==> ";
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "const {:existential true} _c" << candidateNumber << " : bool";
      }, true);
    }
    writeExpr(OS, AtS->getPredicate().get());
    OS << ";\n";
  } else if (isa<ReturnStmt>(S)) {
    OS << "  return;\n";
  } else {
    assert(0 && "Unsupported statement");
  }
}

void BPLFunctionWriter::writeBasicBlock(llvm::raw_ostream &OS, BasicBlock *BB) {
  OS << "$" << BB->getName() << ":\n";
  for (auto i = BB->begin(), e = BB->end(); i != e; ++i)
    writeStmt(OS, *i);
}

void BPLFunctionWriter::writeSourceLoc(llvm::raw_ostream &OS,
                                       const SourceLoc *sourceloc) {
  if (sourceloc != NULL) {
    OS << "{:line " << sourceloc->getLineNo() << "}";
    OS << "{:col " << sourceloc->getColNo() << "}";
    OS << "{:fname \"" << sourceloc->getFileName() << "\"}";
    OS << "{:dir \"" << sourceloc->getPath() << "\"}";
    OS << " ";
  }
}

void BPLFunctionWriter::writeSourceLocMarker(llvm::raw_ostream &OS,
                                             const SourceLoc *sourceloc,
                                             unsigned int indent) {
  if (sourceloc != NULL) {
    OS << std::string(indent, ' ') << "assert {:sourceloc}";
    writeSourceLoc(OS, sourceloc);
    OS << "true;\n";
  }
}

void BPLFunctionWriter::writeVar(llvm::raw_ostream &OS, Var *V) {
  OS << "$" << V->getName() << ":";
  MW->writeType(OS, V->getType());
}

static bool isSpecificationFunction(llvm::StringRef Name) {
  return Name.startswith("__spec");
}

void BPLFunctionWriter::write() {
  OS << "procedure ";
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
    if(isSpecificationFunction(F->getName())) {
      OS << ";";
    }
    OS << "\n";

    if (F->isEntryPoint()) {
      OS << MW->getGlobalInitRequires();
    };

    for (auto i = F->requires_begin(), e = F->requires_end(); i != e; ++i) {
      OS << "requires ";
      writeSourceLoc(OS, (*i)->getSourceLoc());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->globalRequires_begin(), e = F->globalRequires_end(); i != e; ++i) {
      OS << "requires {:do_not_predicate} ";
      writeSourceLoc(OS, (*i)->getSourceLoc());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->ensures_begin(), e = F->ensures_end(); i != e; ++i) {
      OS << "ensures ";
      writeSourceLoc(OS, (*i)->getSourceLoc());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->globalEnsures_begin(), e = F->globalEnsures_end(); i != e; ++i) {
      OS << "ensures {:do_not_predicate} ";
      writeSourceLoc(OS, (*i)->getSourceLoc());
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    for (auto i = F->modifies_begin(), e = F->modifies_end(); i != e; ++i) {
      OS << "modifies ";
      writeExpr(OS, (*i)->getExpr().get());
      OS << ";\n";
    }

    if (isSpecificationFunction(F->getName())) {
      OS << "\n";
      return;
    }

    std::string Body;
    llvm::raw_string_ostream BodyOS(Body);
    std::for_each(F->begin(), F->end(),
                  [&](BasicBlock *BB){ writeBasicBlock(BodyOS, BB); });

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
