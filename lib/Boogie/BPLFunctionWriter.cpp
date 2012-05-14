#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "bugle/BasicBlock.h"
#include "bugle/Casting.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/Stmt.h"

using namespace bugle;

namespace {

struct ScopedParenPrinter {
  llvm::raw_ostream &OS;
  bool ParenRequired;
  ScopedParenPrinter(llvm::raw_ostream &OS, unsigned Depth, unsigned RuleDepth)
    : OS(OS), ParenRequired(RuleDepth < Depth) {
    if (ParenRequired)
      OS << "(";
  }
  ~ScopedParenPrinter() {
    if (ParenRequired)
      OS << ")";
  }
};

}

void BPLFunctionWriter::writeExpr(Expr *E, unsigned Depth = 0) {
  auto id = SSAVarIds.find(E);
  if (id != SSAVarIds.end()) {
    OS << "v" << id->second;
    return;
  }

  if (auto CE = dyn_cast<BVConstExpr>(E)) {
    auto &Val = CE->getValue();
    Val.print(OS, /*isSigned=*/false);
    OS << "bv" << Val.getBitWidth();
  } else if (auto EE = dyn_cast<BVExtractExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 8);
    writeExpr(EE->getSubExpr().get(), 9);
    OS << "[" << (EE->getOffset() + EE->getType().width) << ":"
       << EE->getOffset() << "]";
  } else if (auto AddE = dyn_cast<BVAddExpr>(E)) {
    OS << "BV" << AddE->getType().width << "_ADD(";
    writeExpr(AddE->getLHS().get());
    OS << ", ";
    writeExpr(AddE->getRHS().get());
    OS << ")";
  } else if (auto LE = dyn_cast<LoadExpr>(E)) {
    ref<Expr> PtrArr = LE->getArray();
    if (auto ArrE = dyn_cast<GlobalArrayRefExpr>(PtrArr)) {
      ScopedParenPrinter X(OS, Depth, 8);
      OS << ArrE->getArray()->getName() << "[";
      writeExpr(LE->getOffset().get(), 9);
      OS << "]";
    } else {
      assert(0 && "TODO case split");
    }
  } else if (auto PtrE = dyn_cast<PointerExpr>(E)) {
    OS << "POINTER(";
    writeExpr(PtrE->getArray().get());
    OS << ", ";
    writeExpr(PtrE->getOffset().get());
    OS << ")";
  } else if (auto VarE = dyn_cast<VarRefExpr>(E)) {
    OS << VarE->getVar()->getName();
  } else if (auto ArrE = dyn_cast<GlobalArrayRefExpr>(E)) {
    OS << "ARRAY(" << ArrE->getArray()->getName() << ")";
  } else if (auto ConcatE = dyn_cast<BVConcatExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 4);
    writeExpr(ConcatE->getLHS().get(), 4);
    OS << " ++ ";
    writeExpr(ConcatE->getRHS().get(), 5);
  } else {
    assert(0 && "Unsupported expression");
  }
}

void BPLFunctionWriter::writeStmt(Stmt *S) {
  if (auto ES = dyn_cast<EvalStmt>(S)) {
    unsigned id = SSAVarIds.size();
    OS << "  v" << id << " := ";
    writeExpr(ES->getExpr().get());
    OS << ";\n";
    SSAVarIds[ES->getExpr().get()] = id;
  } else if (auto SS = dyn_cast<StoreStmt>(S)) {
    ref<Expr> PtrArr = SS->getArray();
    if (auto ArrE = dyn_cast<GlobalArrayRefExpr>(PtrArr)) {
      OS << "  " << ArrE->getArray()->getName() << "[";
      writeExpr(SS->getOffset().get());
      OS << "] := ";
      writeExpr(SS->getValue().get());
      OS << ";\n";
    } else {
      assert(0 && "TODO case split");
    }
  } else if (auto VAS = dyn_cast<VarAssignStmt>(S)) {
    OS << "  " << VAS->getVar()->getName() << " := ";
    writeExpr(VAS->getValue().get());
    OS << ";\n";
  } else if (isa<ReturnStmt>(S)) {
    OS << "  return;\n";
  } else {
    assert(0 && "Unsupported statement");
  }
}

void BPLFunctionWriter::writeBasicBlock(BasicBlock *BB) {
  OS << BB->getName() << ":\n";
  for (auto i = BB->begin(), e = BB->end(); i != e; ++i)
    writeStmt(*i);
}

void BPLFunctionWriter::writeVar(Var *V) {
  OS << V->getName() << ":";
  MW->writeType(OS, V->getType());
}

void BPLFunctionWriter::write() {
  OS << "procedure " << F->getName() << "(";
  for (auto b = F->arg_begin(), i = b, e = F->arg_end(); i != e; ++i) {
    if (i != b)
      OS << ", ";
    writeVar(*i);
  }
  OS << ")";

  if (F->return_begin() != F->return_end()) {
    OS << " returns (";
    for (auto b = F->return_begin(), i = b, e = F->return_end(); i != e; ++i) {
      if (i != b)
        OS << ", ";
      writeVar(*i);
    }
    OS << ")";
  }

  if (F->begin() == F->end()) {
    OS << ";\n";
  } else {
    OS << " {\n";
    std::for_each(F->begin(), F->end(),
                  [this](BasicBlock *BB){ writeBasicBlock(BB); });
    OS << "}\n";
  }
}
