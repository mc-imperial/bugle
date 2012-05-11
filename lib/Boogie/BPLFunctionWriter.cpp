#include "bugle/BPLFunctionWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "bugle/BasicBlock.h"
#include "bugle/Casting.h"
#include "bugle/Expr.h"
#include "bugle/Stmt.h"

using namespace bugle;

void BPLFunctionWriter::writeExpr(Expr *E) {
  auto id = SSAVarIds.find(E);
  if (id != SSAVarIds.end()) {
    OS << "v" << id->second;
    return;
  }

  if (auto CE = dyn_cast<BVConstExpr>(E)) {
    auto &Val = CE->getValue();
    Val.print(OS, /*isSigned=*/false);
    OS << "bv" << Val.getBitWidth();
  } else if (auto AddE = dyn_cast<BVAddExpr>(E)) {
    OS << "BV" << AddE->getType().width << "_ADD(";
    writeExpr(AddE->getLHS().get());
    OS << ", ";
    writeExpr(AddE->getRHS().get());
    OS << ")";
  } else if (auto PtrE = dyn_cast<PointerExpr>(E)) {
    OS << "POINTER(";
    writeExpr(PtrE->getArray().get());
    OS << ", ";
    writeExpr(PtrE->getOffset().get());
    OS << ")";
  } else if (auto ArrE = dyn_cast<ArrayRefExpr>(E)) {
    OS << "ARRAY(" << ArrE->getArray() << ")";
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
    ref<Expr> PtrArr = ArrayIdExpr::create(SS->getPointer());
    if (auto ArrE = dyn_cast<ArrayRefExpr>(PtrArr)) {
      OS << "  " << ArrE->getArray() << "[";
      writeExpr(ArrayOffsetExpr::create(SS->getPointer()).get());
      OS << "] := ";
      writeExpr(SS->getValue().get());
      OS << ";\n";
    } else {
      assert(0 && "TODO case split");
    }
  } else if (auto RS = dyn_cast<ReturnStmt>(S)) {
    if (auto E = RS->getValue().get()) {
      OS << "  ret := ";
      writeExpr(E);
      OS << ";\n";
    }
    OS << "  return;\n";
  } else {
    assert(0 && "Unsupported statement");
  }
}

void BPLFunctionWriter::writeBasicBlock(BasicBlock *BB) {
  for (auto i = BB->begin(), e = BB->end(); i != e; ++i)
    writeStmt(*i);
}
