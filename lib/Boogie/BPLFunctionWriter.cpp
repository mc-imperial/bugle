#include "bugle/BPLFunctionWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "bugle/BasicBlock.h"
#include "bugle/Casting.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/Module.h"
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

void BPLFunctionWriter::writeExpr(llvm::raw_ostream &OS, Expr *E,
                                  unsigned Depth = 0) {
  auto id = SSAVarIds.find(E);
  if (id != SSAVarIds.end()) {
    OS << "v" << id->second;
    return;
  }

  if (auto CE = dyn_cast<BVConstExpr>(E)) {
    auto &Val = CE->getValue();
    Val.print(OS, /*isSigned=*/false);
    OS << "bv" << Val.getBitWidth();
  } else if (auto BCE = dyn_cast<BoolConstExpr>(E)) {
    OS << (BCE->getValue() ? "true" : "false");
  } else if (auto EE = dyn_cast<BVExtractExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 8);
    writeExpr(OS, EE->getSubExpr().get(), 9);
    OS << "[" << (EE->getOffset() + EE->getType().width) << ":"
       << EE->getOffset() << "]";
  } else if (auto ZEE = dyn_cast<BVZExtExpr>(E)) {
    OS << "BV" << ZEE->getSubExpr()->getType().width
       << "_ZEXT" << ZEE->getType().width << "(";
    writeExpr(OS, ZEE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = ZEE->getSubExpr()->getType().width,
               ToWidth = ZEE->getType().width;
      OS << "function {:bvbuiltin \"zero_extend " << (ToWidth - FromWidth)
         << "\"} BV" << FromWidth << "_ZEXT" << ToWidth << "(bv" << FromWidth
         << ") : bv" << ToWidth;
    });
  } else if (auto SEE = dyn_cast<BVSExtExpr>(E)) {
    OS << "BV" << SEE->getSubExpr()->getType().width
       << "_SEXT" << SEE->getType().width << "(";
    writeExpr(OS, SEE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = SEE->getSubExpr()->getType().width,
               ToWidth = SEE->getType().width;
      OS << "function {:bvbuiltin \"sign_extend " << (ToWidth - FromWidth)
         << "\"} BV" << FromWidth << "_SEXT" << ToWidth << "(bv" << FromWidth
         << ") : bv" << ToWidth;
    });
  } else if (auto FPCE = dyn_cast<FPConvExpr>(E)) {
    OS << "FP" << FPCE->getSubExpr()->getType().width
       << "_CONV" << FPCE->getType().width << "(";
    writeExpr(OS, FPCE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = FPCE->getSubExpr()->getType().width,
               ToWidth = FPCE->getType().width;
      OS << "function FP" << FromWidth << "_CONV" << ToWidth << "(";
      MW->writeType(OS, FPCE->getSubExpr()->getType());
      OS << ") : ";
      MW->writeType(OS, FPCE->getType());
    });
  } else if (auto PtrE = dyn_cast<PointerExpr>(E)) {
    OS << "MKPTR(";
    writeExpr(OS, PtrE->getArray().get());
    OS << ", ";
    writeExpr(OS, PtrE->getOffset().get());
    OS << ")";
  } else if (auto VarE = dyn_cast<VarRefExpr>(E)) {
    OS << "$" << VarE->getVar()->getName();
  } else if (auto ArrE = dyn_cast<GlobalArrayRefExpr>(E)) {
    OS << "$arrayId$" << ArrE->getArray()->getName();
  } else if (auto ConcatE = dyn_cast<BVConcatExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 4);
    writeExpr(OS, ConcatE->getLHS().get(), 4);
    OS << " ++ ";
    writeExpr(OS, ConcatE->getRHS().get(), 5);
  } else if (auto EE = dyn_cast<EqExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 4);
    writeExpr(OS, EE->getLHS().get(), 4);
    OS << " == ";
    writeExpr(OS, EE->getRHS().get(), 4);
  } else if (auto NE = dyn_cast<NeExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 4);
    writeExpr(OS, NE->getLHS().get(), 4);
    OS << " != ";
    writeExpr(OS, NE->getRHS().get(), 4);
  } else if (auto AE = dyn_cast<AndExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 2);
    writeExpr(OS, AE->getLHS().get(), 3);
    OS << " && ";
    writeExpr(OS, AE->getRHS().get(), 3);
  } else if (auto OE = dyn_cast<OrExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 2);
    writeExpr(OS, OE->getLHS().get(), 3);
    OS << " || ";
    writeExpr(OS, OE->getRHS().get(), 3);
  } else if (auto ITEE = dyn_cast<IfThenElseExpr>(E)) {
    OS << "(if ";
    writeExpr(OS, ITEE->getCond().get());
    OS << " then ";
    writeExpr(OS, ITEE->getTrueExpr().get());
    OS << " else ";
    writeExpr(OS, ITEE->getFalseExpr().get());
    OS << ")";
  } else if (auto B2BVE = dyn_cast<BoolToBVExpr>(E)) {
    OS << "(if ";
    writeExpr(OS, B2BVE->getSubExpr().get());
    OS  << " then 1bv1 else 0bv1)";
  } else if (auto BV2BE = dyn_cast<BVToBoolExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 4);
    writeExpr(OS, BV2BE->getSubExpr().get(), 4);
    OS << " == 1bv1";
  } else if (auto F2BVE = dyn_cast<FloatToBVExpr>(E)) {
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "function FLOAT" << F2BVE->getType().width << "_TO_BV(";
      MW->writeType(OS, F2BVE->getSubExpr()->getType());
      OS << ") : bv" << F2BVE->getType().width;
    });
    OS << "FLOAT" << F2BVE->getType().width << "_TO_BV(";
    writeExpr(OS, F2BVE->getSubExpr().get());
    OS << ")";
  } else if (auto BV2FE = dyn_cast<BVToFloatExpr>(E)) {
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "function BV" << BV2FE->getType().width << "_TO_FLOAT(bv"
         << BV2FE->getType().width << ") : ";
      MW->writeType(OS, BV2FE->getType());
    });
    OS << "BV" << BV2FE->getType().width << "_TO_FLOAT(";
    writeExpr(OS, BV2FE->getSubExpr().get());
    OS << ")";
  } else if (auto P2BVE = dyn_cast<PtrToBVExpr>(E)) {
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "function PTR_TO_BV(ptr) : bv" << P2BVE->getType().width;
    });
    OS << "PTR_TO_BV(";
    writeExpr(OS, P2BVE->getSubExpr().get());
    OS << ")";
  } else if (auto BV2PE = dyn_cast<BVToPtrExpr>(E)) {
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "function BV_TO_PTR(bv" << BV2PE->getType().width << ") : ptr";
    });
    OS << "BV_TO_PTR(";
    writeExpr(OS, BV2PE->getSubExpr().get());
    OS << ")";
  } else if (auto AIE = dyn_cast<ArrayIdExpr>(E)) {
    OS << "base#MKPTR(";
    writeExpr(OS, AIE->getSubExpr().get());
    OS << ")";
  } else if (auto AOE = dyn_cast<ArrayOffsetExpr>(E)) {
    OS << "offset#MKPTR(";
    writeExpr(OS, AOE->getSubExpr().get());
    OS << ")";
  } else if (auto NotE = dyn_cast<NotExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 7);
    OS << "!";
    writeExpr(OS, NotE->getSubExpr().get(), 8);
  } else if (auto CE = dyn_cast<CallExpr>(E)) {
    OS << "$" << CE->getCallee()->getName() << "(";
    for (auto b = CE->getArgs().begin(), i = b, e = CE->getArgs().end();
         i != e; ++i) {
      if (i != b)
        OS << ", ";
      writeExpr(OS, i->get());
    }
    OS << ")";
  } else if (auto PLTE = dyn_cast<PtrLtExpr>(E)) {
    OS << "PTR_LT(";
    writeExpr(OS, PLTE->getLHS().get());
    OS << ", ";
    writeExpr(OS, PLTE->getRHS().get());
    OS << ")";
  } else if (auto PLEE = dyn_cast<PtrLeExpr>(E)) {
    OS << "PTR_LE(";
    writeExpr(OS, PLEE->getLHS().get());
    OS << ", ";
    writeExpr(OS, PLEE->getRHS().get());
    OS << ")";
  } else if (auto BinE = dyn_cast<BinaryExpr>(E)) {
    switch (BinE->getKind()) {
    case Expr::BVAdd:
    case Expr::BVSub:
    case Expr::BVMul:
    case Expr::BVSDiv:
    case Expr::BVUDiv:
    case Expr::BVSRem:
    case Expr::BVURem:
    case Expr::BVShl:
    case Expr::BVAShr:
    case Expr::BVLShr:
    case Expr::BVAnd:
    case Expr::BVOr:
    case Expr::BVXor: {
      const char *IntName, *SMTName;
      switch (BinE->getKind()) {
      case Expr::BVAdd:  IntName = "ADD";  SMTName = "bvadd";  break;
      case Expr::BVSub:  IntName = "SUB";  SMTName = "bvsub";  break;
      case Expr::BVMul:  IntName = "MUL";  SMTName = "bvmul";  break;
      case Expr::BVSDiv: IntName = "SDIV"; SMTName = "bvsdiv"; break;
      case Expr::BVUDiv: IntName = "UDIV"; SMTName = "bvudiv"; break;
      case Expr::BVSRem: IntName = "SREM"; SMTName = "bvsrem"; break;
      case Expr::BVURem: IntName = "UREM"; SMTName = "bvurem"; break;
      case Expr::BVShl:  IntName = "SHL";  SMTName = "bvshl";  break;
      case Expr::BVAShr: IntName = "ASHR"; SMTName = "bvashr"; break;
      case Expr::BVLShr: IntName = "LSHR"; SMTName = "bvlshr"; break;
      case Expr::BVAnd:  IntName = "AND";  SMTName = "bvand";  break;
      case Expr::BVOr:   IntName = "OR";   SMTName = "bvor";   break;
      case Expr::BVXor:  IntName = "XOR";  SMTName = "bvxor";  break;
      default: assert(0 && "huh?");
      }
      OS << "BV" << BinE->getType().width << "_" << IntName;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "function {:bvbuiltin \"" << SMTName << "\"} BV"
           << BinE->getType().width
           << "_" << IntName << "(bv" << BinE->getType().width
           << ", bv" << BinE->getType().width
           << ") : bv" << BinE->getType().width;
      });
      break;
    }
    case Expr::BVUgt:
    case Expr::BVUge:
    case Expr::BVUlt:
    case Expr::BVUle:
    case Expr::BVSgt:
    case Expr::BVSge:
    case Expr::BVSlt:
    case Expr::BVSle: {
      const char *IntName, *SMTName;
      switch (BinE->getKind()) {
      case Expr::BVUgt: IntName = "UGT"; SMTName = "bvugt"; break;
      case Expr::BVUge: IntName = "UGE"; SMTName = "bvuge"; break;
      case Expr::BVUlt: IntName = "ULT"; SMTName = "bvult"; break;
      case Expr::BVUle: IntName = "ULE"; SMTName = "bvule"; break;
      case Expr::BVSgt: IntName = "SGT"; SMTName = "bvsgt"; break;
      case Expr::BVSge: IntName = "SGE"; SMTName = "bvsge"; break;
      case Expr::BVSlt: IntName = "SLT"; SMTName = "bvslt"; break;
      case Expr::BVSle: IntName = "SLE"; SMTName = "bvsle"; break;
      default: assert(0 && "huh?");
      }
      OS << "BV" << BinE->getLHS()->getType().width << "_" << IntName;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "function {:bvbuiltin \"" << SMTName << "\"} BV"
           << BinE->getLHS()->getType().width
           << "_" << IntName << "(bv" << BinE->getLHS()->getType().width
           << ", bv" << BinE->getLHS()->getType().width
           << ") : bool";
      });
      break;
    }
    case Expr::FAdd:
    case Expr::FSub:
    case Expr::FMul:
    case Expr::FDiv: {
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::FAdd: IntName = "FADD"; break;
      case Expr::FSub: IntName = "FSUB"; break;
      case Expr::FMul: IntName = "FMUL"; break;
      case Expr::FDiv: IntName = "FDIV"; break;
      default: assert(0 && "huh?");
      }
      OS << IntName << BinE->getType().width;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "function " << IntName << BinE->getType().width << "(";
        MW->writeType(OS, BinE->getType());
        OS << ", ";
        MW->writeType(OS, BinE->getType());
        OS << ") : ";
        MW->writeType(OS, BinE->getType());
      });
      break;
    }
    default:
      assert(0 && "Unsupported binary expr");
      break;
    }
    OS << "(";
    writeExpr(OS, BinE->getLHS().get());
    OS << ", ";
    writeExpr(OS, BinE->getRHS().get());
    OS << ")";
  } else {
    assert(0 && "Unsupported expression");
  }
}

void BPLFunctionWriter::maybeWriteCaseSplit(llvm::raw_ostream &OS,
                                            Expr *PtrArr,
                                         std::function<void(GlobalArray *)> F) {
  if (auto ArrE = dyn_cast<GlobalArrayRefExpr>(PtrArr)) {
    F(ArrE->getArray());
    OS << "\n";
  } else {
    for (auto i = MW->M->global_begin(), e = MW->M->global_end(); i != e;
         ++i) {
      OS << "if (";
      writeExpr(OS, PtrArr);
      OS << " == $arrayId$" << (*i)->getName() << ") {\n    ";
      F(*i);
      OS << "\n  } else ";
    }
    OS << "{\n    assume false;\n  }\n";
  }
}

void BPLFunctionWriter::writeStmt(llvm::raw_ostream &OS, Stmt *S) {
  if (auto ES = dyn_cast<EvalStmt>(S)) {
    auto i = SSAVarIds.find(ES->getExpr().get());
    if (i != SSAVarIds.end())
      return;
    unsigned id = SSAVarIds.size();
    OS << "  ";
    if (isa<CallExpr>(ES->getExpr()))
      OS << "call ";
    if (auto LE = dyn_cast<LoadExpr>(ES->getExpr())) {
      maybeWriteCaseSplit(OS, LE->getArray().get(), [&](GlobalArray *GA) {
        OS << "v" << id << " := $$" << GA->getName() << "[";
        writeExpr(OS, LE->getOffset().get());
        OS << "];";
      });
    } else {
      OS << "v" << id << " := ";
      writeExpr(OS, ES->getExpr().get());
      OS << ";\n";
    }
    SSAVarIds[ES->getExpr().get()] = id;
  } else if (auto CS = dyn_cast<CallStmt>(S)) {
    OS << "  call $" << CS->getCallee()->getName() << "(";
    for (auto b = CS->getArgs().begin(), i = b, e = CS->getArgs().end();
         i != e; ++i) {
      if (i != b)
        OS << ", ";
      writeExpr(OS, i->get());
    }
    OS << ");\n";
  } else if (auto SS = dyn_cast<StoreStmt>(S)) {
    OS << "  ";
    maybeWriteCaseSplit(OS, SS->getArray().get(), [&](GlobalArray *GA) {
      ModifiesSet.insert(GA);
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
    writeExpr(OS, AS->getPredicate().get());
    OS << ";\n";
  } else if (auto AtS = dyn_cast<AssertStmt>(S)) {
    OS << "  assert ";
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

void BPLFunctionWriter::writeVar(llvm::raw_ostream &OS, Var *V) {
  OS << "$" << V->getName() << ":";
  MW->writeType(OS, V->getType());
}

void BPLFunctionWriter::write() {
  OS << "procedure $" << F->getName() << "(";
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
    std::string Body;
    llvm::raw_string_ostream BodyOS(Body);
    std::for_each(F->begin(), F->end(),
                  [&](BasicBlock *BB){ writeBasicBlock(BodyOS, BB); });
    if (!ModifiesSet.empty()) {
      OS << " modifies ";
      for (auto b = ModifiesSet.begin(), i = b, e = ModifiesSet.end(); i != e;
           ++i) {
        if (i != b)
          OS << ", ";
        OS << "$$" << (*i)->getName();
      }
      OS << ";";
    }

    OS << " {\n";

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
