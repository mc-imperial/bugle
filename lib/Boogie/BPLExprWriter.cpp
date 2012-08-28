#include "bugle/BPLExprWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/Module.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace bugle;

static llvm::cl::opt<bool>
DumpRefCounts("dump-ref-counts", llvm::cl::Hidden,
              llvm::cl::init(false),
              llvm::cl::desc("Dump expression reference counts"));

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

BPLExprWriter::~BPLExprWriter() {}

void BPLExprWriter::writeExpr(llvm::raw_ostream &OS, Expr *E,
                              unsigned Depth) {
  if (DumpRefCounts)
    OS << "/*rc=" << E->refCount << "*/";

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
  } else if (auto PtrE = dyn_cast<PointerExpr>(E)) {
    OS << "MKPTR(";
    writeExpr(OS, PtrE->getArray().get());
    OS << ", ";
    writeExpr(OS, PtrE->getOffset().get());
    OS << ")";
  } else if (auto VarE = dyn_cast<VarRefExpr>(E)) {
    OS << "$" << VarE->getVar()->getName();
  } else if (auto SVarE = dyn_cast<SpecialVarRefExpr>(E)) {
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "const {:" << SVarE->getAttr() << "} " << SVarE->getAttr() << " : ";
      MW->writeType(OS, SVarE->getType());
    });
    OS << SVarE->getAttr();
  } else if (auto ArrE = dyn_cast<GlobalArrayRefExpr>(E)) {
    OS << "$arrayId$$" << ArrE->getArray()->getName();
  } else if (isa<NullArrayRefExpr>(E)) {
    OS << "$arrayId$$null";
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
  } else if (auto IMPLIESE = dyn_cast<ImpliesExpr>(E)) {
	OS << "(";
    writeExpr(OS, IMPLIESE->getLHS().get());
    OS << " ==> ";
    writeExpr(OS, IMPLIESE->getRHS().get());
    OS << ")";
  } else if (auto AHOE = dyn_cast<AccessHasOccurredExpr>(E)) {
    writeAccessLoggingVar(OS, AHOE->getArray().get(), "HAS_OCCURRED", AHOE->getAccessKind(), "false");
  } else if (auto AOE = dyn_cast<AccessOffsetExpr>(E)) {
    writeAccessLoggingVar(OS, AOE->getArray().get(), "OFFSET", AOE->getAccessKind(), "0bv32");
  } else if (auto UnE = dyn_cast<UnaryExpr>(E)) {
    switch (UnE->getKind()) {
    case Expr::BVToFloat:
    case Expr::BVToPtr:
    case Expr::FAbs:
    case Expr::FCos:
    case Expr::FExp:
    case Expr::FFloor:
    case Expr::FloatToBV:
    case Expr::FLog:
    case Expr::FPConv:
    case Expr::FPow:
    case Expr::FPToSI:
    case Expr::FPToUI:
    case Expr::FrexpExp:
    case Expr::FrexpFrac:
    case Expr::FSin:
    case Expr::FSqrt:
    case Expr::OtherInt:
    case Expr::OtherBool:
    case Expr::OtherPtrBase:
    case Expr::PtrToBV:
    case Expr::SIToFP:
    case Expr::UIToFP: {
      std::string IntName; llvm::raw_string_ostream IntS(IntName);
      unsigned FromWidth = UnE->getSubExpr()->getType().width,
               ToWidth = UnE->getType().width;
      switch (UnE->getKind()) {
      case Expr::BVToFloat:
        IntS << "BV" << FromWidth << "_TO_FLOAT";
        break;
      case Expr::BVToPtr:
        IntS << "BV" << FromWidth << "_TO_PTR";
        break;
      case Expr::FAbs:  IntS << "FABS" << ToWidth;  break;
      case Expr::FCos:  IntS << "FCOS" << ToWidth;  break;
      case Expr::FExp:  IntS << "FEXP" << ToWidth;  break;
      case Expr::FFloor:
        IntS << "FFLOOR" << ToWidth;
        break;
      case Expr::FloatToBV:
        IntS << "FLOAT" << FromWidth << "_TO_BV";
        break;
      case Expr::FLog:  IntS << "FLOG" << ToWidth;  break;
      case Expr::FPConv:
        IntS << "FP" << FromWidth << "_CONV" << ToWidth;
        break;
      case Expr::FPow:  IntS << "FPOW" << ToWidth;  break;
      case Expr::FPToSI:
        IntS << "FP" << FromWidth << "_TO_SI" << ToWidth;
        break;
      case Expr::FPToUI:
        IntS << "FP" << FromWidth << "_TO_UI" << ToWidth;
        break;
      case Expr::FrexpExp:
        IntS << "FREXP" << FromWidth << "_EXP";
        break;
      case Expr::FrexpFrac:
        IntS << "FREXP" << FromWidth << "_FRAC" << ToWidth;
        break;
      case Expr::FSin:  IntS << "FSIN" << ToWidth;  break;
      case Expr::FSqrt: IntS << "FSQRT" << ToWidth; break;
      case Expr::OtherInt: IntS << "__other_bv" << ToWidth; break;
      case Expr::OtherBool: IntS << "__other_bool"; break;
      case Expr::OtherPtrBase: IntS << "__other_arrayId"; break;
      case Expr::SIToFP:
        IntS << "SI" << FromWidth << "_TO_FP" << ToWidth;
        break;
      case Expr::UIToFP:
        IntS << "UI" << FromWidth << "_TO_FP" << ToWidth;
        break;
      default: assert(0 && "huh?"); return;
      }
      OS << IntS.str();
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "function " << IntS.str() << "(";
        MW->writeType(OS, UnE->getSubExpr()->getType());
        OS << ") : ";
        MW->writeType(OS, UnE->getType());
      });
      break;
    }
    case Expr::Old: {
      OS << "old";
      break;
    }
    default:
      assert(0 && "Unsupported unary expr");
      break;
    }
    OS << "(";
    writeExpr(OS, UnE->getSubExpr().get());
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
      default: assert(0 && "huh?"); return;
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
      default: assert(0 && "huh?"); return;
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
    case Expr::FDiv:
    case Expr::FPow: {
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::FAdd: IntName = "FADD"; break;
      case Expr::FSub: IntName = "FSUB"; break;
      case Expr::FMul: IntName = "FMUL"; break;
      case Expr::FDiv: IntName = "FDIV"; break;
      case Expr::FPow: IntName = "FPOW"; break;
      default: assert(0 && "huh?"); return;
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
    case Expr::FEq:
    case Expr::FLt:
    case Expr::FUno: {
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::FEq:  IntName = "FEQ";  break;
      case Expr::FLt:  IntName = "FLT";  break;
      case Expr::FUno: IntName = "FUNO"; break;
      default: assert(0 && "huh?"); return;
      }
      OS << IntName << BinE->getLHS()->getType().width;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "function " << IntName << BinE->getLHS()->getType().width << "(";
        MW->writeType(OS, BinE->getLHS()->getType());
        OS << ", ";
        MW->writeType(OS, BinE->getLHS()->getType());
        OS << ") : bool";
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
  } else if (auto LE = dyn_cast<LoadExpr>(E)) {
    // This case is only accessible from the dumper, as we handle LoadExpr
    // specially in BPLFunctionWriter::writeStmt to handle case splitting.
    assert(!MW);
    writeExpr(OS, LE->getArray().get(), 9);
    OS << "[";
    writeExpr(OS, LE->getOffset().get());
    OS << "]";
  } else if (auto MOE = dyn_cast<MemberOfExpr>(E)) {
    // If this is the dumper, show the expression.  Otherwise, this is a no-op.
    if (!MW) {
      OS << "<<member-of";
      for (auto i = MOE->getElems().begin(), e = MOE->getElems().end(); i != e;
           ++i) {
        OS << " " << (*i)->getName();
      }
      OS << ">>(";
    }
    writeExpr(OS, MOE->getSubExpr().get(), Depth);
    if (!MW)
      OS << ")";
  } else {
    assert(0 && "Unsupported expression");
  }
}

void BPLExprWriter::writeAccessLoggingVar(llvm::raw_ostream &OS, 
                                          bugle::Expr* PtrArr, 
                                          std::string accessLoggingVar, 
                                          std::string accessKind, 
                                          std::string unit) {
  if(auto ArrE = dyn_cast<GlobalArrayRefExpr>(PtrArr)) {
    OS << "_" << accessKind << "_" << accessLoggingVar << "_$$" 
       << ArrE->getArray()->getName();
	} else {
    if (MW) {
      OS << "(";
      for (auto i = MW->M->global_begin(), e = MW->M->global_end();
           i != e; ++i) {
        OS << "if (";
        writeExpr(OS, PtrArr);
        OS << " == $arrayId$$" << (*i)->getName() << ") then _" 
           << accessKind << "_" << accessLoggingVar << "_$$" 
           << (*i)->getName() << " else ";
      }
			OS << unit << ")";
    } else {
      OS << "<" << accessLoggingVar << "-case-split>";
    }
  }
}
