#include "bugle/BPLExprWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/Module.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "llvm/Support/raw_ostream.h"

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

BPLExprWriter::~BPLExprWriter() {}

void BPLExprWriter::writeExpr(llvm::raw_ostream &OS, Expr *E,
                              unsigned Depth) {
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
  } else if (auto FPSIE = dyn_cast<FPToSIExpr>(E)) {
    OS << "FP" << FPSIE->getSubExpr()->getType().width
       << "_TO_SI" << FPSIE->getType().width << "(";
    writeExpr(OS, FPSIE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = FPSIE->getSubExpr()->getType().width,
               ToWidth = FPSIE->getType().width;
      OS << "function FP" << FromWidth << "_TO_SI" << ToWidth << "(";
      MW->writeType(OS, FPSIE->getSubExpr()->getType());
      OS << ") : bv" << ToWidth;
    });
  } else if (auto FPUIE = dyn_cast<FPToUIExpr>(E)) {
    OS << "FP" << FPUIE->getSubExpr()->getType().width
       << "_TO_UI" << FPUIE->getType().width << "(";
    writeExpr(OS, FPUIE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = FPUIE->getSubExpr()->getType().width,
               ToWidth = FPUIE->getType().width;
      OS << "function FP" << FromWidth << "_TO_UI" << ToWidth << "(";
      MW->writeType(OS, FPUIE->getSubExpr()->getType());
      OS << ") : bv" << ToWidth;
    });
  } else if (auto SIFPE = dyn_cast<SIToFPExpr>(E)) {
    OS << "SI" << SIFPE->getSubExpr()->getType().width
       << "_TO_FP" << SIFPE->getType().width << "(";
    writeExpr(OS, SIFPE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = SIFPE->getSubExpr()->getType().width,
               ToWidth = SIFPE->getType().width;
      OS << "function SI" << FromWidth << "_TO_FP" << ToWidth << "(bv"
         << FromWidth << ") : ";
      MW->writeType(OS, SIFPE->getType());
    });
  } else if (auto UIFPE = dyn_cast<UIToFPExpr>(E)) {
    OS << "UI" << UIFPE->getSubExpr()->getType().width
       << "_TO_FP" << UIFPE->getType().width << "(";
    writeExpr(OS, UIFPE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = UIFPE->getSubExpr()->getType().width,
               ToWidth = UIFPE->getType().width;
      OS << "function UI" << FromWidth << "_TO_FP" << ToWidth << "(bv"
         << FromWidth << ") : ";
      MW->writeType(OS, UIFPE->getType());
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
    OS << "$arrayId$" << ArrE->getArray()->getName();
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
  } else if (auto IMPLIESE = dyn_cast<ImpliesExpr>(E)) {
	OS << "(";
    writeExpr(OS, IMPLIESE->getLHS().get());
    OS << " ==> ";
    writeExpr(OS, IMPLIESE->getRHS().get());
    OS << ")";
  } else if (auto AHOE = dyn_cast<AccessHasOccurredExpr>(E)) {
    writeAccessLoggingVar(OS, AHOE->getArray().get(), "HAS_OCCURRED", AHOE->getAccessKind());
  } else if (auto AOE = dyn_cast<AccessOffsetExpr>(E)) {
    writeAccessLoggingVar(OS, AOE->getArray().get(), "OFFSET", AOE->getAccessKind());
  } else if (auto UnE = dyn_cast<UnaryExpr>(E)) {
    switch (UnE->getKind()) {
    case Expr::FAbs:
    case Expr::FCos:
    case Expr::FExp:
    case Expr::FLog:
    case Expr::FPow:
    case Expr::FSin:
    case Expr::FSqrt: {
      const char *IntName;
      switch (UnE->getKind()) {
      case Expr::FAbs:  IntName = "FABS";  break;
      case Expr::FCos:  IntName = "FCOS";  break;
      case Expr::FExp:  IntName = "FEXP";  break;
      case Expr::FLog:  IntName = "FLOG";  break;
      case Expr::FPow:  IntName = "FPOW";  break;
      case Expr::FSin:  IntName = "FSIN";  break;
      case Expr::FSqrt: IntName = "FSQRT"; break;
      default: assert(0 && "huh?");
      }
      OS << IntName << UnE->getType().width;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "function " << IntName << UnE->getType().width << "(";
        MW->writeType(OS, UnE->getSubExpr()->getType());
        OS << ") : ";
        MW->writeType(OS, UnE->getType());
      });
      break;
    }
    case Expr::OtherInt: {
      OS << "__other_bv32";
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
       OS << "function __other_bv32(e : bv32) : bv32";
      });
      break;
	}
    case Expr::OtherBool: {
      OS << "__other_bool";
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
       OS << "function __other_bool(e : bool) : bool";
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
    case Expr::FDiv:
    case Expr::FPow: {
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::FAdd: IntName = "FADD"; break;
      case Expr::FSub: IntName = "FSUB"; break;
      case Expr::FMul: IntName = "FMUL"; break;
      case Expr::FDiv: IntName = "FDIV"; break;
      case Expr::FPow: IntName = "FPOW"; break;
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
    case Expr::FEq:
      assert(BinE->getKind() == Expr::FEq);
      writeExpr(OS, BinE->getLHS().get());
      OS << " == "; // TODO: deal with NaN
      writeExpr(OS, BinE->getRHS().get());
	  return; // This case does not conform to the pattern for other binaries, so we return
	case Expr::FLt:
    case Expr::FUno: {
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::FLt:  IntName = "FLT";  break;
      case Expr::FUno: IntName = "FUNO"; break;
      default: assert(0 && "huh?");
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
  } else {
    assert(0 && "Unsupported expression");
  }
}

void BPLExprWriter::writeAccessLoggingVar(llvm::raw_ostream &OS, bugle::Expr* PtrArr, std::string accessLoggingVar, std::string accessKind) {
    if(auto ArrE = dyn_cast<GlobalArrayRefExpr>(PtrArr)) {
      OS << "_" << accessKind << "_" << accessLoggingVar << "_$$" << ArrE->getArray()->getName();
	} else {
          if (MW) {
            for (auto i = MW->M->global_begin(), e = MW->M->global_end(); i != e;
                ++i) {
              if(i != MW->M->global_begin()) {
                OS << " && ";
              }
              OS << "(";
              writeExpr(OS, PtrArr);
              OS << " == $arrayId$" << (*i)->getName() << ") ==> _" << accessKind << 
				  "_" << accessLoggingVar << "_$$" << (*i)->getName();
            }
          } else {
            OS << "<" << accessLoggingVar << "-case-split>";
          }
	}
}
