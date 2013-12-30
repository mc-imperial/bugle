#include "bugle/BPLExprWriter.h"
#include "bugle/BPLModuleWriter.h"
#include "bugle/IntegerRepresentation.h"
#include "bugle/Module.h"
#include "bugle/RaceInstrumenter.h"
#include "bugle/Expr.h"
#include "bugle/Function.h"
#include "bugle/GlobalArray.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <sstream>

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
    MW->IntRep->printVal(OS, Val);
  } else if (auto BCE = dyn_cast<BoolConstExpr>(E)) {
    OS << (BCE->getValue() ? "true" : "false");
  } else if (auto EE = dyn_cast<BVExtractExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 8);
    std::string s; llvm::raw_string_ostream ss(s);
    writeExpr(ss, EE->getSubExpr().get(), 9);
    OS << MW->IntRep->getExtractExpr(ss.str(), EE->getOffset() + EE->getType().width, EE->getOffset());
    if (MW->IntRep->abstractsExtract()) {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getExtract();
      }, false);
    }
  } else if (auto ZEE = dyn_cast<BVZExtExpr>(E)) {
    OS << "BV" << ZEE->getSubExpr()->getType().width
       << "_ZEXT" << ZEE->getType().width << "(";
    writeExpr(OS, ZEE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = ZEE->getSubExpr()->getType().width,
               ToWidth = ZEE->getType().width;
      OS << MW->IntRep->getZeroExtend(FromWidth, ToWidth);
    }, false);
  } else if (auto SEE = dyn_cast<BVSExtExpr>(E)) {
    OS << "BV" << SEE->getSubExpr()->getType().width
       << "_SEXT" << SEE->getType().width << "(";
    writeExpr(OS, SEE->getSubExpr().get());
    OS << ")";
    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      unsigned FromWidth = SEE->getSubExpr()->getType().width,
               ToWidth = SEE->getType().width;
      OS << MW->IntRep->getSignExtend(FromWidth, ToWidth);
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
    MW->UsesPointers = true;
    OS << "$arrayId$$" << ArrE->getArray()->getName();
  } else if (isa<NullArrayRefExpr>(E)) {
    MW->UsesPointers = true;
    OS << "$arrayId$$null$";
  } else if (auto ConcatE = dyn_cast<BVConcatExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 4);
    std::string lhsS; llvm::raw_string_ostream lhsSS(lhsS);
    std::string rhsS; llvm::raw_string_ostream rhsSS(rhsS);
    writeExpr(lhsSS, ConcatE->getLHS().get(), 4);
    writeExpr(rhsSS, ConcatE->getRHS().get(), 5);
    OS << MW->IntRep->getConcatExpr(lhsSS.str(), rhsSS.str());
    if (MW->IntRep->abstractsConcat()) {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getConcat();
      }, false);
    }
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
  } else if (isa<HavocExpr>(E)) {
    llvm_unreachable("Handled at statement level");
  } else if (auto B2BVE = dyn_cast<BoolToBVExpr>(E)) {
    OS << "(if ";
    writeExpr(OS, B2BVE->getSubExpr().get());
    OS  << " then " << MW->IntRep->getLiteral(1, 1) << " else " <<
      MW->IntRep->getLiteral(0, 1) << ")";
  } else if (auto BV2BE = dyn_cast<BVToBoolExpr>(E)) {
    ScopedParenPrinter X(OS, Depth, 4);
    writeExpr(OS, BV2BE->getSubExpr().get(), 4);
    OS << " == " << MW->IntRep->getLiteral(1, 1);
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
  } else if (auto ANOVE = dyn_cast<AddNoovflExpr>(E)) {
    unsigned width = ANOVE->getFirst()->getType().width;
    OS << "$__add_noovfl_" << (ANOVE->getIsSigned() ? "signed" : "unsigned")
      << "_" << width << "(";
    writeExpr(OS, ANOVE->getFirst().get());
    OS << ", ";
    writeExpr(OS, ANOVE->getSecond().get());
    OS << ")";

    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << MW->IntRep->getArithmeticBinary("ADD", bugle::Expr::Kind::BVAdd, width);
    }, false);

    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << MW->IntRep->getArithmeticBinary("ADD", bugle::Expr::Kind::BVAdd, width + 1);
    }, false);

    if (MW->IntRep->abstractsConcat()) {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getConcat();
      }, false);
    }

    if (MW->IntRep->abstractsExtract()) {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getExtract();
      }, false);
    }

    if (ANOVE->getIsSigned()) {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "procedure {:inline 1} $__add_noovfl_signed_"
           << width << "(x : " << MW->IntRep->getType(width) << ", y : " 
           << MW->IntRep->getType(width)
           << ") returns (z : " << MW->IntRep->getType(width) << ") {\n"
           << "  assume ";

        {
          std::stringstream ss;
          ss << "BV" << (width + 1) << "_ADD("
                      << MW->IntRep->getConcatExpr(MW->IntRep->getLiteral(0, 1), "x") << ", "
                      << MW->IntRep->getConcatExpr(MW->IntRep->getLiteral(0, 1), "y") << ")";

          OS << MW->IntRep->getExtractExpr(ss.str(), width + 1, width);
        }

        OS << " == " << MW->IntRep->getLiteral(0, 1) << ";\n"
           << "  assume " << MW->IntRep->getExtractExpr("x", width, width - 1) << " == "
           << MW->IntRep->getExtractExpr("y", width, width - 1) << " ==> ";

        {
          std::stringstream ss;
          ss << "BV" << width << "_ADD(x, y)";
          OS << MW->IntRep->getExtractExpr(ss.str(), width, width - 1);
        }

        OS << " == " << MW->IntRep->getExtractExpr("x", width, width - 1) << ";\n"
           << "  z := BV" << width << "_ADD(x, y);\n"
           << "}";
      }, false);
    } else {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        std::stringstream ss;
        ss << "BV" << (width + 1) << "_ADD(" 
           << MW->IntRep->getConcatExpr(MW->IntRep->getLiteral(0, 1), "x")
           << ", " << MW->IntRep->getConcatExpr(MW->IntRep->getLiteral(0, 1), "y") << ")";
        OS << "procedure {:inline 1} $__add_noovfl_unsigned_"
           << width << "(x : " << MW->IntRep->getType(width) 
           << ", y : " << MW->IntRep->getType(width)
           << ") returns (z : " << MW->IntRep->getType(width) << ") {\n"
           << "  assume "
           << MW->IntRep->getExtractExpr(
            ss.str(),
            width + 1, width) + " == " << MW->IntRep->getLiteral(0, 1) << ";\n"
           << "  z := BV" << width << "_ADD(x, y);\n"
           << "}";
      }, false);
    }
  } else if (auto ANOVPE = dyn_cast<AddNoovflPredicateExpr>(E)) {
    auto exprs = ANOVPE->getExprs();
    unsigned n = exprs.size();
    unsigned width = exprs[0]->getType().width;
    OS << "__add_noovfl_" << n << "(";
    for (auto b = exprs.begin(), i = b, e = exprs.end(); i != e; ++i) {
      OS << (i != b ? ", " : "");
      writeExpr(OS, i->get());
    }
    OS << ")";

    unsigned b = (unsigned)std::ceil(std::log((float)n) / std::log(2.0));
    std::stringstream ss;
    ss << MW->IntRep->getConcatExpr(MW->IntRep->getLiteral(0, b), "v0");
    std::string lhs = ss.str();
    for (unsigned i=1; i<n; ++i) {
      std::stringstream ss;
      std::stringstream vi;
      vi << "v" << i;
      ss << "BV" << (width+b) << "_ADD("
         << lhs
         << ", " << MW->IntRep->getConcatExpr(MW->IntRep->getLiteral(0, b),
            vi.str()) << ")";
      lhs = ss.str();
    }

    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << MW->IntRep->getArithmeticBinary("ADD", bugle::Expr::Kind::BVAdd, width + b); 
    }, false);

    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "function {:inline true} __add_noovfl_" << n << "(";
      for (unsigned i=0; i<n; ++i) {
        OS << (i > 0 ? ", " : "") << "v" << i << ":" << MW->IntRep->getType(width);
      }
      OS << ") : " << MW->IntRep->getType(1) << " {";
      if (n == 1) {
        OS << MW->IntRep->getLiteral(1, 1);
      } else {
        OS << "if " << MW->IntRep->getExtractExpr(lhs, width+b, width)
           << " == " << MW->IntRep->getLiteral(0, b)
           << " then " << MW->IntRep->getLiteral(1, 1)
           << " else " << MW->IntRep->getLiteral(0, 1);
      }
      OS << "}";
    }, false);

    if (MW->IntRep->abstractsConcat()) {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getConcat();
      }, false);
    }

    if (MW->IntRep->abstractsExtract()) {
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getExtract();
      }, false);
    }

  } else if (auto UFE = dyn_cast<UninterpretedFunctionExpr>(E)) {
    OS << UFE->getName() << "(";
    for (unsigned i = 0; i < UFE->getNumOperands(); ++i) {
      if (i > 0) {
        OS << ", ";
      }
      writeExpr(OS, UFE->getOperand(i).get());
    }
    OS << ")";

    MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
      OS << "function " << UFE->getName() << "(";
      for (unsigned i = 0; i < UFE->getNumOperands(); ++i) {
        if (i > 0) {
          OS << ", ";
        }
        MW->writeType(OS, UFE->getOperand(i)->getType());
      }
      OS << ") : ";
      MW->writeType(OS, UFE->getType());
    });
  } else if (auto AHTVE = dyn_cast<AtomicHasTakenValueExpr>(E)) {
      auto Array = AHTVE->getArray().get();
      assert(!(isa<NullArrayRefExpr>(Array) || 
        MW->M->global_begin() == MW->M->global_end()));

      std::set<GlobalArray *> Globals;
      if (!Array->computeArrayCandidates(Globals)) {
        Globals.insert(MW->M->global_begin(), MW->M->global_end());
      }

      if (Globals.size() == 1) {
        OS << "_USED_$$" << (*Globals.begin())->getName()
           << "[";
        writeExpr(OS, AHTVE->getOffset().get());
        OS << "][";
        writeExpr(OS, AHTVE->getValue().get());
        OS << "]";
        MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
          OS << "var {:atomic_usedmap} _USED_$$" << (*Globals.begin())->getName()
             << " : ["; 
          MW->writeType(OS, AHTVE->getOffset()->getType());
          OS << "][";
          MW->writeType(OS, AHTVE->getValue()->getType());
          OS << "]bool";
        });
      } else {
        ErrorReporter::reportImplementationLimitation(
                     "\"Atomic has taken value\" expressions for pointers not supported");
      }
  } else if (auto IMPLIESE = dyn_cast<ImpliesExpr>(E)) {
    OS << "(";
    writeExpr(OS, IMPLIESE->getLHS().get());
    OS << " ==> ";
    writeExpr(OS, IMPLIESE->getRHS().get());
    OS << ")";
  } else if (auto AHOE = dyn_cast<AccessHasOccurredExpr>(E)) {
    writeAccessHasOccurredVar(OS, AHOE->getArray().get(), AHOE->getAccessKind());
  } else if (auto AOE = dyn_cast<AccessOffsetExpr>(E)) {
    writeAccessOffsetVar(OS, AOE->getArray().get(), AOE->getAccessKind());
  } else if (auto NAE = dyn_cast<NotAccessedExpr>(E)) {
    auto GARE = dyn_cast<GlobalArrayRefExpr>(NAE->getArray().get());
    if (!GARE)
      llvm_unreachable("NotAccessedExpr must have array name argument");
    OS << "_NOT_ACCESSED_$$" << GARE->getArray()->getName();
  } else if (auto UnE = dyn_cast<UnaryExpr>(E)) {
    switch (UnE->getKind()) {
    case Expr::BVToPtr:
    case Expr::FAbs:
    case Expr::FCos:
    case Expr::FExp:
    case Expr::FFloor:
    case Expr::FLog:
    case Expr::FPConv:
    case Expr::FPow:
    case Expr::FPToSI:
    case Expr::FPToUI:
    case Expr::FrexpExp:
    case Expr::FrexpFrac:
    case Expr::FSin:
    case Expr::FSqrt:
    case Expr::FRsqrt:
    case Expr::OtherInt:
    case Expr::OtherBool:
    case Expr::OtherPtrBase:
    case Expr::PtrToBV:
    case Expr::SIToFP:
    case Expr::UIToFP:
    case Expr::GetImageWidth:
    case Expr::GetImageHeight: {
      std::string IntName; llvm::raw_string_ostream IntS(IntName);
      unsigned FromWidth = UnE->getSubExpr()->getType().width,
               ToWidth = UnE->getType().width;
      switch (UnE->getKind()) {
      case Expr::BVToPtr:
        IntS << "BV" << FromWidth << "_TO_PTR";
        break;
      case Expr::PtrToBV:
        IntS << "PTR_TO_BV" << ToWidth;
        break;
      case Expr::FAbs:
        IntS << "FABS" << ToWidth;
        break;
      case Expr::FCos:
        IntS << "FCOS" << ToWidth;
        break;
      case Expr::FExp:
        IntS << "FEXP" << ToWidth;
        break;
      case Expr::FFloor:
        IntS << "FFLOOR" << ToWidth;
        break;
      case Expr::FLog:
        IntS << "FLOG" << ToWidth;
        break;
      case Expr::FPConv:
        IntS << "FP" << FromWidth << "_CONV" << ToWidth;
        break;
      case Expr::FPow:
        IntS << "FPOW" << ToWidth;
        break;
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
      case Expr::FSin:
        IntS << "FSIN" << ToWidth;
        break;
      case Expr::FSqrt:
        IntS << "FSQRT" << ToWidth;
        break;
      case Expr::FRsqrt:
        IntS << "FRSQRT" << ToWidth;
        break;
      case Expr::OtherInt:
        IntS << "__other_bv" << ToWidth;
        break;
      case Expr::OtherBool:
        IntS << "__other_bool";
        break;
      case Expr::OtherPtrBase:
        IntS << "__other_arrayId";
        break;
      case Expr::SIToFP:
        IntS << "SI" << FromWidth << "_TO_FP" << ToWidth;
        break;
      case Expr::UIToFP:
        IntS << "UI" << FromWidth << "_TO_FP" << ToWidth;
        break;
      case Expr::GetImageWidth:
        IntS << "GET_IMAGE_WIDTH";
        break;
      case Expr::GetImageHeight:
        IntS << "GET_IMAGE_HEIGHT";
        break;
      default:
        llvm_unreachable("Unsupported unary expr opcode");
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
      llvm_unreachable("Unsupported unary expr");
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
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::BVAdd:  IntName = "ADD";  break;
      case Expr::BVSub:  IntName = "SUB";  break;
      case Expr::BVMul:  IntName = "MUL";  break;
      case Expr::BVSDiv: IntName = "SDIV"; break;
      case Expr::BVUDiv: IntName = "UDIV"; break;
      case Expr::BVSRem: IntName = "SREM"; break;
      case Expr::BVURem: IntName = "UREM"; break;
      case Expr::BVShl:  IntName = "SHL";  break;
      case Expr::BVAShr: IntName = "ASHR"; break;
      case Expr::BVLShr: IntName = "LSHR"; break;
      case Expr::BVAnd:  IntName = "AND";  break;
      case Expr::BVOr:   IntName = "OR";   break;
      case Expr::BVXor:  IntName = "XOR";  break;
      default: llvm_unreachable("huh?");
      }
      OS << "BV" << BinE->getType().width << "_" << IntName;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getArithmeticBinary(IntName, BinE->getKind(),
                                              BinE->getType().width);
      }, false);
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
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::BVUgt: IntName = "UGT"; break;
      case Expr::BVUge: IntName = "UGE"; break;
      case Expr::BVUlt: IntName = "ULT"; break;
      case Expr::BVUle: IntName = "ULE"; break;
      case Expr::BVSgt: IntName = "SGT"; break;
      case Expr::BVSge: IntName = "SGE"; break;
      case Expr::BVSlt: IntName = "SLT"; break;
      case Expr::BVSle: IntName = "SLE"; break;
      default: llvm_unreachable("huh?");
      }
      OS << "BV" << BinE->getLHS()->getType().width << "_" << IntName;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << MW->IntRep->getBooleanBinary(IntName, BinE->getKind(),
                                           BinE->getLHS()->getType().width);
      }, false);
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
      default: llvm_unreachable("huh?");
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
      default: llvm_unreachable("huh?");
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
    case Expr::PtrLt: {
      const char *IntName;
      switch (BinE->getKind()) {
      case Expr::PtrLt:     IntName = "PTR_LT"; break;
      default:  llvm_unreachable("huh?");
      }
      OS << IntName;
      MW->writeIntrinsic([&](llvm::raw_ostream &OS) {
        OS << "function " << IntName << "(";
        MW->writeType(OS, BinE->getLHS()->getType());
        OS << ", ";
        MW->writeType(OS, BinE->getLHS()->getType());
        OS << ") : bool";
      });
      break;
    }
    default:
      llvm_unreachable("Unsupported binary expr");
    }
    OS << "(";
    writeExpr(OS, BinE->getLHS().get());
    OS << ", ";
    writeExpr(OS, BinE->getRHS().get());
    OS << ")";
  } else if (auto LE = dyn_cast<LoadExpr>(E)) {
    auto PtrArr = LE->getArray().get();
    assert(!(isa<NullArrayRefExpr>(PtrArr) ||
             MW->M->global_begin() == MW->M->global_end()));
    std::set<GlobalArray *> Globals;
    if (!PtrArr->computeArrayCandidates(Globals)) {
      Globals.insert(MW->M->global_begin(), MW->M->global_end());
    }

    if (Globals.size() == 1) {
      OS << "$$" << (*Globals.begin())->getName() << "[";
      writeExpr(OS, LE->getOffset().get());
      OS << "]";
    } else {
      ErrorReporter::reportImplementationLimitation(
                                "Load expressions from pointers not supported");
    }
  } else if (auto AE = dyn_cast<AtomicExpr>(E)) {
    auto PtrArr = AE->getArray().get();
    assert(!(isa<NullArrayRefExpr>(PtrArr) ||
             MW->M->global_begin() == MW->M->global_end()));
    std::set<GlobalArray *> Globals;
    if (!PtrArr->computeArrayCandidates(Globals)) {
      Globals.insert(MW->M->global_begin(), MW->M->global_end());
    }

    if (Globals.size() == 1) {
      OS << "_ATOMIC_OP($$" << (*Globals.begin())->getName() << ", ";
      writeExpr(OS, AE->getOffset().get());
      OS << ")";
    } else {
      ErrorReporter::reportImplementationLimitation(
                              "Atomic expressions from pointers not supported");
    }
  } else if (dyn_cast<ArraySnapshotExpr>(E)) {
    llvm_unreachable("Handled at statement level");
  } else if (auto UAE = dyn_cast<UnderlyingArrayExpr>(E)) {
      auto Array = UAE->getArray().get();
      assert(!(isa<NullArrayRefExpr>(Array) || 
        MW->M->global_begin() == MW->M->global_end()));

      std::set<GlobalArray *> Globals;
      if (!Array->computeArrayCandidates(Globals)) {
        Globals.insert(MW->M->global_begin(), MW->M->global_end());
      }

      if (Globals.size() == 1) {
        OS << "$$" << (*Globals.begin())->getName();
      } else {
        ErrorReporter::reportImplementationLimitation(
                     "Underlying array expressions for pointers not supported");
      }
  } else if (auto MOE = dyn_cast<ArrayMemberOfExpr>(E)) {
    writeExpr(OS, MOE->getSubExpr().get(), Depth);
  } else {
    llvm_unreachable("Unsupported expression");
  }
}

void BPLExprWriter::writeAccessHasOccurredVar(llvm::raw_ostream &OS,
                                          bugle::Expr* PtrArr,
                                          std::string accessKind) {

  std::string prefix = "_" + accessKind + "_HAS_OCCURRED_$$";

  if (auto GARE = dyn_cast<GlobalArrayRefExpr>(PtrArr)) {
    OS << prefix << GARE->getArray()->getName();
  } else {
    std::set<GlobalArray *> Globals;
    if (!PtrArr->computeArrayCandidates(Globals)) {
      Globals.insert(MW->M->global_begin(), MW->M->global_end());
    }

    if (Globals.size() == 1 &&
        (*Globals.begin())->isGlobalOrGroupShared()) {
      OS << prefix << (*Globals.begin())->getName();
    } else {
      MW->UsesPointers = true;
      OS << "(";
      for (auto i = MW->M->global_begin(), e = MW->M->global_end();
           i != e; ++i) {
        if (!(*i)->isGlobalOrGroupShared())
          continue; // Accesses of local arrays are not tracked
        OS << "if (";
        writeExpr(OS, PtrArr);
        OS << " == $arrayId$$" << (*i)->getName() << ") then "
           << prefix << (*i)->getName() << " else ";
      }
      OS << "false)";
    }
  }
}

void BPLExprWriter::writeAccessOffsetVar(llvm::raw_ostream &OS,
                                          bugle::Expr* PtrArr,
                                          std::string accessKind) {

  if (MW->RaceInst == bugle::RaceInstrumenter::WATCHDOG_SINGLE) {
    OS << "_WATCHED_OFFSET";
    return;
  }

  std::string prefix;
  if(MW->RaceInst == bugle::RaceInstrumenter::STANDARD) {
    prefix = "_" + accessKind + "_OFFSET_$$";
  } else {
    assert(MW->RaceInst == bugle::RaceInstrumenter::WATCHDOG_MULTIPLE);
    prefix = "_WATCHED_OFFSET_$$";
  }

  if (auto GARE = dyn_cast<GlobalArrayRefExpr>(PtrArr)) {
    OS << prefix << GARE->getArray()->getName();
  } else {
    std::set<GlobalArray *> Globals;
    if (!PtrArr->computeArrayCandidates(Globals)) {
      Globals.insert(MW->M->global_begin(), MW->M->global_end());
    }

    if (Globals.size() == 1 &&
        (*Globals.begin())->isGlobalOrGroupShared()) {
      OS << prefix << (*Globals.begin())->getName();
    } else {
      MW->UsesPointers = true;
      OS << "(";
      for (auto i = MW->M->global_begin(), e = MW->M->global_end();
           i != e; ++i) {
        if (!(*i)->isGlobalOrGroupShared())
          continue; // Offsets of local arrays are not tracked
        OS << "if (";
        writeExpr(OS, PtrArr);
        OS << " == $arrayId$$" << (*i)->getName() << ") then "
           << prefix << (*i)->getName() << " else ";
      }
      OS << MW->IntRep->getLiteral(0, MW->M->getPointerWidth()) << ")";
    }
  }
}
