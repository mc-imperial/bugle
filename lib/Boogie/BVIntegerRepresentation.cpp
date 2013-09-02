#include "bugle/IntegerRepresentation.h"
#include "llvm/Support/raw_ostream.h"

namespace bugle {

std::string BVIntegerRepresentation::getType(unsigned bitWidth) {
  return getLiteralSuffix(bitWidth);
}

std::string BVIntegerRepresentation::getLiteralSuffix(unsigned bitWidth) {
  std::stringstream ss;
  ss << "bv" << bitWidth;
  return ss.str();
}

std::string BVIntegerRepresentation::getLiteral(unsigned literal, unsigned bitWidth) {
  std::stringstream ss;
  ss << literal << getLiteralSuffix(bitWidth);
  return ss.str();
}

std::string BVIntegerRepresentation::getZeroExtend(unsigned FromWidth, unsigned ToWidth) {
  std::stringstream ss;
  ss << "function {:bvbuiltin \"zero_extend " << (ToWidth - FromWidth)
         << "\"} BV" << FromWidth << "_ZEXT" << ToWidth << "(bv" << FromWidth
         << ") : bv" << ToWidth << ";";
  return ss.str();
}

std::string BVIntegerRepresentation::getSignExtend(unsigned FromWidth, unsigned ToWidth) {
  std::stringstream ss;
  ss << "function {:bvbuiltin \"sign_extend " << (ToWidth - FromWidth)
         << "\"} BV" << FromWidth << "_SEXT" << ToWidth << "(bv" << FromWidth
         << ") : bv" << ToWidth;
  return ss.str();
}

std::string BVIntegerRepresentation::getArithmeticBinary(std::string Name, bugle::Expr::Kind Kind, unsigned Width) {
  const char *SMTName;
  switch (Kind) {
  case Expr::BVAdd:  SMTName = "bvadd";  break;
  case Expr::BVSub:  SMTName = "bvsub";  break;
  case Expr::BVMul:  SMTName = "bvmul";  break;
  case Expr::BVSDiv: SMTName = "bvsdiv"; break;
  case Expr::BVUDiv: SMTName = "bvudiv"; break;
  case Expr::BVSRem: SMTName = "bvsrem"; break;
  case Expr::BVURem: SMTName = "bvurem"; break;
  case Expr::BVShl:  SMTName = "bvshl";  break;
  case Expr::BVAShr: SMTName = "bvashr"; break;
  case Expr::BVLShr: SMTName = "bvlshr"; break;
  case Expr::BVAnd:  SMTName = "bvand";  break;
  case Expr::BVOr:   SMTName = "bvor";   break;
  case Expr::BVXor:  SMTName = "bvxor";  break;
  default: assert(0 && "huh?"); return 0;
  }

	std::stringstream ss;
	ss << "function {:bvbuiltin \"" << SMTName << "\"} BV"
           << Width
           << "_" << Name << "(bv" << Width
           << ", bv" << Width
           << ") : bv" << Width << ";";
	return ss.str();
}

std::string BVIntegerRepresentation::getBooleanBinary(std::string Name, bugle::Expr::Kind Kind, unsigned Width) {
  const char *SMTName;
  switch (Kind) {
  case Expr::BVUgt: SMTName = "bvugt"; break;
  case Expr::BVUge: SMTName = "bvuge"; break;
  case Expr::BVUlt: SMTName = "bvult"; break;
  case Expr::BVUle: SMTName = "bvule"; break;
  case Expr::BVSgt: SMTName = "bvsgt"; break;
  case Expr::BVSge: SMTName = "bvsge"; break;
  case Expr::BVSlt: SMTName = "bvslt"; break;
  case Expr::BVSle: SMTName = "bvsle"; break;
  default: assert(0 && "huh?"); return 0;
  }

	std::stringstream ss;
	ss << "function {:bvbuiltin \"" << SMTName << "\"} BV"
           << Width
           << "_" << Name << "(bv" << Width
           << ", bv" << Width
           << ") : bool;";
	return ss.str();
}

void BVIntegerRepresentation::printVal(llvm::raw_ostream &OS, const llvm::APInt &Val) {
  Val.print(OS, /*isSigned=*/false);
  OS << getLiteralSuffix(Val.getBitWidth());
}

std::string BVIntegerRepresentation::getExtractExpr(const std::string &Expr, unsigned UpperBit, unsigned LowerBit) {
  std::stringstream ss;
  ss << Expr << "[" << UpperBit << ":" << LowerBit << "]";
  return ss.str();
}

bool BVIntegerRepresentation::abstractsExtract() {
  return false;
}

std::string BVIntegerRepresentation::getExtract() {
  assert(0 && "BVIntegerRepresentation should generate Boogie extract syntax");
  return 0;
}

bool BVIntegerRepresentation::abstractsConcat() {
  return false;
}

std::string BVIntegerRepresentation::getConcat() {
  assert(0 && "BVIntegerRepresentation should generate Boogie concatenation syntax");
  return 0;
}

std::string BVIntegerRepresentation::getConcatExpr(const std::string &Lhs, const std::string &Rhs) {
  std::stringstream ss;
  ss << Lhs << " ++ " << Rhs;
  return ss.str();
}


}