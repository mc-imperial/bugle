#ifndef BUGLE_INTEGERREPRESENTATION_H
#define BUGLE_INTEGERREPRESENTATION_H

#include "bugle/Expr.h"

namespace bugle {

class IntegerRepresentation {

public:
  virtual std::string getType(unsigned bitWidth) = 0;
  virtual std::string getLiteral(unsigned literal, unsigned bitWidth) = 0;
  virtual std::string getLiteralSuffix(unsigned bitWidth) = 0;
  virtual std::string getZeroExtend(unsigned FromWidth, unsigned ToWidth) = 0;
  virtual std::string getSignExtend(unsigned FromWidth, unsigned ToWidth) = 0;
  virtual std::string getExtract() = 0;
  virtual std::string getExtractExpr(const std::string &Expr, unsigned UpperBit,
                                     unsigned LowerBit) = 0;
  virtual std::string getConcat() = 0;
  virtual std::string getConcatExpr(const std::string &Lhs,
                                    const std::string &Rhs) = 0;
  virtual std::string getArithmeticBinary(std::string Name,
                                          bugle::Expr::Kind Kind,
                                          unsigned Width) = 0;
  virtual std::string getBooleanBinary(std::string Name, bugle::Expr::Kind Kind,
                                       unsigned Width) = 0;
  virtual void printVal(llvm::raw_ostream &OS, const llvm::APInt &Val) = 0;
  virtual bool abstractsExtract() = 0;
  virtual bool abstractsConcat() = 0;

  virtual ~IntegerRepresentation() {};
};

class BVIntegerRepresentation : public IntegerRepresentation {

public:
  virtual std::string getType(unsigned bitWidth);
  virtual std::string getLiteralSuffix(unsigned bitWidth);
  virtual std::string getLiteral(unsigned literal, unsigned bitWidth);
  virtual std::string getZeroExtend(unsigned FromWidth, unsigned ToWidth);
  virtual std::string getSignExtend(unsigned FromWidth, unsigned ToWidth);
  virtual std::string getExtract();
  virtual std::string getExtractExpr(const std::string &Expr, unsigned UpperBit,
                                     unsigned LowerBit);
  virtual std::string getConcat();
  virtual std::string getConcatExpr(const std::string &Lhs,
                                    const std::string &Rhs);
  virtual std::string
  getArithmeticBinary(std::string Name, bugle::Expr::Kind Kind, unsigned Width);
  virtual std::string getBooleanBinary(std::string Name, bugle::Expr::Kind Kind,
                                       unsigned Width);
  virtual void printVal(llvm::raw_ostream &OS, const llvm::APInt &Val);
  virtual bool abstractsExtract();
  virtual bool abstractsConcat();
};

class MathIntegerRepresentation : public IntegerRepresentation {

public:
  virtual std::string getType(unsigned bitWidth);
  virtual std::string getLiteralSuffix(unsigned bitWidth);
  virtual std::string getLiteral(unsigned literal, unsigned bitWidth);
  virtual std::string getZeroExtend(unsigned FromWidth, unsigned ToWidth);
  virtual std::string getSignExtend(unsigned FromWidth, unsigned ToWidth);
  virtual std::string getExtract();
  virtual std::string getExtractExpr(const std::string &Expr, unsigned UpperBit,
                                     unsigned LowerBit);
  virtual std::string getConcat();
  virtual std::string getConcatExpr(const std::string &Lhs,
                                    const std::string &Rhs);
  virtual std::string
  getArithmeticBinary(std::string Name, bugle::Expr::Kind Kind, unsigned Width);
  virtual std::string getBooleanBinary(std::string Name, bugle::Expr::Kind Kind,
                                       unsigned Width);
  virtual void printVal(llvm::raw_ostream &OS, const llvm::APInt &Val);
  virtual bool abstractsExtract();
  virtual bool abstractsConcat();
};
}

#endif