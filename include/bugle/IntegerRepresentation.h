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
  virtual std::string getCtlz(unsigned Width) = 0;
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
  std::string getType(unsigned bitWidth) override;
  std::string getLiteralSuffix(unsigned bitWidth) override;
  std::string getLiteral(unsigned literal, unsigned bitWidth) override;
  std::string getZeroExtend(unsigned FromWidth, unsigned ToWidth) override;
  std::string getSignExtend(unsigned FromWidth, unsigned ToWidth) override;
  std::string getExtract() override;
  std::string getExtractExpr(const std::string &Expr, unsigned UpperBit,
                             unsigned LowerBit) override;
  std::string getConcat() override;
  std::string getConcatExpr(const std::string &Lhs,
                            const std::string &Rhs) override;
  std::string getCtlz(unsigned Width) override;
  std::string getArithmeticBinary(std::string Name, bugle::Expr::Kind Kind,
                                  unsigned Width) override;
  std::string getBooleanBinary(std::string Name, bugle::Expr::Kind Kind,
                               unsigned Width) override;
  void printVal(llvm::raw_ostream &OS, const llvm::APInt &Val) override;
  bool abstractsExtract() override;
  bool abstractsConcat() override;
};

class MathIntegerRepresentation : public IntegerRepresentation {
public:
  std::string getType(unsigned bitWidth) override;
  std::string getLiteralSuffix(unsigned bitWidth) override;
  std::string getLiteral(unsigned literal, unsigned bitWidth) override;
  std::string getZeroExtend(unsigned FromWidth, unsigned ToWidth) override;
  std::string getSignExtend(unsigned FromWidth, unsigned ToWidth) override;
  std::string getExtract() override;
  std::string getExtractExpr(const std::string &Expr, unsigned UpperBit,
                             unsigned LowerBit) override;
  std::string getConcat() override;
  std::string getConcatExpr(const std::string &Lhs,
                            const std::string &Rhs) override;
  std::string getCtlz(unsigned Width) override;
  std::string getArithmeticBinary(std::string Name, bugle::Expr::Kind Kind,
                                  unsigned Width) override;
  std::string getBooleanBinary(std::string Name, bugle::Expr::Kind Kind,
                               unsigned Width) override;
  void printVal(llvm::raw_ostream &OS, const llvm::APInt &Val) override;
  bool abstractsExtract() override;
  bool abstractsConcat() override;
};
}

#endif
