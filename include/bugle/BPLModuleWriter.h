#ifndef BUGLE_BPLMODULEWRITER_H
#define BUGLE_BPLMODULEWRITER_H

#include "bugle/BPLExprWriter.h"
#include <functional>
#include <set>
#include <string>

namespace llvm {

class raw_ostream;

}

namespace bugle {

class IntegerRepresentation;
class Module;
struct Type;

class BPLModuleWriter : BPLExprWriter {
  llvm::raw_ostream &OS;
  bugle::Module *M;
  bugle::IntegerRepresentation *IntRep;
  std::set<std::string> IntrinsicSet;
  bool UsesPointers;
  std::string GlobalInitRequires;
  unsigned candidateNumber;

  const std::string &getGlobalInitRequires();
  void writeType(llvm::raw_ostream &OS, const bugle::Type &t);
  void writeIntrinsic(std::function<void(llvm::raw_ostream &)> F,
                      bool addSeparator = true);
  unsigned nextCandidateNumber();

public:
  BPLModuleWriter(llvm::raw_ostream &OS, bugle::Module *M, 
	              bugle::IntegerRepresentation *IntRep) :
    BPLExprWriter(this), OS(OS), M(M), IntRep(IntRep), UsesPointers(false),
    candidateNumber(0) {}

  void write();

  friend class BPLExprWriter;
  friend class BPLFunctionWriter;
};

}

#endif
