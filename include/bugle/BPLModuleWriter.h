#ifndef BUGLE_BPLMODULEWRITER_H
#define BUGLE_BPLMODULEWRITER_H

#include "bugle/BPLExprWriter.h"
#include "bugle/RaceInstrumenter.h"
#include <functional>
#include <set>
#include <string>

namespace llvm {

class raw_ostream;
}

namespace bugle {

class IntegerRepresentation;
class Module;
class SourceLocWriter;
struct Type;

class BPLModuleWriter : BPLExprWriter {
  llvm::raw_ostream &OS;
  bugle::Module *M;
  bugle::IntegerRepresentation *IntRep;
  bugle::RaceInstrumenter RaceInst;
  bugle::SourceLocWriter *SLW;
  std::set<std::string> IntrinsicSet;
  bool UsesPointers, UsesFunctionPointers;
  bool RepresentPointersAsDatatype;
  std::string GlobalInitRequires;
  unsigned candidateNumber;

  const std::string &getGlobalInitRequires();
  void writeType(llvm::raw_ostream &OS, const bugle::Type &t);
  void writeIntrinsic(std::function<void(llvm::raw_ostream &)> F,
                      bool addSeparator = true);
  unsigned nextCandidateNumber();

public:
  BPLModuleWriter(llvm::raw_ostream &OS, bugle::Module *M,
                  bugle::IntegerRepresentation *IntRep,
                  bugle::RaceInstrumenter RaceInst, bugle::SourceLocWriter *SLW,
                  bool RepresentPointersAsDatatype)
      : BPLExprWriter(this), OS(OS), M(M), IntRep(IntRep), RaceInst(RaceInst),
        SLW(SLW), UsesPointers(false), UsesFunctionPointers(false),
        RepresentPointersAsDatatype(RepresentPointersAsDatatype),
        candidateNumber(0) {}

  void write();

  friend class BPLExprWriter;
  friend class BPLFunctionWriter;
};
}

#endif
