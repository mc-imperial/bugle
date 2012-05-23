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

class Module;
struct Type;

class BPLModuleWriter : BPLExprWriter {
  llvm::raw_ostream &OS;
  bugle::Module *M;
  std::set<std::string> IntrinsicSet;
  std::string GlobalInitRequires;

  BPLModuleWriter *getModuleWriter();
  const std::string &getGlobalInitRequires();
  void writeType(llvm::raw_ostream &OS, const bugle::Type &t);
  void writeIntrinsic(std::function<void(llvm::raw_ostream &)> F);

public:
  BPLModuleWriter(llvm::raw_ostream &OS, bugle::Module *M) : OS(OS), M(M) {}

  void write();

  friend class BPLExprWriter;
  friend class BPLFunctionWriter;
};

}

#endif
