#ifndef BUGLE_SOURCELOCWRITER_H
#define BUGLE_SOURCELOCWRITER_H

#include "bugle/SourceLoc.h"

namespace llvm {

class ToolOutputFile;
}

namespace bugle {

class SourceLocWriter {
  llvm::ToolOutputFile *L;
  unsigned SourceLocCounter;

public:
  SourceLocWriter(llvm::ToolOutputFile *L) : L(L), SourceLocCounter(0) {}
  unsigned writeSourceLocs(const SourceLocsRef &sourcelocs);
};
}

#endif
