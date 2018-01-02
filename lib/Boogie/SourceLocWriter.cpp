#include "bugle/SourceLocWriter.h"
#include "bugle/SourceLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace bugle;

unsigned SourceLocWriter::writeSourceLocs(const SourceLocsRef &SourceLocs) {
  ++SourceLocCounter;
  if (L == nullptr)
    return SourceLocCounter - 1;

  llvm::raw_ostream &OS = L->os();

  for (const auto &SL : *SourceLocs) {
    OS << SL.getLineNo() << "\x1F";   // unit separator
    OS << SL.getColNo() << "\x1F";    // unit separator
    OS << SL.getFileName() << "\x1F"; // unit separator
    OS << SL.getPath() << "\x1F";     // unit separator
    OS << "\x1E";                     // record separator
  }
  OS << "\x1D"; // group separator

  return SourceLocCounter - 1;
}
