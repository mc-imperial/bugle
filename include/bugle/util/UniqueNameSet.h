#ifndef BUGLE_UTIL_UNIQUENAMESET_H
#define BUGLE_UTIL_UNIQUENAMESET_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace bugle {

class UniqueNameSet {
  llvm::StringSet<> Names;

public:
  std::string makeName(llvm::StringRef OrigName);
};
}

#endif
