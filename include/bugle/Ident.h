#ifndef BUGLE_IDENT_H
#define BUGLE_IDENT_H

#include <string>

namespace llvm {

class StringRef;

}

namespace bugle {

std::string makeBoogieIdent(llvm::StringRef S);

}

#endif
