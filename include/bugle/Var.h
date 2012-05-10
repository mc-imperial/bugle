#include "bugle/Ref.h"
#include "bugle/Type.h"
#include <string>

#ifndef BUGLE_VAR_H
#define BUGLE_VAR_H

namespace bugle {

class Var {
  Type type;
  std::string name;

public:
  Type getType() { return type; }
  const std::string &getName() { return name; }
};

}

#endif
