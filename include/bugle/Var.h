#ifndef BUGLE_VAR_H
#define BUGLE_VAR_H

#include "bugle/Ref.h"
#include "bugle/Type.h"
#include <string>

namespace bugle {

class Var {
  Type type;
  std::string name;

public:
  Var(Type type, const std::string &name) : type(type), name(name) {}
  Type getType() { return type; }
  const std::string &getName() { return name; }
};
}

#endif
