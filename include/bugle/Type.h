#ifndef BUGLE_TYPE_H
#define BUGLE_TYPE_H

#include <assert.h>

namespace bugle {

struct Type {
  enum Kind {
    Bool,
    BV,
    Float,
    Pointer,
    ArrayId
  };

  Kind kind;
  unsigned width;

  Type(Kind kind) : kind(kind), width(0) {
    assert(kind == ArrayId || kind == Bool);
  }

  Type(Kind kind, unsigned width) : kind(kind), width(width) {
    assert(kind != ArrayId && kind != Bool);
  }

  bool operator==(const Type &other) const {
    return kind == other.kind && width == other.width;
  }

  bool operator!=(const Type &other) const {
    return kind != other.kind || width != other.width;
  }
};

}

#endif
