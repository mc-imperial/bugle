#ifndef BUGLE_TYPE_H
#define BUGLE_TYPE_H

#include <assert.h>

namespace bugle {

struct Type {
  enum ArrayKind {
    ArrayOf
  };

  enum Kind {
    Bool,
    BV,
    Float,
    Pointer
  };

  bool array:1;
  Kind kind:31;
  unsigned width;

  Type(Kind kind) : array(false), kind(kind), width(0) {
    assert(kind == Bool);
  }

  Type(Kind kind, unsigned width) : array(false), kind(kind), width(width) {
    assert(kind != Bool);
  }

  Type(ArrayKind ak, Kind kind, unsigned width) :
    array(true), kind(kind), width(width) {
    assert(kind != Bool);
  }

  Type(ArrayKind ak, Type subType) :
    array(true), kind(subType.kind), width(subType.width) {
    assert(!subType.array);
  }

  bool operator==(const Type &other) const {
    return array == other.array && kind == other.kind && width == other.width;
  }

  bool operator!=(const Type &other) const {
    return array != other.array || kind != other.kind || width != other.width;
  }

  bool isKind(Kind k) const {
    return !array && kind == k;
  }

  Type range() const {
    assert(array);
    return Type(kind, width);
  }
};

}

#endif
