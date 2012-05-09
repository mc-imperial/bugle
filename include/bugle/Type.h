namespace bugle {

#ifndef BUGLE_TYPE_H
#define BUGLE_TYPE_H

struct Type {
  enum {
    BV,
    Float,
    Pointer,
    ArrayId
  } Kind;

  Type kind;
  unsigned width;
};

}

#endif
