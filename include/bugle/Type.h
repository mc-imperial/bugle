namespace bugle {

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
