void f(int *i);

void foo() {
  int i[3];
  i[2] = 42;
  f(i);
}
