#!/bin/sh

BASE=`basename $1 .c`

@LLVM_BINDIR@/clang -I@CMAKE_SOURCE_DIR@/include -emit-llvm -c -o $BASE.bc "$@"
@LLVM_BINDIR@/opt -mem2reg -o $BASE.opt.bc $BASE.bc
@CMAKE_BINARY_DIR@/bugle -o $BASE.bpl $BASE.opt.bc
