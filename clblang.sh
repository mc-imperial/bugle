#!/bin/sh

BASE=`basename $1 .cl`

@LLVM_BINDIR@/clang -Xclang -cl-std=CL1.2 -O0 -ccc-host-triple nvptx--nvidiacl -I@CMAKE_SOURCE_DIR@/include -I@CMAKE_SOURCE_DIR@/include-libclc -I@LIBCLC_DIR@/generic/include -Xclang -mlink-bitcode-file -Xclang @LIBCLC_DIR@/nvptx--nvidiacl/lib/builtins.bc -Dcl_khr_fp64 -Dcl_clang_storage_class_specifiers -emit-llvm -c -o $BASE.bc "$@"
@LLVM_BINDIR@/opt -mem2reg -globaldce -o $BASE.opt.bc $BASE.bc
@CMAKE_BINARY_DIR@/bugle -l cl -o $BASE.gbpl $BASE.opt.bc
