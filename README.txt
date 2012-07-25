Compiling on OS X/Linux
-----------------------

1. Compile LLVM and Clang, as usual.
   http://clang.llvm.org/get_started.html

2. (Optional, only required when compiling OpenCL kernels) 
   Compile libclc, as usual.
   http://libclc.llvm.org/

3. Run CMake from a build directory:
     $ mkdir /path/to/bugle-build
     $ cd /path/to/bugle-build
     $ cmake -DCMAKE_PREFIX_PATH=/path/to/llvm-build -DCMAKE_BUILD_TYPE=Release -DLIBCLC_DIR=/path/to/libclc /path/to/bugle
   The LIBCLC_DIR parameter is optional.  Be sure to select the same
   build type (CMAKE_BUILD_TYPE) used when compiling LLVM.

4. Run 'make' from the build directory.

Compiling on Windows
--------------------

1. Compile LLVM and Clang, as usual.
   http://clang.llvm.org/get_started.html

2. (Optional, only required when compiling OpenCL kernels)
   Build libclc on an OS X/Linux machine, following the instructions
   above, but supply one additional parameter to configure.py:
     --prefix=/path/to/libclc-inst
   Install libclc to the libclc-inst directory using "make install",
   and copy the contents of libclc-inst to the Windows machine.

3. Run CMake from a build directory:
     $ mkdir C:\path\to\bugle-build
     $ cd C:\path\to\bugle-build
     $ cmake -G "Visual Studio 10" -DLLVM_SRC=C:\path\to\llvm-source -DLLVM_BUILD=C:\path\to\llvm-build -DLIBCLC_DIR=C:\path\to\libclc-inst -DLLVM_BUILD_TYPE=Release C:\path\to\bugle
   The LIBCLC_DIR parameter is optional.  Be sure to select the same
   build type (LLVM_BUILD_TYPE) used when compiling LLVM.

4. Open the Visual Studio project 'Bugle.sln'.  When building, use the build
   type you used for LLVM.

Running Bugle
-------------

To build a Boogie program using Bugle, use one of the convenience
scripts located in the Bugle build directory: blang (for C/C++) or
clblang (for OpenCL).  These scripts generate LLVM IR using Clang
which is then passed to Bugle.  For example:

  $ blang main.c                       # Generates main.bpl
  $ clblang kernel.cl                  # Generates kernel.gbpl
