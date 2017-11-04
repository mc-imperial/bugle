Bugle
=====

GPUVerify's LLVM IR to Boogie translator.

Compiling on OS X/Linux
-----------------------

1. Build LLVM and Clang, as usual (see http://clang.llvm.org/get_started.html).

2. Run CMake from a build directory:
   ```
   $ mkdir /path/to/bugle-build
   $ cd /path/to/bugle-build
   $ cmake -DCMAKE_PREFIX_PATH=/path/to/llvm-build -DCMAKE_BUILD_TYPE=Release /path/to/bugle
   ```
   Be sure to select the same build type (CMAKE_BUILD_TYPE) used when compiling
   LLVM.

3. Run 'make' from the build directory.

Compiling on Windows
--------------------

1. Build LLVM and Clang, as usual (see http://clang.llvm.org/get_started.html).

2. Run CMake from a build directory:
   ```
   $ mkdir C:\path\to\bugle-build
   $ cd C:\path\to\bugle-build
   $ cmake -G "Visual Studio 14.0" -DLLVM_SRC=C:\path\to\llvm-source -DLLVM_BUILD=C:\path\to\llvm-build -DLLVM_BUILD_TYPE=Release C:\path\to\bugle
   ```
   Be sure to select the same build type (LLVM_BUILD_TYPE) used when compiling
   LLVM.

3. Open the Visual Studio project 'Bugle.sln'.  When building, use the build
   type you used for LLVM.

Running Bugle
-------------

Bugle is best run as part of GPUVerify. 
