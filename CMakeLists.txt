cmake_minimum_required(VERSION 2.8)
project(Bugle)

if (NOT WIN32 OR MSYS OR CYGWIN)
  find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config DOC "llvm-config executable")
  if(LLVM_CONFIG_EXECUTABLE STREQUAL "LLVM_CONFIG_EXECUTABLE-NOTFOUND")
    message(FATAL_ERROR "llvm-config could not be found!")
  endif()

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --cflags
    OUTPUT_VARIABLE LLVM_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(LLVM_CFLAGS "${LLVM_CFLAGS} -fno-exceptions -fno-rtti")

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs bitreader target
    OUTPUT_VARIABLE LLVM_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --bindir
    OUTPUT_VARIABLE LLVM_BINDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(CFLAGS_CXX11 "-std=c++0x")

  set(GPUVERIFY_BENCHMARKING_DIR "" CACHE PATH "Path to GPUVerifyBenchmarking")
  set(LIBCLC_DIR "" CACHE PATH "Path to libclc")

  configure_file(${CMAKE_SOURCE_DIR}/blang.sh ${CMAKE_BINARY_DIR}/blang)
  configure_file(${CMAKE_SOURCE_DIR}/clblang.sh ${CMAKE_BINARY_DIR}/clblang)
  configure_file(${CMAKE_SOURCE_DIR}/clblang-libclc.sh ${CMAKE_BINARY_DIR}/clblang-libclc)
  configure_file(${CMAKE_SOURCE_DIR}/cublang.sh ${CMAKE_BINARY_DIR}/cublang)
else()
  set(LLVM_SRC "" CACHE PATH "LLVM source directory")
  set(LLVM_BUILD "" CACHE PATH "LLVM build directory")
  set(LLVM_BUILD_TYPE "" CACHE STRING "LLVM build type")

  if (NOT EXISTS "${LLVM_SRC}/include/llvm")
    message(FATAL_ERROR "Invalid LLVM source directory: ${LLVM_SRC}")
  endif()

  set(LLVM_LIBDIR "${LLVM_BUILD}/lib/${LLVM_BUILD_TYPE}")
  if (NOT EXISTS "${LLVM_LIBDIR}")
    message(FATAL_ERROR "Invalid LLVM build directory: ${LLVM_BUILD}")
  endif()

  set(LLVM_CFLAGS "\"/I${LLVM_SRC}/include\" \"/I${LLVM_BUILD}/include\" -D_SCL_SECURE_NO_WARNINGS -wd4146 -wd4244 -wd4355 -wd4800")
  set(LLVM_LDFLAGS "")
  set(LLVM_LIBS "${LLVM_LIBDIR}/LLVMTarget.lib" "${LLVM_LIBDIR}/LLVMMC.lib" "${LLVM_LIBDIR}/LLVMObject.lib" "${LLVM_LIBDIR}/LLVMBitReader.lib" "${LLVM_LIBDIR}/LLVMCore.lib" "${LLVM_LIBDIR}/LLVMSupport.lib")

  set(CFLAGS_CXX11 "")
endif()

include_directories(include)

add_library(bugleAnalysis STATIC
  lib/Analysis/PointerAnalysis.cpp
)

add_library(bugleBoogie STATIC
  lib/Boogie/BPLExprWriter.cpp
  lib/Boogie/BPLFunctionWriter.cpp
  lib/Boogie/BPLModuleWriter.cpp
  lib/Boogie/Expr.cpp
  lib/Boogie/Ident.cpp
)

add_library(bugleTranslator STATIC
  lib/Translator/TranslateModule.cpp
  lib/Translator/TranslateFunction.cpp
)

add_library(bugleTransform STATIC
  lib/Transform/SimplifyStmt.cpp
)

add_library(bugleUtil STATIC
  lib/Util/UniqueNameSet.cpp
)

add_executable(bugle
  tools/bugle.cpp
)

set_target_properties(bugle bugleAnalysis bugleBoogie bugleTransform
                      bugleTranslator bugleUtil
    PROPERTIES COMPILE_FLAGS "${CFLAGS_CXX11} ${LLVM_CFLAGS}")
set_target_properties(bugle
    PROPERTIES LINK_FLAGS "${LLVM_LDFLAGS}")

target_link_libraries(bugleTranslator bugleBoogie ${LLVM_LIBS})
target_link_libraries(bugleTransform bugleBoogie)
target_link_libraries(bugleBoogie bugleUtil)
target_link_libraries(bugle bugleAnalysis bugleBoogie bugleTranslator bugleTransform bugleUtil)
