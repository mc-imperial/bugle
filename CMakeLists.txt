cmake_minimum_required(VERSION 2.8)
project(Bugle)

set(LIBCLC_DIR "" CACHE PATH "Path to libclc")

set(CLBLANG_CLANG_OPTS "-Xclang -cl-std=CL1.2 -O0 -target nvptx-- -I${LIBCLC_DIR}/generic/include -Xclang -mlink-bitcode-file -Xclang ${LIBCLC_DIR}/lib/clc/nvptx--.bc -Dcl_khr_fp64 -Dcl_clang_storage_class_specifiers -D__OPENCL_VERSION__")
set(CLBLANG_BUGLE_OPTS "-l cl")

set(CUBLANG_CLANG_OPTS "-target nvptx-- -emit-llvm -Xclang -fcuda-is-device -D__CUDA_ARCH__")
set(CUBLANG_BUGLE_OPTS "-l cu")

if (NOT WIN32 OR MSYS OR CYGWIN)
  find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config DOC "llvm-config executable")
  if(LLVM_CONFIG_EXECUTABLE STREQUAL "LLVM_CONFIG_EXECUTABLE-NOTFOUND")
    message(FATAL_ERROR "llvm-config could not be found!")
  endif()

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --cxxflags
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(LLVM_CXXFLAGS "${LLVM_CXXFLAGS} -fno-exceptions -fno-rtti")

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs bitreader ipo target transformutils
    OUTPUT_VARIABLE LLVM_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --system-libs
    OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(LLVM_LIBS "${LLVM_LIBS} ${LLVM_SYSTEM_LIBS}")

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

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --src-root
    OUTPUT_VARIABLE LLVM_SRC
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND ${LLVM_CONFIG_EXECUTABLE} --obj-root
    OUTPUT_VARIABLE LLVM_BUILD
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(PLUGIN_CXXFLAGS "${LLVM_CXXFLAGS} -I\"${LLVM_SRC}/tools/clang/include\" -I\"${LLVM_BUILD}/tools/clang/include\"")

  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(PLUGIN_CXXFLAGS "${PLUGIN_CXXFLAGS} -Wno-strict-aliasing")
  endif()

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(PLUGIN_LDFLAGS "-Wl,-undefined,dynamic_lookup")
  endif()

  set(EXT ".c")
  set(OUTEXT "bpl")
  set(CLANG_OPTS "")
  set(BUGLE_OPTS "")
  configure_file(${CMAKE_SOURCE_DIR}/blang.sh.in ${CMAKE_BINARY_DIR}/blang @ONLY)

  set(EXT ".cl")
  set(OUTEXT "gbpl")
  set(CLANG_OPTS "${CLBLANG_CLANG_OPTS}")
  set(BUGLE_OPTS "${CLBLANG_BUGLE_OPTS}")
  configure_file(${CMAKE_SOURCE_DIR}/blang.sh.in ${CMAKE_BINARY_DIR}/clblang @ONLY)

  set(EXT ".cu")
  set(OUTEXT "gbpl")
  set(CLANG_OPTS "${CUBLANG_CLANG_OPTS}")
  set(BUGLE_OPTS "${CUBLANG_BUGLE_OPTS}")
  configure_file(${CMAKE_SOURCE_DIR}/blang.sh.in ${CMAKE_BINARY_DIR}/cublang @ONLY)
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

  set(LLVM_CXXFLAGS "\"/I${LLVM_SRC}/include\" \"/I${LLVM_BUILD}/include\" -D_SCL_SECURE_NO_WARNINGS -wd4146 -wd4244 -wd4355 -wd4482 -wd4800")
  set(LLVM_LDFLAGS "")
  set(LLVM_LIBS "${LLVM_LIBDIR}/LLVMipo.lib" "${LLVM_LIBDIR}/LLVMObjCARCOpts.lib" "${LLVM_LIBDIR}/LLVMVectorize.lib" "${LLVM_LIBDIR}/LLVMScalarOpts.lib" "${LLVM_LIBDIR}/LLVMInstCombine.lib" "${LLVM_LIBDIR}/LLVMTransformUtils.lib" "${LLVM_LIBDIR}/LLVMipa.lib" "${LLVM_LIBDIR}/LLVMAnalysis.lib" "${LLVM_LIBDIR}/LLVMTarget.lib" "${LLVM_LIBDIR}/LLVMMC.lib" "${LLVM_LIBDIR}/LLVMObject.lib" "${LLVM_LIBDIR}/LLVMBitReader.lib" "${LLVM_LIBDIR}/LLVMCore.lib" "${LLVM_LIBDIR}/LLVMSupport.lib")

  set(OUTEXT "bpl")
  set(CLANG_OPTS "")
  set(BUGLE_OPTS "")
  configure_file(${CMAKE_SOURCE_DIR}/blang.bat.in ${CMAKE_BINARY_DIR}/blang.bat)

  set(OUTEXT "gbpl")
  set(CLANG_OPTS "${CLBLANG_CLANG_OPTS}")
  set(BUGLE_OPTS "${CLBLANG_BUGLE_OPTS}")
  configure_file(${CMAKE_SOURCE_DIR}/blang.bat.in ${CMAKE_BINARY_DIR}/clblang.bat)

  set(OUTEXT "gbpl")
  set(CLANG_OPTS "${CUBLANG_CLANG_OPTS}")
  set(BUGLE_OPTS "${CUBLANG_BUGLE_OPTS}")
  configure_file(${CMAKE_SOURCE_DIR}/blang.bat.in ${CMAKE_BINARY_DIR}/cublang.bat)
endif()

include_directories(include)

add_library(bugleBoogie STATIC
  lib/Boogie/BPLExprWriter.cpp
  lib/Boogie/BPLFunctionWriter.cpp
  lib/Boogie/BPLModuleWriter.cpp
  lib/Boogie/BVIntegerRepresentation.cpp
  lib/Boogie/Expr.cpp
  lib/Boogie/Ident.cpp
  lib/Boogie/MathIntegerRepresentation.cpp
  lib/Boogie/SourceLocWriter.cpp
  lib/Boogie/Stmt.cpp
  include/bugle/BPLExprWriter.h
  include/bugle/BPLFunctionWriter.h
  include/bugle/BPLModuleWriter.h
  include/bugle/BasicBlock.h
  include/bugle/Casting.h
  include/bugle/Expr.h
  include/bugle/Function.h
  include/bugle/GlobalArray.h
  include/bugle/Ident.h
  include/bugle/IntegerRepresentation.h
  include/bugle/Module.h
  include/bugle/OwningPtrVector.h
  include/bugle/RaceInstrumenter.h
  include/bugle/Ref.h
  include/bugle/SourceLoc.h
  include/bugle/SourceLocWriter.h
  include/bugle/SpecificationInfo.h
  include/bugle/Stmt.h
  include/bugle/Type.h
  include/bugle/Var.h
)

add_library(buglePreprocessing STATIC
  lib/Preprocessing/CycleDetectPass.cpp
  lib/Preprocessing/InlinePass.cpp
  lib/Preprocessing/RestrictDetectPass.cpp
  lib/Preprocessing/SimpleInternalizePass.cpp
  include/bugle/Preprocessing/CycleDetectPass.h
  include/bugle/Preprocessing/InlinePass.h
  include/bugle/Preprocessing/RestrictDetectPass.h
  include/bugle/Preprocessing/SimpleInternalizePass.h
)

add_library(bugleTranslator STATIC
  lib/Translator/TranslateModule.cpp
  lib/Translator/TranslateFunction.cpp
  include/bugle/Translator/TranslateModule.h
  include/bugle/Translator/TranslateFunction.h
)

add_library(bugleTransform STATIC
  lib/Transform/SimplifyStmt.cpp
  include/bugle/Transform/SimplifyStmt.h
)

add_library(bugleUtil STATIC
  lib/Util/ErrorReporter.cpp
  lib/Util/UniqueNameSet.cpp
  include/bugle/util/ErrorReporter.h
  include/bugle/util/UniqueNameSet.h
  include/bugle/util/Functional.h
)

add_executable(bugle
  tools/bugle.cpp
)

set_target_properties(bugle bugleBoogie buglePreprocessing
                      bugleTransform bugleTranslator bugleUtil
    PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")

target_link_libraries(bugleTranslator bugleBoogie ${LLVM_LIBS} ${LLVM_LDFLAGS})
target_link_libraries(bugleTransform bugleBoogie)
target_link_libraries(buglePreprocessing bugleTranslator)
target_link_libraries(bugleBoogie bugleUtil)
target_link_libraries(bugle bugleBoogie buglePreprocessing bugleTranslator bugleTransform bugleUtil)

if(NOT WIN32 OR MSYS OR CYGWIN)

add_library(bugleInlineCheckPlugin SHARED
  lib/InlineCheckPlugin/InlineCheckPlugin.cpp
)

set_target_properties(bugleInlineCheckPlugin
    PROPERTIES COMPILE_FLAGS "${PLUGIN_CXXFLAGS}")

target_link_libraries(bugleInlineCheckPlugin ${PLUGIN_LIBS} ${PLUGIN_LDFLAGS})

endif()
