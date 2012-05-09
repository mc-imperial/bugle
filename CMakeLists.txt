cmake_minimum_required(VERSION 2.8)

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
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --libs bitreader
        OUTPUT_VARIABLE LLVM_LIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --ldflags
        OUTPUT_VARIABLE LLVM_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(bugleAnalysis STATIC
  lib/Analysis/PointerAnalysis.cpp
)

add_library(bugleTranslator STATIC
  lib/Translator/BasicBlockTranslator.cpp
)

add_executable(bugle
  tools/bugle.cpp
)

set_target_properties(bugle bugleAnalysis bugleTranslator PROPERTIES COMPILE_FLAGS "${LLVM_CFLAGS}")
set_target_properties(bugle PROPERTIES LINK_FLAGS "${LLVM_LDFLAGS}")

target_link_libraries(bugle bugleAnalysis bugleTranslator ${LLVM_LIBS})
