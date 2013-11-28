#include "bugle/util/ErrorReporter.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

#if defined(__clang__) || defined(__GNUC__)
#include <cxxabi.h>
#elif defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <Dbghelp.h>
#endif

using namespace bugle;
using namespace llvm;

std::string ErrorReporter::FileName;

void ErrorReporter::printErrorMsg(const std::string &msg) {
  errs() << FileName << ": ";
  if (errs().has_colors()) errs().changeColor(raw_ostream::Colors::RED);
  errs() << "error:";
  if (errs().has_colors()) errs().resetColor();
  errs() << " " << msg << "\n";
}

void ErrorReporter::setFileName(const std::string &FN) {
  std::string::size_type pos = FN.find_last_of("\\/");

  if (pos != std::string::npos && pos != FN.length() - 1)
    FileName = FN.substr(pos + 1);
  else
    FileName = FN;
}

std::string ErrorReporter::demangleName(const std::string &name, bool isCPPName) {
  if (!isCPPName)
    return std::string(name);

#if defined(__clang__) || defined(__GNUC__)
  int status;
  std::string demangledName = name;
  char *DN = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);
  if (status == 0) {
    demangledName = DN;
    free(DN);
  }
  return demangledName;
#elif defined(_MSC_VER)
  char DN[1024];
  std::string mangledName;
  // Mangled Microsoft C++ names start with a 0x1 in clang
  if (name[0] != 0x1)
    mangledName = name;
  else
    mangledName = name.substr(1);

  if (0 != UnDecorateSymbolName(mangledName.c_str(), DN, sizeof(DN),
                                UNDNAME_COMPLETE))
    return std::string(DN);
  else
    return std::string(name);
#endif
}

void ErrorReporter::emitWarning(const std::string &msg) {
  errs() << FileName << ": ";
  if (errs().has_colors()) errs().changeColor(raw_ostream::Colors::MAGENTA);
  errs() << "warning:";
  if (errs().has_colors()) errs().resetColor();
  errs() << " " << msg << "\n";
}

void ErrorReporter::reportParameterError(const std::string &msg) {
  if (errs().has_colors()) errs().changeColor(raw_ostream::Colors::RED);
  errs() << "error:";
  if (errs().has_colors()) errs().resetColor();
  errs() << " " << msg << "\n";
  std::exit(1);
}

void ErrorReporter::reportFatalError(const std::string &msg) {
  printErrorMsg(msg);
  std::exit(1);
}

void ErrorReporter::reportImplementationLimitation(const std::string &msg) {
  printErrorMsg(msg);
  errs() << "Please contact the developers;"
         << " this is an implementation limitation\n";
  std::exit(1);
}
