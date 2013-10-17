#include "bugle/util/ErrorReporter.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

using namespace bugle;
using namespace llvm;

std::string ErrorReporter::ApplicationName;

void ErrorReporter::printErrorMsg(const std::string &msg) {
  errs() << ApplicationName << ": ";
  if (errs().has_colors()) errs().changeColor(raw_ostream::Colors::RED, true);
  errs() << "error:";
  if (errs().has_colors()) errs().resetColor();
  errs() << " " << msg << "\n";
}

void ErrorReporter::setApplicationName(const std::string &AN) {
  std::string::size_type pos = AN.find_last_of("\\/");

  if (pos != std::string::npos && pos != AN.length() - 1)
    ApplicationName = AN.substr(pos + 1);
  else
    ApplicationName = AN;
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
