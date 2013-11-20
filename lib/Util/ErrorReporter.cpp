#include "bugle/util/ErrorReporter.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

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
