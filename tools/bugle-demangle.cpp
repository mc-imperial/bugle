#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"

#include "bugle/util/ErrorReporter.h"

using namespace llvm;

static cl::opt<std::string>
MangledName(cl::Positional, cl::desc("<mangled name>"),
    cl::init(""), cl::value_desc("name"));

static cl::opt<std::string>
SourceLanguage("l", cl::desc("Name source language (c, cu, cl; default cu)"),
    cl::value_desc("language"));

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "C++ Name demangler\n");

  bool isCPPName = false;
  if (SourceLanguage == "cu")
    isCPPName = true;
  else if (!SourceLanguage.empty() && SourceLanguage != "c" &&
           SourceLanguage != "cl") {
    std::string msg = "Unsupported source language: " + SourceLanguage;
    bugle::ErrorReporter::reportParameterError(msg);
  }

  outs() << bugle::ErrorReporter::demangleName(MangledName, isCPPName) << "\n";
  return 0;
}
