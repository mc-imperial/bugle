#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "bugle/BPLModuleWriter.h"
#include "bugle/IntegerRepresentation.h"
#include "bugle/Module.h"
#include "bugle/SourceLocWriter.h"
#include "bugle/Preprocessing/CycleDetectPass.h"
#include "bugle/Preprocessing/InlinePass.h"
#include "bugle/Preprocessing/RemoveBodyPass.h"
#include "bugle/Preprocessing/RemovePrototypePass.h"
#include "bugle/Preprocessing/RestrictDetectPass.h"
#include "bugle/RaceInstrumenter.h"
#include "bugle/Transform/SimplifyStmt.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/util/ErrorReporter.h"

using namespace llvm;

static cl::opt<std::string> InputFilename(
    cl::Positional, cl::desc("<input bitcode file>"), cl::init("-"),
    cl::value_desc("filename"));

static cl::opt<std::string> OutputFilename(
    "o", cl::desc("Override output filename"), cl::init(""),
    cl::value_desc("filename"));

static cl::opt<std::string> SourceLocationFilename(
    "s", cl::desc("File for saving source locations"), cl::init(""),
    cl::value_desc("filename"));

static cl::opt<std::string> GPUEntryPoints(
    "k", cl::ZeroOrMore, cl::desc("GPU entry point function name"),
    cl::value_desc("function"));

static cl::opt<std::string> SourceLanguage(
    "l", cl::desc("Module source language (c, cu, cl; default c)"),
    cl::value_desc("language"));

static cl::opt<std::string> IntegerRepresentation(
    "i", cl::desc("Integer representation (bv, math; default bv)"),
    cl::value_desc("intrep"));

static cl::opt<bool> Inlining(
    "inline", cl::ValueDisallowed, cl::desc("Inline all function calls"));

static cl::opt<std::string> RaceInstrumentation(
    "race-instrumentation",
    cl::desc("Race instrumentation method to use (original, watchdog-single, "
             "watchdog-multiple; default watchdog-single)"));

static cl::opt<bool> DatatypePointerRepresentation(
    "datatype", cl::ValueDisallowed,
    cl::desc("Use datatype representation for pointers"));

// The default values for the address spaces match NVPTXAddrSpaceMap in
// Targets.cpp. There does not appear to be a header file in which they are
// symbolically defined
static cl::opt<unsigned> GlobalAddrSpace(
    "global-space",
    cl::desc("Set address space used as \"global\" (default 1)"),
    cl::value_desc("integer"), cl::init(1));

static cl::opt<unsigned> GroupSharedAddrSpace(
    "group-shared-space",
    cl::desc("Set address space used as \"group shared\" (default 3)"),
    cl::value_desc("integer"), cl::init(3));

static cl::opt<unsigned> ConstantAddrSpace(
    "constant-space",
    cl::desc("Set address space used as \"constant\" (default 4)"),
    cl::value_desc("integer"), cl::init(4));

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  LLVMContext &Context = getGlobalContext();

  cl::ParseCommandLineOptions(argc, argv, "LLVM to Boogie translator\n");

  std::string DisplayFilename;
  if (InputFilename == "-")
    DisplayFilename = "<stdin>";
  else
    DisplayFilename = InputFilename;
  bugle::ErrorReporter::setFileName(DisplayFilename);

  std::string ErrorMessage;
  std::unique_ptr<Module> M;

  // Use the bitcode streaming interface
  DataStreamer *streamer = getDataFileStreamer(InputFilename, &ErrorMessage);
  if (streamer) {
    M.reset(getStreamedBitcodeModule(DisplayFilename, streamer, Context,
                                     &ErrorMessage));
    if (M.get() != 0) {
      if (M->MaterializeAllPermanently(&ErrorMessage)) {
        M.reset();
      }
    }
  }

  if (M.get() == 0) {
    if (ErrorMessage.size())
      bugle::ErrorReporter::reportFatalError(ErrorMessage);
    else
      bugle::ErrorReporter::reportFatalError("Bitcode did not read correctly");
  }

  bugle::TranslateModule::SourceLanguage SL;
  if (SourceLanguage.empty() || SourceLanguage == "c")
    SL = bugle::TranslateModule::SL_C;
  else if (SourceLanguage == "cu")
    SL = bugle::TranslateModule::SL_CUDA;
  else if (SourceLanguage == "cl")
    SL = bugle::TranslateModule::SL_OpenCL;
  else {
    std::string msg = "Unsupported source language: " + SourceLanguage;
    bugle::ErrorReporter::reportParameterError(msg);
  }

  std::unique_ptr<bugle::IntegerRepresentation> IntRep;
  if (IntegerRepresentation.empty() || IntegerRepresentation == "bv")
    IntRep.reset(new bugle::BVIntegerRepresentation());
  else if (IntegerRepresentation == "math")
    IntRep.reset(new bugle::MathIntegerRepresentation);
  else {
    std::string msg =
        "Unsupported integer representation: " + IntegerRepresentation;
    bugle::ErrorReporter::reportParameterError(msg);
  }

  bugle::RaceInstrumenter RaceInst;
  if (RaceInstrumentation.empty() || RaceInstrumentation == "watchdog-single")
    RaceInst = bugle::RaceInstrumenter::WatchdogSingle;
  else if (RaceInstrumentation == "original")
    RaceInst = bugle::RaceInstrumenter::Original;
  else if (RaceInstrumentation == "watchdog-multiple")
    RaceInst = bugle::RaceInstrumenter::WatchdogMultiple;
  else {
    std::string msg =
        "Unsupported race instrumentation: " + RaceInstrumentation;
    bugle::ErrorReporter::reportParameterError(msg);
  }

  if (GlobalAddrSpace == 0 || GlobalAddrSpace == GroupSharedAddrSpace ||
      GlobalAddrSpace == ConstantAddrSpace) {
    std::string msg =
        "Global address space cannot be 0 or equal to group shared or constant "
        "address space";
    bugle::ErrorReporter::reportParameterError(msg);
  } else if (GroupSharedAddrSpace == 0 ||
             GroupSharedAddrSpace == GlobalAddrSpace ||
             GroupSharedAddrSpace == ConstantAddrSpace) {
    std::string msg =
        "Group shared address space cannot be 0 or equal to global or constant "
        "address space";
    bugle::ErrorReporter::reportParameterError(msg);
  } else if (ConstantAddrSpace == 0 || ConstantAddrSpace == GlobalAddrSpace ||
             ConstantAddrSpace == GroupSharedAddrSpace) {
    std::string msg =
        "Constant address space cannot be 0 or equal to global or group shared "
        "address space";
    bugle::ErrorReporter::reportParameterError(msg);
  }
  bugle::TranslateModule::AddressSpaceMap AddressSpaces(
      GlobalAddrSpace, GroupSharedAddrSpace, ConstantAddrSpace);

  std::set<std::string> EP;
  for (auto i = GPUEntryPoints.begin(), e = GPUEntryPoints.end(); i != e; ++i)
    EP.insert(&*i);

  PassManager PM;
  if (Inlining) {
    PM.add(new bugle::CycleDetectPass());
    PM.add(new bugle::InlinePass(SL, EP));
    PM.add(new bugle::RemoveBodyPass(SL, EP));
    PM.add(new bugle::RemovePrototypePass());
  }
  PM.add(new bugle::RestrictDetectPass(SL, EP, AddressSpaces));
  PM.run(*M.get());

  bugle::TranslateModule TM(M.get(), SL, EP, RaceInst, AddressSpaces);
  TM.translate();
  std::unique_ptr<bugle::Module> BM(TM.takeModule());

  bugle::simplifyStmt(BM.get());

  std::string OutFile = OutputFilename;
  if (OutFile.empty()) {
    SmallString<128> Path(InputFilename);
    sys::path::replace_extension(Path, "bpl");
    OutFile = sys::path::filename(Path);
  }

  std::string ErrorInfo;
  tool_output_file F(OutFile.c_str(), ErrorInfo);
  if (!ErrorInfo.empty())
    bugle::ErrorReporter::reportFatalError(ErrorInfo);

  tool_output_file *L = 0;
  if (!SourceLocationFilename.empty()) {
    L = new tool_output_file(SourceLocationFilename.c_str(), ErrorInfo);
    if (!ErrorInfo.empty())
      bugle::ErrorReporter::reportFatalError(ErrorInfo);
  }
  std::unique_ptr<bugle::SourceLocWriter> SLW(new bugle::SourceLocWriter(L));

  bugle::BPLModuleWriter MW(F.os(), BM.get(), IntRep.get(), RaceInst, SLW.get(),
                            DatatypePointerRepresentation);
  MW.write();

  F.os().flush();
  F.keep();

  if (L != 0) {
    L->os().flush();
    L->keep();
  }

  return 0;
}
