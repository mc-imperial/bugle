#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"

#include "bugle/BPLModuleWriter.h"
#include "bugle/IntegerRepresentation.h"
#include "bugle/Module.h"
#include "bugle/SourceLocWriter.h"
#include "bugle/Preprocessing/ArgumentPromotionPass.h"
#include "bugle/Preprocessing/ArgumentRenamePass.h"
#include "bugle/Preprocessing/CycleDetectPass.h"
#include "bugle/Preprocessing/InlinePass.h"
#include "bugle/Preprocessing/RestrictDetectPass.h"
#include "bugle/Preprocessing/SimpleInternalizePass.h"
#include "bugle/Preprocessing/StructSimplificationPass.h"
#include "bugle/Preprocessing/Vector3SimplificationPass.h"
#include "bugle/RaceInstrumenter.h"
#include "bugle/Transform/SimplifyStmt.h"
#include "bugle/Translator/TranslateModule.h"
#include "bugle/util/ErrorReporter.h"

#include <map>
#include <set>
#include <vector>

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

static cl::list<std::string> GPUEntryPoints(
    "k", cl::ZeroOrMore, cl::desc("GPU entry point function name"),
    cl::value_desc("function"));

static cl::opt<bugle::TranslateModule::SourceLanguage> SourceLanguage(
    "l", cl::desc("Module source language"),
    cl::init(bugle::TranslateModule::SL_C),
    cl::values(clEnumValN(bugle::TranslateModule::SL_C, "c", "C (default)"),
               clEnumValN(bugle::TranslateModule::SL_CUDA, "cu", "CUDA"),
               clEnumValN(bugle::TranslateModule::SL_OpenCL, "cl", "OpenCL")));

enum IntRep { BVIntRep, MathIntRep };

static cl::opt<IntRep> IntegerRepresentation(
    "i", cl::desc("Integer representation"), cl::init(BVIntRep),
    cl::values(clEnumValN(BVIntRep, "bv",
                          "Bitvector integer representation (default)"),
               clEnumValN(MathIntRep, "math",
                          "Mathematical integer representation")));

static cl::opt<bool> Inlining(
    "inline", cl::ValueDisallowed, cl::desc("Inline all function calls"));

#ifndef NDEBUG
static cl::opt<bool> DumpIR("dump-ir", cl::ValueDisallowed, cl::Hidden,
                            cl::desc("Dump the preprocessed IR"));
#endif

static cl::opt<bugle::RaceInstrumenter> RaceInstrumentation(
    "race-instrumentation", cl::desc("Race instrumentation method to use"),
    cl::init(bugle::RaceInstrumenter::WatchdogSingle),
    cl::values(clEnumValN(bugle::RaceInstrumenter::Original,
                          "original", "Original"),
               clEnumValN(bugle::RaceInstrumenter::WatchdogSingle,
                          "watchdog-single", "Watchdog single (default)"),
               clEnumValN(bugle::RaceInstrumenter::WatchdogMultiple,
                          "watchdog-multiple", "Watchdog multiple")));

static cl::list<std::string>
    GPUArraySizes("kernel-array-sizes", cl::ZeroOrMore,
                  cl::desc("Specify GPU entry point array sizes in bytes"),
                  cl::value_desc("function(,int)*"));

static cl::opt<bool> OnlyExplicitGPUEntryPoints(
    "only-explicit-entry-points", cl::ValueDisallowed,
    cl::desc("Only translate GPU entry points specified with k option"));

// The default values for the address spaces match NVPTXAddrSpaceMap in
// Targets.cpp. There does not appear to be a header file in which they are
// symbolically defined.
static cl::opt<unsigned> GlobalAddrSpace(
    "global-space", cl::desc("Global address space (default 1)"),
    cl::value_desc("int"), cl::init(1));

static cl::opt<unsigned> GroupSharedAddrSpace(
    "group-shared-space", cl::desc("Group shared address space (default 3)"),
    cl::value_desc("int"), cl::init(3));

static cl::opt<unsigned> ConstantAddrSpace(
    "constant-space", cl::desc("Constant address space (default 4)"),
    cl::value_desc("int"), cl::init(4));


static void CheckAddressSpaces() {
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
}

static void GetArraySizes(std::map<std::string, bugle::ArraySpec> &KAS) {
  Regex RegEx = Regex("([a-zA-Z_][a-zA-Z_0-9]*)((,[0-9\\*]+)*)");
  for (auto i = GPUArraySizes.begin(), e = GPUArraySizes.end(); i != e; ++i) {
    SmallVector<StringRef, 1> Matches;
    if (!RegEx.match(*i, &Matches) || Matches[0] != *i) {
      std::string msg = "Invalid GPU array size specifier: " + *i;
      bugle::ErrorReporter::reportParameterError(msg);
    }
    if (KAS.find(Matches[1]) != KAS.end()) {
      std::string msg =
          "Array sizes for " + Matches[1].str() + " specified multiple times";
      bugle::ErrorReporter::reportParameterError(msg);
    }
    SmallVector<StringRef, 1> MatchSizes;
    bugle::ArraySpec ArraySizes;
    Matches[2].split(MatchSizes, ",");
    for (auto si = MatchSizes.begin() + 1, se = MatchSizes.end(); si != se;
         ++si) {
      uint64_t size = 0;
      bool IsConstrained = !si->equals("*");
      if (IsConstrained && si->getAsInteger(0, size)) {
        std::string msg = "Array size too large: " + si->str();
        bugle::ErrorReporter::reportParameterError(msg);
      }
      ArraySizes.push_back(std::make_pair(IsConstrained, size));
    }
    KAS[Matches[1].str()] = ArraySizes;
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::PrettyStackTraceProgram X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  LLVMContext Context;

  cl::ParseCommandLineOptions(argc, argv, "LLVM to Boogie translator\n");

  std::string DisplayFilename;
  if (InputFilename == "-")
    DisplayFilename = "<stdin>";
  else
    DisplayFilename = InputFilename;
  bugle::ErrorReporter::setFileName(DisplayFilename);

  std::string ErrorMessage;
  std::unique_ptr<Module> M;

  // Read module
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputFilename);

  if (std::error_code EC = BufferOrErr.getError()) {
    ErrorMessage = EC.message();
  } else {
    std::unique_ptr<MemoryBuffer> &BufferPtr = BufferOrErr.get();
    ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
        expectedToErrorOrAndEmitErrors(
            Context,
            parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), Context));
    if (std::error_code EC = ModuleOrErr.getError())
      ErrorMessage = EC.message();
    else
      M = std::move(ModuleOrErr.get());
  }

  if (!M) {
    if (ErrorMessage.size())
      bugle::ErrorReporter::reportFatalError(ErrorMessage);
    else
      bugle::ErrorReporter::reportFatalError("Bitcode did not read correctly");
  }

  std::unique_ptr<bugle::IntegerRepresentation> IntRep;
  switch (IntegerRepresentation) {
  case BVIntRep:
    IntRep.reset(new bugle::BVIntegerRepresentation());
    break;
  case MathIntRep:
    IntRep.reset(new bugle::MathIntegerRepresentation());
    break;
  }

  CheckAddressSpaces();
  bugle::TranslateModule::AddressSpaceMap AddressSpaces(
      GlobalAddrSpace, GroupSharedAddrSpace, ConstantAddrSpace);

  std::set<std::string> EP;
  for (auto i = GPUEntryPoints.begin(), e = GPUEntryPoints.end(); i != e; ++i)
    EP.insert(*i);

  std::map<std::string, bugle::ArraySpec> KAS;
  GetArraySizes(KAS);

  legacy::PassManager PM;
  PM.add(new bugle::Vector3SimplificationPass());
  PM.add(new bugle::ArgumentPromotionPass(SourceLanguage, EP));
  PM.add(new bugle::StructSimplificationPass());
  if (Inlining) {
    PM.add(new bugle::CycleDetectPass());
    PM.add(new bugle::InlinePass(SourceLanguage, EP));
  }
  if (Inlining || OnlyExplicitGPUEntryPoints) {
    PM.add(new bugle::SimpleInternalizePass(SourceLanguage, EP,
                                            OnlyExplicitGPUEntryPoints));
  }
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createGlobalDCEPass());
  PM.add(new bugle::RestrictDetectPass(SourceLanguage, EP, AddressSpaces));
  PM.add(new bugle::ArgumentRenamePass());
#ifndef NDEBUG
  PM.add(createVerifierPass());
#endif
  PM.run(*M.get());

#ifndef NDEBUG
  if (DumpIR)
    M->dump();
#endif

  bugle::TranslateModule TM(M.get(), SourceLanguage, EP, RaceInstrumentation,
                            AddressSpaces, KAS);
  TM.translate();
  std::unique_ptr<bugle::Module> BM(TM.takeModule());

  bugle::simplifyStmt(BM.get());

  std::string OutFile = OutputFilename;
  if (OutFile.empty()) {
    SmallString<128> Path(InputFilename);
    sys::path::replace_extension(Path, "bpl");
    OutFile = sys::path::filename(Path);
  }

  std::error_code ErrorCode;
  tool_output_file F(OutFile, ErrorCode, sys::fs::F_Text);
  if (ErrorCode)
    bugle::ErrorReporter::reportFatalError(ErrorCode.message());

  tool_output_file *L = nullptr;
  if (!SourceLocationFilename.empty()) {
    L = new tool_output_file(SourceLocationFilename, ErrorCode,
                             sys::fs::F_Text);
    if (ErrorCode)
      bugle::ErrorReporter::reportFatalError(ErrorCode.message());
  }
  std::unique_ptr<bugle::SourceLocWriter> SLW(new bugle::SourceLocWriter(L));

  bugle::BPLModuleWriter MW(F.os(), BM.get(), IntRep.get(), RaceInstrumentation,
                            SLW.get());
  MW.write();

  F.os().flush();
  F.keep();

  if (L) {
    L->os().flush();
    L->keep();
  }

  return 0;
}
