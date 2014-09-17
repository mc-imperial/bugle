#include "bugle/Preprocessing/CycleDetectPass.h"
#include "bugle/util/ErrorReporter.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/CallGraph.h"

using namespace llvm;
using namespace bugle;

bool CycleDetectPass::runOnModule(llvm::Module &M) {
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  scc_iterator<CallGraph *> i = scc_begin(&CG), e = scc_end(&CG);
  while (i != e) {
    if (i.hasLoop()) {
      ErrorReporter::reportFatalError(
          "Cannot inline, detected cycle in callgraph");
    }
    ++i;
  }

  return false;
}

char CycleDetectPass::ID = 0;
