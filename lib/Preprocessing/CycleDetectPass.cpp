#include "bugle/Preprocessing/CycleDetectPass.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace bugle;

CycleDetectPass::CycleDetectPass() : ModulePass(ID) {
  initializeCallGraphAnalysisGroup(*PassRegistry::getPassRegistry());
}

bool CycleDetectPass::runOnModule(llvm::Module &M) {
  CallGraphNode* N = getAnalysis<CallGraph>().getRoot();
  scc_iterator<CallGraphNode*> i = scc_begin(N), e = scc_end(N);
  while (i != e) {
    if (i.hasLoop()) {
      llvm::errs() << "Cannot inline, detected cycle in callgraph\n";
      std::exit(1);
    }
    ++i;
  }

  return false;
}

char CycleDetectPass::ID = 0;
