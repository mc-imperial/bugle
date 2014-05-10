#include "clang/AST/ASTConsumer.h"
#include "clang/AST/AST.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

namespace {

class CheckInlineVisitor : public RecursiveASTVisitor<CheckInlineVisitor> {
public:
  CheckInlineVisitor(CompilerInstance &CI) : Instance(CI) {}

  bool VisitFunctionDecl(FunctionDecl *F) {
    if (F->isInlineSpecified() && F->hasBody() &&
        !F->hasAttr<AlwaysInlineAttr>()) {
      FullSourceLoc FL = Instance.getASTContext().getFullLoc(F->getLocStart());
      DiagnosticsEngine &D = Instance.getDiagnostics();
      unsigned DiagID =
          D.getCustomDiagID(DiagnosticsEngine::Error,
                            "inline occurs without always_inline attribute");

      if (FL.isValid())
        D.Report(FL, DiagID);
      else
        D.Report(DiagID);
    }

    return true;
  }

private:
  CompilerInstance &Instance;
};

class CheckInlineConsumer : public ASTConsumer {
public:
  explicit CheckInlineConsumer(CompilerInstance &CI) : Visitor(CI) {}

  virtual void HandleTranslationUnit(ASTContext &AT) {
    Visitor.TraverseDecl(AT.getTranslationUnitDecl());
  }

private:
  CheckInlineVisitor Visitor;
};

class CheckInlineAction : public PluginASTAction {
protected:
  ASTConsumer *CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) {
    return new CheckInlineConsumer(CI);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) {
    if (args.size() == 1 && args[0] == "help") {
      PrintHelp(llvm::errs());
      return false;
    } else if (args.size()) {
      DiagnosticsEngine &D = CI.getDiagnostics();
      unsigned DiagID =
          D.getCustomDiagID(DiagnosticsEngine::Error, "invalid argument '%0'");
      D.Report(DiagID) << args[0];
      return false;
    }

    return true;
  }

  void PrintHelp(llvm::raw_ostream &ros) {
    ros << "Check that inline is combined with always_inline attribute\n";
  }
};
}

static FrontendPluginRegistry::Add<CheckInlineAction>
X("inline-check", "check that inline is combined with always_inline attribute");
