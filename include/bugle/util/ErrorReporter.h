#ifndef BUGLE_UTIL_ERRORREPORTER_H
#define BUGLE_UTIL_ERRORREPORTER_H

#include <string>

#if defined(__clang__) || defined(__GNUC__)
#define NO_RETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define NO_RETURN __declspec(noreturn)
#endif

namespace bugle {

class ErrorReporter {
private:
  ErrorReporter();

  static std::string FileName;
  static void printErrorMsg(const std::string &msg);

public:
  static void setFileName(const std::string &FN);
  static void emitWarning(const std::string &msg);
  NO_RETURN static void reportParameterError(const std::string &msg);
  NO_RETURN static void reportFatalError(const std::string &msg);
  NO_RETURN static void reportImplementationLimitation(const std::string &msg);
};

}

#endif
