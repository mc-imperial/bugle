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

  static std::string ApplicationName;
  static void printErrorMsg(const std::string &msg);

public:
  static void setApplicationName(const std::string &AN);
  NO_RETURN static void reportFatalError(const std::string &msg);
  NO_RETURN static void reportImplementationLimitation(const std::string &msg);
};

}

#endif
