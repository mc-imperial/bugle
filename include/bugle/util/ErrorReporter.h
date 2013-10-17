#ifndef BUGLE_UTIL_ERRORREPORTER_H
#define BUGLE_UTIL_ERRORREPORTER_H

#include <string>

namespace bugle {

class ErrorReporter {
private:
  ErrorReporter();

  static std::string ApplicationName;
  static void printErrorMsg(const std::string &msg);

public:
  static void setApplicationName(const std::string &AN);
  [[noreturn]] static void reportFatalError(const std::string &msg);
  [[noreturn]] static void reportImplementationLimitation(const std::string &msg);
};

}

#endif
