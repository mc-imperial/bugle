#ifndef BUGLE_SOURCELOC_H
#define BUGLE_SOURCELOC_H

#include <string>

namespace bugle {

class SourceLoc
{
private:
  unsigned    lineno;
  unsigned    colno;
  std::string fname;
  std::string path;

public:
  SourceLoc(unsigned          lineno, 
            unsigned          colno, 
            const std::string &fname, 
            const std::string &path) : lineno(lineno), 
                                       colno(colno), 
                                       fname(fname), 
                                       path(path) {}

  unsigned           getLineNo()   const { return lineno; }
  unsigned           getColNo()    const { return colno;  }
  const std::string &getFileName() const { return fname;  }
  const std::string &getPath()     const { return path;   }
};
}
#endif
