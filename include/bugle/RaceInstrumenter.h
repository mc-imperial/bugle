#ifndef BUGLE_RACEINSTRUMENTATION_H
#define BUGLE_RACEINSTRUMENTATION_H

namespace bugle {

enum RaceInstrumenter {
  Standard,
  WatchdogSingle,
  WatchdogMultiple
};

}

#endif