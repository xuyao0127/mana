#ifndef __MANA_LOGGER_H__
#define __MANA_LOGGER_H__

#include <string>
#include "dmtcpalloc.h"

namespace Logger
{
    void init();
    void record(dmtcp::string const& str);
    dmtcp::string getLogStr();
    void publishLogToCoordinator();
    void disable();
    void enable();
};

#endif // #ifndef __MANA_LOGGER_H__
