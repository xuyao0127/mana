/****************************************************************************
 *   Copyright (C) 2019-2021 by Gene Cooperman, Rohan Garg, Yao Xu          *
 *   gene@ccs.neu.edu, rohgarg@ccs.neu.edu, xu.yao1@northeastern.edu        *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License in the files COPYING and COPYING.LESSER.  If not, see           *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#include <signal.h>
#include <sys/personality.h>
#include <sys/mman.h>

#include "lower_half_api.h"
#include "split_process.h"
#include "p2p_log_replay.h"
#include "p2p_drain_send_recv.h"
#include "record-replay.h"
#include "seq_num.h"

#include "config.h"
#include "dmtcp.h"
#include "util.h"
#include "jassert.h"
#include "jfilesystem.h"
#include "protectedfds.h"
#include "procselfmaps.h"

using namespace dmtcp;

/* Global variables */

int g_numMmaps = 0;
MmapInfo_t *g_list = NULL;

// #define DEBUG

#undef dmtcp_skip_memory_region_ckpting
const VA HIGH_ADDR_START = (VA)0x7fff00000000;

static inline int
regionContains(const void *haystackStart,
               const void *haystackEnd,
               const void *needleStart,
               const void *needleEnd)
{
  return needleStart >= haystackStart && needleEnd <= haystackEnd;
}

EXTERNC int
dmtcp_skip_memory_region_ckpting(const ProcMapsArea *area)
{
  if (area->addr == lh_info.startText ||
      strstr(area->name, "/dev/zero") ||
      strstr(area->name, "/dev/kgni") ||
      // FIXME: must comment out for VASP 5/RPA jobs on 2 knl nodes,
      // don't know why.
      strstr(area->name, "/SYSV") ||
      strstr(area->name, "/dev/xpmem") ||
      strstr(area->name, "/dev/shm") ||
      area->addr == lh_info.startData) {
    JTRACE("Ignoring region")(area->name)((void*)area->addr);
    return 1;
  }
  if (!g_list) return 0;
  for (int i = 0; i < g_numMmaps; i++) {
    void *lhMmapStart = g_list[i].addr;
    void *lhMmapEnd = (VA)g_list[i].addr + g_list[i].len;
    if (!g_list[i].unmapped &&
        regionContains(lhMmapStart, lhMmapEnd, area->addr, area->endAddr)) {
      JTRACE("Ignoring region")
           (area->name)((void*)area->addr)(area->size)
           (lhMmapStart)(lhMmapEnd);
      return 1;
    } else if (!g_list[i].unmapped &&
               regionContains(area->addr, area->endAddr,
                              lhMmapStart, lhMmapEnd)) {
      JTRACE("Unhandled case")
           (area->name)((void*)area->addr)(area->size)
           (lhMmapStart)(lhMmapEnd);
    }
  }
  return 0;
}

// Handler for SIGSEGV: forces the code into an infinite loop for attaching
// GDB and debugging
void
segvfault_handler(int signum, siginfo_t *siginfo, void *context)
{
  int dummy = 0;
  JNOTE("Caught a segmentation fault. Attach gdb to inspect...");
  while (!dummy);
}

// Installs a handler for SIGSEGV; useful for debugging crashes
void
initialize_segv_handler()
{
  static struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = segvfault_handler;
  sigemptyset(&action.sa_mask);

  JASSERT(sigaction(SIGSEGV, &action, NULL) != -1)
    (JASSERT_ERRNO).Text("Could not set up the segfault handler");
}

// Sets the global 'g_list' pointer to the beginning of the MmapInfo_t array
// in the lower half
static void
getLhMmapList()
{
  getMmappedList_t fnc = (getMmappedList_t)lh_info.getMmappedListFptr;
  if (fnc) {
    g_list = fnc(&g_numMmaps);
  }
  JTRACE("Lower half region info")(g_numMmaps);
  for (int i = 0; i < g_numMmaps; i++) {
    JTRACE("Lh region")(g_list[i].addr)(g_list[i].len)(g_list[i].unmapped);
  }
}

// Sets the lower half's __environ variable to point to upper half's __environ
static void
updateLhEnviron()
{
  updateEnviron_t fnc = (updateEnviron_t)lh_info.updateEnvironFptr;
  fnc(__environ);
}

static void
computeUnionOfCkptImageAddresses()
{
  ProcSelfMaps procSelfMaps; // This will be deleted when out of scope.
  Area area;

  // This code assumes that /proc/sys/kernel/randomize_va_space is 0
  //   or that personality(ADDR_NO_RANDOMIZE) was set.
  char buf;
  JWARNING(Util::readAll("/proc/sys/kernel/randomize_va_space", &buf, 1) != 1 ||
           buf == '0' ||
           (personality(0xffffffff) & ADDR_NO_RANDOMIZE) == 0)
    .Text("FIXME:  Linux process randomization is turned on.\n");

  // Initialize the MPI information to NULL (unknown).
  void *libsStart = NULL;
  void *minLibsStart = NULL;
  void *libsEnd = NULL;
  void *maxLibsEnd = NULL;
  void *highMemStart = NULL;
  void *minHighMemStart = NULL;

  // We discover the last address that the kernel had mapped:
  // ASSUMPTIONS:  Target app is not using mmap
  // ASSUMPTIONS:  Kernel does not go back and fill previos gap after munmap
  libsEnd = mmap(NULL, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  JASSERT(libsEnd != MAP_FAILED);
  munmap(libsEnd, 4096);

  // Preprocess memory regions as needed.
  while (procSelfMaps.getNextArea(&area)) {
    if (Util::strEndsWith(area.name, ".so")) {
      if (libsStart == NULL) {
        libsStart = area.addr;
      }
      libsEnd = area.endAddr;
    }

    // Set mtcpHdr->highMemStart if possible.
    // highMemStart not currently used.  We could change the constant logic.
    if (highMemStart == NULL && area.addr > HIGH_ADDR_START) {
      highMemStart = area.addr;
    }
  }

  string kvdb = "MANA_CKPT_UNION";
  dmtcp_kvdb64(DMTCP_KVDB_MIN, kvdb.c_str(), 0, (int64_t) libsStart);
  dmtcp_kvdb64(DMTCP_KVDB_MAX, kvdb.c_str(), 1, (int64_t) libsEnd);
  dmtcp_kvdb64(DMTCP_KVDB_MIN, kvdb.c_str(), 2, (int64_t) highMemStart);

  dmtcp_global_barrier(kvdb.c_str());

  JASSERT(dmtcp_kvdb64_get(kvdb.c_str(), 0, (int64_t*) &minLibsStart) == 0);
  JASSERT(dmtcp_kvdb64_get(kvdb.c_str(), 1, (int64_t*) &maxLibsEnd) == 0);
  JASSERT(dmtcp_kvdb64_get(kvdb.c_str(), 2, (int64_t*) &minHighMemStart) == 0);

  ostringstream o;

#define HEXSTR(o, x) o << #x << std::hex << x;
  HEXSTR(o, libsStart);
  HEXSTR(o, libsEnd);
  HEXSTR(o, highMemStart);
  HEXSTR(o, minLibsStart);
  HEXSTR(o, maxLibsEnd);
  HEXSTR(o, minHighMemStart);

  JTRACE("Union of memory regions") (o.str());

  string minLibsStartStr = jalib::XToString(minLibsStart);
  string maxLibsEndStr = jalib::XToString(maxLibsEnd);
  string minHighMemStartStr = jalib::XToString(minHighMemStart);

  // Now publish these values to DMTCP ckpt-header.
  dmtcp_add_to_ckpt_header("MANA_MinLibsStart", minLibsStartStr.c_str());
  dmtcp_add_to_ckpt_header("MANA_MaxLibsEnd", maxLibsEndStr.c_str());
  dmtcp_add_to_ckpt_header("MANA_MinHighMemStart", minHighMemStartStr.c_str());
}

static void
mpi_plugin_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  switch (event) {
    case DMTCP_EVENT_INIT:
    {
      JTRACE("*** DMTCP_EVENT_INIT");
      initialize_segv_handler();
      JASSERT(!splitProcess()).Text("Failed to create, initialize lower haf");
      break;
    }
    case DMTCP_EVENT_EXIT:
      JTRACE("*** DMTCP_EVENT_EXIT");
      break;

    case DMTCP_EVENT_PRESUSPEND:
      drainMpiCollectives(); // seq_num.cpp
      break;

    case DMTCP_EVENT_PRECHECKPOINT:
      dmtcp_local_barrier("MPI:GetLocalLhMmapList");
      getLhMmapList();
      dmtcp_local_barrier("MPI:GetLocalRankInfo");
      getLocalRankInfo(); // p2p_log_replay.cpp
      dmtcp_global_barrier("MPI:update-ckpt-dir-by-rank");
      updateCkptDirByRank(); // mpi_plugin.cpp
      dmtcp_global_barrier("MPI:Register-local-sends-and-receives");
      registerLocalSendsAndRecvs(); // p2p_drain_send_recv.cpp
      dmtcp_global_barrier("MPI:Drain-Send-Recv");
      drainSendRecv(); // p2p_drain_send_recv.cpp
      computeUnionOfCkptImageAddresses();

    case DMTCP_EVENT_RESUME:
      dmtcp_local_barrier("MPI:Reset-Drain-Send-Recv-Counters");
      resetDrainCounters(); // p2p_drain_send_recv.cpp
      seq_num_reset();
      break;

    case DMTCP_EVENT_RESTART:
      dmtcp_local_barrier("MPI:updateEnviron");
      updateLhEnviron(); // mpi-plugin.cpp
      dmtcp_local_barrier("MPI:Reset-Drain-Send-Recv-Counters");
      resetDrainCounters(); // p2p_drain_send_recv.cpp
      dmtcp_global_barrier("MPI:restoreMpiLogState");
      restoreMpiLogState(); // record-replay.cpp
      dmtcp_global_barrier("MPI:record-replay.cpp-void");
      replayMpiP2pOnRestart(); // p2p_log_replay.cpp
      dmtcp_local_barrier("MPI:p2p_log_replay.cpp-void");
      seq_num_reset();
      dmtcp_local_barrier("MPI:seq_num_reset");
      break;

    default:
      break;
  }
}

DmtcpPluginDescriptor_t mpi_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  PACKAGE_VERSION,
  "mpi_plugin",
  "DMTCP",
  "dmtcp@ccs.neu.edu",
  "MPI Proxy Plugin",
  mpi_plugin_event_hook
};

DMTCP_DECL_PLUGIN(mpi_plugin);
