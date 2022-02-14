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

#include "lower_half_api.h"
#include "split_process.h"
#include "p2p_log_replay.h"
#include "p2p_drain_send_recv.h"
#include "record-replay.h"
#include "two-phase-algo.h"

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
      // drainMpiCollectives() will send worker state and get coord response.
      // Unfortunately, dmtcp_global_barrier()/DMT_BARRIER can't send worker
      // state and get a coord responds.  So, drainMpiCollective() will use the
      // special messages:  DMT_MPI_PRESUSPEND and DMT_MPI_PRESUSPEND_RESPONSE
      // 'INTENT' (intend to ckpt) acts as the first corodinator response.
      // * drainMpiCollective() calls preSuspendBarrier()
      // * mpi_presuspend_barrier() calls waitForMpiPresuspendBarrier()
      // FIXME:  See commant at: dmtcpplugin.cpp:'case DMTCP_EVENT_PRESUSPEND'
      {
        query_t coord_response = INTENT;
        int64_t round = 0;
        while (1) {
          // FIXME: see informCoordinator...() for the 2pc_data that we send
          //       to the coordinator.  Now, return it and use it below.
          rank_state_t data_to_coord = drainMpiCollectives(coord_response);

          string barrierId = "MANA-PRESUSPEND-" + jalib::XToString(round);
          string csId = "MANA-PRESUSPEND-CS-" + jalib::XToString(round);
          string commId = "MANA-PRESUSPEND-COMM-" + jalib::XToString(round);
          int64_t commKey = (int64_t) data_to_coord.comm;

          if (data_to_coord.st == IN_CS) {
            dmtcp_kvdb64(DMTCP_KVDB_INCRBY, csId.c_str(), 0, 1);
            dmtcp_kvdb64(DMTCP_KVDB_OR, commId.c_str(), commKey, 1);
          }

          dmtcp_global_barrier(barrierId.c_str());

          int64_t counter;
          if (dmtcp_kvdb64_get(csId.c_str(), 0, &counter) == -1) {
            // No rank published IN_CS state.
            coord_response == SAFE_TO_CHECKPOINT;
            break;
          }

          int64_t commStatus;
          if (dmtcp_kvdb64_get(commId.c_str(), commKey, &commStatus) == -1) {
            // No rank in our communicator is in CS; set our state to
            // WAIT_STRAGGLER
            coord_response = WAIT_STRAGGLER;
          } else if (data_to_coord.st == PHASE_1) {
            // We are in Phase 1, so we get a free pass.
            coord_response = FREE_PASS;
          }

          round++;
        }
      }
      break;

    case DMTCP_EVENT_PRECHECKPOINT:
      logIbarrierIfInTrivBarrier(); // two-phase-algo.cpp
      dmtcp_local_barrier("MPI:GetLocalLhMmapList");
      getLhMmapList(); // two-phase-algo.cpp
      dmtcp_local_barrier("MPI:GetLocalRankInfo");
      getLocalRankInfo(); // p2p_log_replay.cpp
      dmtcp_global_barrier("MPI:update-ckpt-dir-by-rank");
      updateCkptDirByRank(); // mpi_plugin.cpp
      dmtcp_global_barrier("MPI:Register-local-sends-and-receives");
      registerLocalSendsAndRecvs(); // p2p_drain_send_recv.cpp
      dmtcp_global_barrier("MPI:Drain-Send-Recv");
      drainSendRecv(); // p2p_drain_send_recv.cpp

    case DMTCP_EVENT_RESUME:
      clearPendingCkpt(); // two-phase-algo.cpp
      dmtcp_local_barrier("MPI:Reset-Drain-Send-Recv-Counters");
      resetDrainCounters(); // p2p_drain_send_recv.cpp
      break;

    case DMTCP_EVENT_RESTART:
      save2pcGlobals(); // two-phase-algo.cpp
      dmtcp_local_barrier("MPI:updateEnviron");
      updateLhEnviron(); // mpi-plugin.cpp
      dmtcp_local_barrier("MPI:Clear-Pending-Ckpt-Msg-Post-Restart");
      clearPendingCkpt(); // two-phase-algo.cpp
      dmtcp_local_barrier("MPI:Reset-Drain-Send-Recv-Counters");
      resetDrainCounters(); // p2p_drain_send_recv.cpp
      dmtcp_global_barrier("MPI:restoreMpiLogState");
      restoreMpiLogState(); // record-replay.cpp
      dmtcp_global_barrier("MPI:record-replay.cpp-void");
      replayMpiP2pOnRestart(); // p2p_log_replay.cpp
      dmtcp_local_barrier("MPI:p2p_log_replay.cpp-void");
      restore2pcGlobals(); // two-phase-algo.cpp
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
