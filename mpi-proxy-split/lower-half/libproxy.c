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

// This could be libmpi.a or libproxy.a, with code to translate
//   between an MPI function and its address (similarly to dlsym()).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <sys/auxv.h>
#include <linux/limits.h>
#include <mpi.h>
#include <limits.h>
#include <fcntl.h>
#include <errno.h>
#include <complex.h>

#ifdef SINGLE_CART_REORDER
#include "../cartesian.h"
#endif

#include "libproxy.h"
#include "mpi_copybits.h"
#include "procmapsutils.h"
#include "lower_half_api.h"
#include "../mana_header.h"

LowerHalfInfo_t lh_info = {0};
// This is the allocated buffer for lh_info.memRange
MemRange_t lh_memRange = {0};
LhCoreRegions_t lh_core_regions[MAX_LH_REGIONS] = {0};
int totalRegions = 0;

static ucontext_t g_appContext;

static void* MPI_Fnc_Ptrs[] = {
  NULL,
  FOREACH_FNC(GENERATE_FNC_PTR)
  NULL,
};

// Local functions

LhCoreRegions_t*
getLhRegionsList(int *num)
{
  if (!num || *num > MAX_LH_REGIONS) return NULL;
  *num = totalRegions;
  return lh_core_regions;
}

static void
getDataFromMaps(const Area *text, Area *heap)
{
  Area area;
  int mapsfd = open("/proc/self/maps", O_RDONLY);
  void *heap_sbrk = sbrk(0);
  int idx = 0;
  // For a static LH, mark all the regions till heap as core regions.
  // TODO: for a dynamic lower-half, core regions list will include libraries
  //       and libraries are usually mapped beyond the heap.
  while (readMapsLine(mapsfd, &area)) {
    lh_core_regions[idx].start_addr = area.addr;
    lh_core_regions[idx].end_addr = area.endAddr;
    lh_core_regions[idx].prot = area.prot;
    idx++;
    if (strstr(area.name, "[heap]") && area.endAddr >= (VA)heap_sbrk) {
      *heap = area;
      break;
    }
  }
  totalRegions = idx;
  close(mapsfd);
}

// FIXME: This code is duplicated in proxy and plugin. Refactor into utils.
static void
getTextSegmentRange(pid_t proc,                 // IN
                    unsigned long *start,       // OUT
                    unsigned long *end,         // OUT
                    unsigned long *stackstart)  // OUT
{
  // From man 5 proc: See entry for /proc/[pid]/stat
  int pid;
  char cmd[PATH_MAX]; char state;
  int ppid; int pgrp; int session; int tty_nr; int tpgid;
  unsigned flags;
  unsigned long minflt; unsigned long cminflt; unsigned long majflt;
  unsigned long cmajflt; unsigned long utime; unsigned long stime;
  long cutime; long cstime; long priority; long nice;
  long num_threads; long itrealvalue;
  unsigned long long starttime;
  unsigned long vsize;
  long rss;
  unsigned long rsslim; unsigned long startcode; unsigned long endcode;
  unsigned long startstack; unsigned long kstkesp; unsigned long kstkeip;
  unsigned long signal_map; unsigned long blocked; unsigned long sigignore;
  unsigned long sigcatch; unsigned long wchan; unsigned long nswap;
  unsigned long cnswap;
  int exit_signal; int processor;
  unsigned rt_priority; unsigned policy;

  FILE *f = NULL;
  if (proc == -1) {
    f = fopen("/proc/self/stat", "r");
  } else {
    // On 64-bit systems, pid_max can be set to any value up to 2^22
    // (PID_MAX_LIMIT, approximately 4 million).
    char pids[PATH_MAX];
    snprintf(pids, sizeof pids, "/proc/%u/stat", proc);
    f = fopen(pids, "r");
  }
  if (f) {
    fscanf(f, "%d "
              "%s %c "
              "%d %d %d %d %d "
              "%u "
              "%lu %lu %lu %lu %lu %lu "
              "%ld %ld %ld %ld %ld %ld "
              "%llu "
              "%lu "
              "%ld "
              "%lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu "
              "%d %d %u %u",
           &pid,
           cmd, &state,
           &ppid, &pgrp, &session, &tty_nr, &tpgid,
           &flags,
           &minflt, &cminflt, &majflt, &cmajflt, &utime, &stime,
           &cutime, &cstime, &priority, &nice, &num_threads, &itrealvalue,
           &starttime,
           &vsize,
           &rss,
           &rsslim, &startcode, &endcode, &startstack, &kstkesp, &kstkeip,
           &signal_map, &blocked, &sigignore, &sigcatch, &wchan, &nswap,
           &cnswap,
           &exit_signal, &processor,
           &rt_priority, &policy);
  }
  fclose(f);
  *start      = startcode;
  *end        = endcode;
  *stackstart = startstack;
}

static char**
copyArgv(int argc, char **argv)
{
  char **new_argv = malloc((argc+1) * sizeof *new_argv);
  for(int i = 0; i < argc; ++i)
  {
      size_t length = strlen(argv[i])+1;
      new_argv[i] = malloc(length);
      memcpy(new_argv[i], argv[i], length);
  }
  new_argv[argc] = NULL;
  return new_argv;
}

static int
isValidFd(int fd)
{
  return fcntl(fd, F_GETFL, 0) != -1;
}

// Global functions

void
updateEnviron(const char **newenviron)
{
  __environ = (char **)newenviron;
}

// MPI Spec: A call to MPI_INIT has the same effect as a call to
// MPI_INIT_THREAD with a required = MPI_THREAD_SINGLE.
int
getRank(int init_flag)
{
  int flag;
  int world_rank = -1;
  int retval = MPI_SUCCESS;
  int provided;

  MPI_Initialized(&flag);
  if (!flag) {
    if (init_flag == MPI_INIT_NO_THREAD) {
      retval = MPI_Init(NULL, NULL);
    }
    else {
      retval = MPI_Init_thread(NULL, NULL, init_flag, &provided);
    }
  }
  if (retval == MPI_SUCCESS) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  }
  return world_rank;
}

#ifdef SINGLE_CART_REORDER
// Prior to checkpoint we will use the normal variable names, and
// after restart we will use the '_prime' suffix with variable names.
MPI_Comm comm_cart_prime;

int
getCoordinates(CartesianProperties *cp, int *coords, int init_flag)
{
  int flag;
  int comm_old_rank = -1;
  int comm_cart_rank = -1;
  int retval = MPI_SUCCESS;
  int provided;

  MPI_Initialized(&flag);
  if (!flag) {
    if (init_flag == MPI_INIT_NO_THREAD) {
      retval = MPI_Init(NULL, NULL);
    }
    else {
      retval = MPI_Init_thread(NULL, NULL, init_flag, &provided);
    }
  }
  if (retval == MPI_SUCCESS) {
    MPI_Cart_create(MPI_COMM_WORLD, cp->ndims, cp->dimensions, cp->periods,
                    cp->reorder, &comm_cart_prime);
    MPI_Comm_rank(comm_cart_prime, &comm_cart_rank);
    MPI_Cart_coords(comm_cart_prime, comm_cart_rank, cp->ndims, coords);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_old_rank);
  }
  return comm_old_rank;
}

void
getCartesianCommunicator(MPI_Comm **comm_cart)
{
  *comm_cart = &comm_cart_prime;
}
#endif

void*
mydlsym(enum MPI_Fncs fnc)
{
  if (fnc < MPI_Fnc_NULL || fnc > MPI_Fnc_Invalid) {
    return NULL;
  }
  return MPI_Fnc_Ptrs[fnc];
}

__attribute__((constructor))
void first_constructor()
{
  static int firstTime = 1;

  if (firstTime) {
    int i,myid, numprocs;
    int source,count;
    int msg_size = 4000;
    double complex *local_buf = (double complex*) malloc(msg_size * sizeof(double complex));
    double complex *global_buf = (double complex*) malloc(msg_size * sizeof(double complex));
    MPI_Group group;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_group(MPI_COMM_WORLD, &group);

    source = 0;
    count = msg_size;
    if(myid == source){
      for(i=0; i<count; i++)
        local_buf[i] = 1.0 + 2.0 * I;
    }
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    uint64_t mpi_timer = 0;
    for (int j = 0; j < 1000000; j++) {
      MPI_Bcast(global_buf, msg_size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (myid == 0) {
      fprintf(stderr, "Time used: %f seconds\n", end_time - start_time);
      fflush(stderr);
    }
    free(local_buf);
    free(global_buf);
    MPI_Finalize();
  } else {
    DLOG(NOISE, "(2) Constructor: Running in the parent?\n");
  }
}

__attribute__((destructor))
void second_destructor()
{
  // Destructor: The application called exit in the destructor to
  // get here. After this, we call setcontext() to get back in the
  // application.
  DLOG(NOISE, "Destructor!\n");
}
