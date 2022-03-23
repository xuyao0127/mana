#include <mpi.h>

#include <map>
#include <pthread.h>
#include <semaphore.h>

#include "jassert.h"
#include "seq_num.h"
#include "mpi_nextfunc.h"
#include "virtual-ids.h"

using namespace dmtcp_mpi;

#define DB_CONVERGED 1

#define DEBUG_SEQ_NUM

extern int g_world_rank;
extern int g_world_size;
volatile bool ckpt_pending;
volatile bool in_cs;
pthread_rwlock_t seq_num_lock;
sem_t user_thread_sem;
sem_t ckpt_thread_sem;
std::map<unsigned int, unsigned int> seq_num;
std::map<unsigned int, unsigned int> target_seq_num;
typedef std::pair<unsigned int, unsigned int> comm_seq_pair_t;

void seq_num_init() {
  ckpt_pending = false;
  pthread_rwlock_init(&seq_num_lock, NULL);
  sem_init(&user_thread_sem, 0, 0);
  sem_init(&ckpt_thread_sem, 0, 0);
}

void seq_num_reset() {
  ckpt_pending = false;
  for (comm_seq_pair_t i : seq_num) {
    i.second = 0;
  }

  // release user thread
  sem_trywait(&user_thread_sem);
  sem_post(&user_thread_sem);
}

void seq_num_destroy() {
  pthread_rwlock_destroy(&seq_num_lock);
  sem_destroy(&user_thread_sem);
  sem_destroy(&ckpt_thread_sem);
}

void check_seq_num(int *converged, int *new_target_found) {
  unsigned int comm_id;
  unsigned int seq;
  *converged = 1;
  *new_target_found = 0;
  for (comm_seq_pair_t pair : seq_num) {
    comm_id = pair.first;
    seq = pair.second;
    if (target_seq_num[comm_id] > seq_num[comm_id]) {
      *converged = 0;
    } else if (target_seq_num[comm_id] < seq_num[comm_id]) {
      *new_target_found = 1;
    }
  }
}

void commit_begin(MPI_Comm comm, const char *name) {
  pthread_rwlock_rdlock(&seq_num_lock);
  int converged = 0;
  int new_target_found = 0;
  unsigned int global_comm_id =
    VirtualGlobalCommId::instance().getGlobalId(comm);
  if (ckpt_pending) {
#if 0
    sem_post(&ckpt_thread_sem);
    sem_wait(&user_thread_sem);
#else
    sem_wait(&user_thread_sem);
    printf("rank %d comm id %x seq %d\n", g_world_rank, global_comm_id, seq_num[global_comm_id]);
    fflush(stdout);
  }
#endif
  seq_num[global_comm_id]++;
  in_cs = true;
  pthread_rwlock_unlock(&seq_num_lock);
}

void commit_finish(MPI_Comm comm) {
  pthread_rwlock_rdlock(&seq_num_lock);
  in_cs = false;
  pthread_rwlock_unlock(&seq_num_lock);
}

void upload_seq_num() {
  unsigned int comm_id;
  unsigned int seq;
  for (comm_seq_pair_t pair : seq_num) {
    comm_id = pair.first;
    seq = pair.second;
    dmtcp_kvdb64(DMTCP_KVDB_MAX, "/mana/comm-seq-max", comm_id, seq);
  }
}

void download_targets() {
  int64_t max_seq = 0;
  unsigned int comm_id;
  for (comm_seq_pair_t pair : seq_num) {
    comm_id = pair.first;
    dmtcp_kvdb64_get("/mana/comm-seq-max", comm_id, &max_seq);
    target_seq_num[comm_id] = max_seq;
  }
}

void drainMpiCollectives() {
  int64_t all_converged;
  int converged = 0;
  int new_target_found = 0;

  pthread_rwlock_wrlock(&seq_num_lock);
  ckpt_pending = true;
  upload_seq_num();
  dmtcp_global_barrier("mana/comm-seq-round");
  download_targets();
  pthread_rwlock_unlock(&seq_num_lock);

  all_converged = 0;
  while (all_converged != g_world_size) {
    // Reset all converged scaler
    if (g_world_rank == 0) {
      dmtcp_kvdb64(DMTCP_KVDB_SET, "/mana/seq-num-states",
                   DB_CONVERGED, 0);
    }
    dmtcp_global_barrier("mana/comm-seq-round-start");
#if 0
    while (1) {
      // Check local sequence number and targets
      check_seq_num(&converged, &new_target_found);
      if (converged || new_target_found) {
        dmtcp_global_barrier("mana/comm-seq-round-finish");
        break;
      } else {
        // Notify the user thread to continue and resume when a new
        // communication is found.
        sem_post(&user_thread_sem);
        sem_wait(&ckpt_thread_sem);
      }
    }
#endif
    // Only ranks found new targets update the global sequence numbers
    if (new_target_found) {
      upload_seq_num();
    }
    dmtcp_global_barrier("mana/comm-seq-round-finish");
    download_targets();
    // After downloading new targets, if all ranks agrees that local
    // sequence numbers are the same as new targets, we are done.
    check_seq_num(&converged, &new_target_found);
    if (!converged) {
      printf("rank %d not converged\n", g_world_rank);
      fflush(stdout);
      // Notify the user thread to continue and resume when a new
      // communication is found.
      sem_trywait(&user_thread_sem);
      sem_post(&user_thread_sem);
    }
    dmtcp_kvdb64(DMTCP_KVDB_INCRBY, "/mana/seq-num-states",
                 DB_CONVERGED, converged);
    dmtcp_global_barrier("mana/comm-seq-round");
    dmtcp_kvdb64_get("/mana/seq-num-states",
                     DB_CONVERGED, &all_converged);
  }
  while (in_cs) {
    // sleep(1);
  }
  #ifdef DEBUG_SEQ_NUM
  printf("rank %d sequence number algorithm completed\n", g_world_rank);
  fflush(stdout);
  #endif
}
