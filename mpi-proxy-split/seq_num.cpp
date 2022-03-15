#include <mpi.h>

#include <unordered_map>
#include <pthread.h>

#include "jassert.h"
#include "seq_num.h"
#include "mpi_nextfunc.h"
#include "virtual-ids.h"

using namespace dmtcp_mpi;

#define DB_NONE_IN_CS 0
#define DB_CONVERGED 1

// #define DEBUG_SEQ_NUM

extern int g_world_rank;
unsigned int global_comm_id;
bool ckpt_pending;
bool first_time_blocking;
bool freepass;
bool should_update_db;
pthread_mutex_t block_cv_lock;
pthread_cond_t block_cv;
pthread_mutex_t update_cv_lock;
pthread_cond_t update_cv;
std::map<unsigned int, unsigned int> seq_num;
std::map<unsigned int, unsigned int> target_seq_num;
typedef std::pair<unsigned int, unsigned int> comm_seq_pair_t;

void release_user_thread();
void update_db();

void seq_num_init() {
#ifdef DEBUG_SEQ_NUM
  printf("seq_num initialized\n");
  fflush(stdout);
#endif
  ckpt_pending = false;
  first_time_blocking = true;
  freepass = false;
  current_state = NOT_IN_CS;
  pthread_mutex_init(&block_cv_lock, NULL);
  pthread_cond_init(&block_cv, NULL);
  pthread_mutex_init(&update_cv_lock, NULL);
  pthread_cond_init(&update_cv, NULL);
}

void seq_num_reset() {
  freepass = false;
  first_time_blocking = true;
  ckpt_pending = false;
  for (comm_seq_pair_t i : seq_num) {
    i.second = 0;
  }

  // realse user thread
  pthread_mutex_lock(&block_cv_lock);
  pthread_cond_signal(&block_cv);
  pthread_mutex_unlock(&block_cv_lock);
}

void seq_num_destroy() {
  pthread_mutex_destroy(&block_cv_lock);
  pthread_cond_destroy(&block_cv);
  pthread_mutex_destroy(&update_cv_lock);
  pthread_cond_destroy(&update_cv);
}

// FIXME: Combine two lock/unlock functions into one
void block_user_thread() {
#ifdef DEBUG_SEQ_NUM
  printf("rank %d user thread blocked\n", g_world_rank);
  fflush(stdout);
#endif
  pthread_mutex_lock(&block_cv_lock);
  freepass = false;
  while (ckpt_pending && !freepass) {
    pthread_cond_wait(&block_cv, &block_cv_lock);
  }
  pthread_mutex_unlock(&block_cv_lock);
#ifdef DEBUG_SEQ_NUM
  printf("rank %d user thread released\n", g_world_rank);
  fflush(stdout);
#endif
}

void release_user_thread() {
#ifdef DEBUG_SEQ_NUM
  printf("rank %d user thread releasing\n", g_world_rank);
  fflush(stdout);
#endif
  pthread_mutex_lock(&block_cv_lock);
  freepass = true;
  pthread_cond_signal(&block_cv);
  pthread_mutex_unlock(&block_cv_lock);
}

void wait_to_update_db() {
#ifdef DEBUG_SEQ_NUM
  printf("rank %d ckpt thread blocked\n", g_world_rank);
  fflush(stdout);
#endif
  pthread_mutex_lock(&update_cv_lock);
  should_update_db = false;
  while (!should_update_db) {
    pthread_cond_wait(&update_cv, &update_cv_lock);
  }
  pthread_mutex_unlock(&update_cv_lock);
#ifdef DEBUG_SEQ_NUM
  printf("rank %d ckpt thread released\n", g_world_rank);
  fflush(stdout);
#endif
}

void update_db() {
#ifdef DEBUG_SEQ_NUM
  printf("rank %d ckpt thread releasing\n", g_world_rank);
  fflush(stdout);
#endif
  pthread_mutex_lock(&update_cv_lock);
  should_update_db = true;
  pthread_cond_signal(&update_cv);
  pthread_mutex_unlock(&update_cv_lock);
}

void commit_begin(MPI_Comm comm, const char *name) {
  global_comm_id = VirtualGlobalCommId::instance().getGlobalId(comm);
  if (ckpt_pending) {
    bool target_reached = true;
    bool new_target_found = false;
    for (comm_seq_pair_t pair : seq_num) {
      int comm_id = pair.first;
      int seq = pair.second;
      if (target_seq_num[comm_id] < seq) {
        new_target_found = true;
        break;
      } else if (target_seq_num[comm_id] > seq) {
        target_reached = false;
        break;
      }
    }
    if (target_reached || new_target_found) {
      update_db();
      block_user_thread();
    }
  }
  seq_num[global_comm_id]++;
#ifdef DEBUG_SEQ_NUM
  printf("rank %d %s comm %x seq %u\n", g_world_rank, name, 
         global_comm_id, seq_num[global_comm_id]);
  fflush(stdout);
#endif
}

void drainMpiCollectives() {
  int64_t converged;
  int64_t max_seq;
  unsigned int comm_id;
  unsigned int seq;
  ckpt_pending = true;

  while (1) {
    // Reset golbal flags
    converged = 1;
    if (g_world_rank == 0) {
      dmtcp_kvdb64(DMTCP_KVDB_OR, "/mana/seq-num-global-states", DB_CONVERGED, 1);
    }

    for (comm_seq_pair_t pair : seq_num) {
      comm_id = pair.first;
      seq = pair.second;
      dmtcp_kvdb64(DMTCP_KVDB_MAX, "/mana/comm-seq-max", comm_id, seq);
    }
    dmtcp_global_barrier("mana/comm-seq-round");

    for (comm_seq_pair_t pair : seq_num) {
      comm_id = pair.first;
      seq = pair.second;
      dmtcp_kvdb64_get("/mana/comm-seq-max", comm_id, &max_seq);

      // Update target sequence number hashmap.
      target_seq_num[comm_id] = max_seq;

      if (max_seq > seq) {
        // This rank is behind.
        converged = 0;
        printf("SEQ rank %d comm_id %x man_seq %ld seq %u\n", g_world_rank, comm_id, max_seq, seq);
        fflush(stdout);
      }
    }
    if (!converged) {
        dmtcp_kvdb64(DMTCP_KVDB_AND, "/mana/seq-num-states",
                     DB_CONVERGED, converged);
        // Give free pass to the user thread.
        release_user_thread();
        wait_to_update_db();
    }
    dmtcp_global_barrier("mana/comm-seq-round");

    dmtcp_kvdb64_get("/mana/seq-num-global-states", DB_CONVERGED, &converged);
    if (converged) {
      // Safe to checkpoint
      break;
    }
  }
 #ifdef DEBUG_SEQ_NUM
  printf("Sequence Number algorithm completed\n");
  fflush(stdout);
 #endif
}
