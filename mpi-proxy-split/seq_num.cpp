#include <mpi.h>

#include <map>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>

#include "jassert.h"
#include "kvdb.h"
#include "seq_num.h"
#include "mpi_nextfunc.h"
#include "virtual-ids.h"
#include "record-replay.h"

using namespace dmtcp_mpi;
using dmtcp::kvdb::KVDBRequest;
using dmtcp::kvdb::KVDBResponse;

// #define DEBUG_SEQ_NUM

extern int g_world_rank;
extern int g_world_size;
// Global communicator for MANA internal use
MPI_Comm g_world_comm;
extern int p2p_deterministic_skip_save_request;
volatile bool ckpt_pending;
int converged;
volatile phase_t current_phase;
unsigned int comm_gid;
int num_converged;
reset_type_t reset_type;

pthread_mutex_t seq_num_lock;
std::map<unsigned int, unsigned long> blocking_seq_num;
std::map<unsigned int, unsigned long> blocking_target;
std::map<unsigned int, unsigned long> nonblocking_seq_num;
std::map<unsigned int, unsigned long> nonblocking_target;
std::vector<MPI_Request*> active_requests;

constexpr const char *blocking_comm_seq_max_db =
  "/plugin/MANA/blocking-comm-seq-max";
constexpr const char *nonblocking_comm_seq_max_db =
  "/plugin/MANA/nonblocking-comm-seq-max";

void seq_num_init() {
  ckpt_pending = false;
  pthread_mutex_init(&seq_num_lock, NULL);
}

void seq_num_reset(reset_type_t type) {
  ckpt_pending = false;
}

void seq_num_destroy() {
  pthread_mutex_destroy(&seq_num_lock);
}

void print_seq_nums() {
  unsigned int comm_id;
  unsigned long seq;
  printf("blocking\n");
  for (comm_seq_pair_t pair : nonblocking_seq_num) {
    comm_id = pair.first;
    seq = pair.second;
    printf("%d, %u, %lu\n", g_world_rank, comm_id, seq);
  }
  printf("nonblocking\n");
  for (comm_seq_pair_t pair : nonblocking_seq_num) {
    comm_id = pair.first;
    seq = pair.second;
    printf("%d, %u, %lu\n", g_world_rank, comm_id, seq);
  }
  fflush(stdout);
}

int check_seq_nums(bool exclusive) {
  unsigned int comm_id;
  unsigned long seq_num;
  int target_reached = 1;
  for (comm_seq_pair_t pair : blocking_seq_num) {
    comm_id = pair.first;
    seq_num = pair.second;
    if (exclusive) {
      if (blocking_target[comm_id] + 1 > seq_num) {
        target_reached = 0;
        break;
      }
    } else {
      if (blocking_target[comm_id] > seq_num) {
        target_reached = 0;
        break;
      }
    }
  }
  for (comm_seq_pair_t pair : nonblocking_seq_num) {
    comm_id = pair.first;
    seq_num = pair.second;
    if (exclusive) {
      if (nonblocking_target[comm_id] + 1 > seq_num) {
        target_reached = 0;
        break;
      }
    } else {
      if (nonblocking_target[comm_id] > seq_num) {
        target_reached = 0;
        break;
      }
    }
  }
  return target_reached;
}

int twoPhaseCommit(MPI_Comm comm,
                   std::function<int(void)>doRealCollectiveComm) {
  if (!MPI_LOGGING() || comm == MPI_COMM_NULL) {
    return doRealCollectiveComm(); // lambda function: already captured args
  }

  commit_begin(comm, false, true);
  int retval = doRealCollectiveComm();
  commit_finish(comm, false, true);
  return retval;
}

void seq_num_broadcast(MPI_Comm comm, unsigned long new_target) {
  unsigned int comm_gid = VirtualGlobalCommId::instance().getGlobalId(comm);
  unsigned long msg[2] = {comm_gid, new_target};
  int comm_size;
  int comm_rank;
  int world_rank;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Group world_group, local_group;
  MPI_Comm real_local_comm = VIRTUAL_TO_REAL_COMM(comm);
  MPI_Comm real_world_comm = VIRTUAL_TO_REAL_COMM(g_world_comm);
  JUMP_TO_LOWER_HALF(lh_info.fsaddr);
  NEXT_FUNC(Comm_group)(real_world_comm, &world_group);
  NEXT_FUNC(Comm_group)(real_local_comm, &local_group);
  RETURN_TO_UPPER_HALF();
  for (int i = 0; i < comm_size; i++) {
    if (i != comm_rank) {
      JUMP_TO_LOWER_HALF(lh_info.fsaddr);
      NEXT_FUNC(Group_translate_ranks)(local_group, 1, &i,
                world_group, &world_rank);
      NEXT_FUNC(Send)(&msg, 2, MPI_UNSIGNED_LONG, world_rank,
                      0, real_world_comm);
      RETURN_TO_UPPER_HALF();
    }
  }
  JUMP_TO_LOWER_HALF(lh_info.fsaddr);
  NEXT_FUNC(Group_free)(&world_group);
  NEXT_FUNC(Group_free)(&local_group);
  RETURN_TO_UPPER_HALF();
}

void commit_begin(MPI_Comm comm, bool passthrough, bool blocking) {
  if (mana_state == RESTART_REPLAY || comm == MPI_COMM_NULL) {
    return;
  }
  comm_seq_t &seq_num = blocking ? blocking_seq_num : nonblocking_seq_num;
  comm_seq_t &target = blocking ? blocking_target : nonblocking_target;
  while (ckpt_pending && check_seq_nums(passthrough)) {
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, g_world_comm, &flag, &status);
    if (flag) {
      unsigned long new_target[2];
      MPI_Comm real_world_comm = VIRTUAL_TO_REAL_COMM(g_world_comm);
      JUMP_TO_LOWER_HALF(lh_info.fsaddr);
      NEXT_FUNC(Recv)(&new_target, 2, MPI_UNSIGNED_LONG,
          status.MPI_SOURCE, status.MPI_TAG, real_world_comm,
          MPI_STATUS_IGNORE);
      RETURN_TO_UPPER_HALF();
      unsigned int updated_comm = (unsigned int) new_target[0];
      unsigned long updated_target = new_target[1];
      std::map<unsigned int, unsigned long>::iterator it =
        target.find(updated_comm);
      if (it != target.end() && it->second < updated_target) {
        target[updated_comm] = updated_target;
      }
    }
  }
  pthread_mutex_lock(&seq_num_lock);
  current_phase = IN_CS;
  unsigned int comm_gid = VirtualGlobalCommId::instance().getGlobalId(comm);
  seq_num[comm_gid]++;
  pthread_mutex_unlock(&seq_num_lock);
  if (ckpt_pending && seq_num[comm_gid] > target[comm_gid]) {
    target[comm_gid] = seq_num[comm_gid];
    seq_num_broadcast(comm, seq_num[comm_gid]);
  }
}

void commit_finish(MPI_Comm comm, bool passthrough, bool blocking) {
  if (mana_state == RESTART_REPLAY) {
    return;
  }
  current_phase = IS_READY;
  if (passthrough) {
    return;
  }
  comm_seq_t &seq_num = blocking ? blocking_seq_num : nonblocking_seq_num;
  comm_seq_t &target = blocking ? blocking_target : nonblocking_target;
  while (ckpt_pending && check_seq_nums(false)) {
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, g_world_comm, &flag, &status);
    if (flag) {
      unsigned long new_target[2];
      MPI_Comm real_world_comm = VIRTUAL_TO_REAL_COMM(g_world_comm);
      JUMP_TO_LOWER_HALF(lh_info.fsaddr);
      NEXT_FUNC(Recv)(&new_target, 2, MPI_UNSIGNED_LONG,
          status.MPI_SOURCE, status.MPI_TAG, real_world_comm,
          MPI_STATUS_IGNORE);
      RETURN_TO_UPPER_HALF();
      unsigned int updated_comm = (unsigned int) new_target[0];
      unsigned long updated_target = new_target[1];
      std::map<unsigned int, unsigned long>::iterator it =
        target.find(updated_comm);
      if (it != target.end() && it->second < updated_target) {
        target[updated_comm] = updated_target;
#ifdef DEBUG_SEQ_NUM
        printf("rank %d received new target comm %u seq %lu target %lu\n",
            g_world_rank, updated_comm, seq_num[updated_comm], updated_target);
        fflush(stdout);
#endif
      }
    }
  }
}

void upload_seq_num() {
  for (comm_seq_pair_t pair : blocking_seq_num) {
    dmtcp::string comm_id_str(jalib::XToString(pair.first));
    unsigned int seq = pair.second;
    JASSERT(dmtcp::kvdb::request64(KVDBRequest::MAX, blocking_comm_seq_max_db,
                                   comm_id_str, seq) == KVDBResponse::SUCCESS);
  }
  for (comm_seq_pair_t pair : nonblocking_seq_num) {
    dmtcp::string comm_id_str(jalib::XToString(pair.first));
    unsigned int seq = pair.second;
    JASSERT(dmtcp::kvdb::request64(KVDBRequest::MAX,
                                   nonblocking_comm_seq_max_db,
                                   comm_id_str, seq) == KVDBResponse::SUCCESS);
  }
}

void download_targets() {
  int64_t max_seq = 0;
  unsigned int comm_id;
  for (comm_seq_pair_t pair : blocking_seq_num) {
    comm_id = pair.first;
    dmtcp::string comm_id_str(jalib::XToString(pair.first));
    JASSERT(dmtcp::kvdb::get64(blocking_comm_seq_max_db, comm_id_str, &max_seq) ==
            KVDBResponse::SUCCESS);
    blocking_target[comm_id] = max_seq;
  }
  for (comm_seq_pair_t pair : nonblocking_seq_num) {
    comm_id = pair.first;
    dmtcp::string comm_id_str(jalib::XToString(pair.first));
    JASSERT(dmtcp::kvdb::get64(nonblocking_comm_seq_max_db, comm_id_str, &max_seq) ==
            KVDBResponse::SUCCESS);
    nonblocking_target[comm_id] = max_seq;
  }
}

void share_seq_nums() {
  upload_seq_num();
  dmtcp_global_barrier("mana/share-seq-num");
  download_targets();
}

void drain_mpi_collective() {
  int round_num = 0;
  int64_t num_converged = 0;
  int64_t in_cs = 0;
  pthread_mutex_lock(&seq_num_lock);
  ckpt_pending = true;
  share_seq_nums();
  pthread_mutex_unlock(&seq_num_lock);
  while (1) {
    char key[32] = {0};
    char barrier_id[32] = {0};
    constexpr const char *cs_id = "/plugin/MANA/CRITICAL-SECTION";
    constexpr const char *converge_id = "/plugin/MANA/CONVERGE";
    sprintf(barrier_id, "MANA-PRESUSPEND-%06d", round_num);
    sprintf(key, "round-%06d", round_num);

    JASSERT(dmtcp::kvdb::request64(KVDBRequest::INCRBY, converge_id, key,
                                   check_seq_nums(false)) ==
            KVDBResponse::SUCCESS);
    JASSERT(dmtcp::kvdb::request64(KVDBRequest::OR, cs_id, key,
                                   current_phase == IN_CS) ==
            KVDBResponse::SUCCESS);

    dmtcp_global_barrier(barrier_id);

    JASSERT(dmtcp::kvdb::get64(converge_id, key, &num_converged) ==
            KVDBResponse::SUCCESS);
    JASSERT(dmtcp::kvdb::get64(cs_id, key, &in_cs) == KVDBResponse::SUCCESS);

    if (in_cs == 0 && num_converged == g_world_size) {
      break;
    }
    round_num++;
  }
  while (!active_requests.empty()) {
    std::vector<MPI_Request*>::iterator it = active_requests.begin();
    while (it != active_requests.end()) {
      int flag = 0;
      // FIXME: need to handle MPI status in user's program.
      MPI_Test(*it, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        active_requests.erase(it);
      } else {
        it++;
      }
    }
  }
}
