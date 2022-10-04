#include <mpi.h>

#include <unordered_map>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>

#include "jassert.h"
#include "seq_num.h"
#include "mpi_nextfunc.h"
#include "virtual-ids.h"
#include "record-replay.h"

using namespace dmtcp_mpi;

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

std::unordered_map<MPI_Comm, unsigned long> seq_num;
std::unordered_map<MPI_Comm, unsigned long> target;
std::unordered_map<MPI_Comm, unsigned int> global_id_table = {
  {MPI_COMM_NULL, (unsigned int) MPI_COMM_NULL},
  {MPI_COMM_WORLD, (unsigned int) MPI_COMM_WORLD},
};
std::unordered_map<unsigned int, MPI_Comm> global_id_to_comm_table = {
  {(unsigned int) MPI_COMM_NULL, MPI_COMM_NULL},
  {(unsigned int) MPI_COMM_WORLD, MPI_COMM_WORLD},
};
typedef std::pair<MPI_Comm, unsigned long> comm_seq_pair_t;

// from https://stackoverflow.com/questions/664014/
// what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
int hash(int i) {
  return i * 2654435761 % ((unsigned long)1 << 32);
}

unsigned int create_global_id(MPI_Comm comm) {
  unsigned int gid = 0;
  int commSize;
  MPI_Group world_group, local_group;
  int rbuf[commSize];
  // FIXME: cray cc complains "catastrophic error" that can't find
  // split-process.h

  // Use MPI_Group_translate_ranks to translate local rank numbers
  // to global rank numbers. Then call the hash function on world
  // rank numbers to get the global communicator ID.
  MPI_Comm real_local_comm = VIRTUAL_TO_REAL_COMM(comm);
  MPI_Comm real_world_comm = VIRTUAL_TO_REAL_COMM(g_world_comm);
  DMTCP_PLUGIN_DISABLE_CKPT();
  JUMP_TO_LOWER_HALF(lh_info.fsaddr);
  NEXT_FUNC(Comm_size)(real_local_comm, &commSize);
  NEXT_FUNC(Comm_group)(real_world_comm, &world_group);
  NEXT_FUNC(Comm_group)(real_local_comm, &local_group);
  for (int i = 0; i < commSize; i++) {
    NEXT_FUNC(Group_translate_ranks)(local_group, 1, &i,
                                     world_group, &rbuf[i]);
  }
  RETURN_TO_UPPER_HALF();
  DMTCP_PLUGIN_ENABLE_CKPT();
  for (int i = 0; i < commSize; i++) {
    gid ^= hash(rbuf[i] + 1);
  }
  // FIXME: We assume the hash collision between communicators who
  // have different members is low.
  // FIXME: In VASP we observed that for the same virtual communicator
  // (adding 1 to each new communicator with the same rank members),
  // the virtual group can change over time, using:
  // virtual Comm -> real Comm -> real Group -> virtual Group
  // We don't understand why since vasp does not seem to free groups.
  global_id_table[comm] = gid;
  global_id_to_comm_table[gid] = comm;
  return gid;
}

unsigned int get_global_id(MPI_Comm comm) {
  unsigned int gid;
  std::unordered_map<MPI_Comm, unsigned int>::iterator it =
    global_id_table.find(comm);
  if (it != global_id_table.end()) {
    gid = it->second;
  } else {
    gid = create_global_id(comm);
    global_id_table[comm] = gid;
  }
  return gid;
}

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

int print_seq_nums() {
  MPI_Comm comm_id;
  unsigned long seq;
  int target_reached = 1;
  for (comm_seq_pair_t pair : seq_num) {
    comm_id = pair.first;
    seq = pair.second;
    printf("%d, %u, %lu\n", g_world_rank, comm_id, seq);
  }
  fflush(stdout);
  return target_reached;
}

int check_seq_nums(bool exclusive) {
  MPI_Comm comm;
  unsigned long seq;
  int target_reached = 1;
  for (comm_seq_pair_t pair : seq_num) {
    comm = pair.first;
    seq = pair.second;
    if (exclusive) {
      if (target[comm] + 1 > seq_num[comm]) {
        target_reached = 0;
        break;
      }
    } else {
      if (target[comm] > seq_num[comm]) {
        target_reached = 0;
        break;
      }
    }
  }
  return target_reached;
}

void seq_num_broadcast(MPI_Comm comm, unsigned long new_target) {
  unsigned int comm_gid = get_global_id(comm);
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
#ifdef DEBUG_SEQ_NUM
      printf("rank %d sending to rank %d new target comm %u seq %lu target %lu\n",
             g_world_rank, world_rank, comm_gid, seq_num[comm_gid], new_target);
      fflush(stdout);
#endif
    }
  }
  JUMP_TO_LOWER_HALF(lh_info.fsaddr);
  NEXT_FUNC(Group_free)(&world_group);
  NEXT_FUNC(Group_free)(&local_group);
  RETURN_TO_UPPER_HALF();
}

void commit_begin(MPI_Comm comm, bool passthrough) {
  if (mana_state == RESTART_REPLAY || comm == MPI_COMM_NULL) {
    return;
  }
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
      MPI_Comm updated_comm =
        global_id_to_comm_table[(unsigned int) new_target[0]];
      unsigned long updated_target = new_target[1];
      std::unordered_map<MPI_Comm, unsigned long>::iterator it =
        target.find(updated_comm);
      if (it != target.end() && it->second < updated_target) {
        target[updated_comm] = updated_target;
#ifdef DEBUG_SEQ_NUM
        printf("rank %d received new target comm %u seq %lu target %lu\n",
               g_world_rank, updated_comm, seq_num[updated_comm],
               updated_target);
        fflush(stdout);
#endif
      }
    }
  }
  pthread_mutex_lock(&seq_num_lock);
  current_phase = IN_CS;
  seq_num[comm]++;
  pthread_mutex_unlock(&seq_num_lock);
#ifdef DEBUG_SEQ_NUM
  // print_seq_nums();
#endif
  if (ckpt_pending && seq_num[comm] > target[comm]) {
    target[comm] = seq_num[comm];
    seq_num_broadcast(comm, seq_num[comm]);
  }
}

void commit_finish(MPI_Comm comm, bool passthrough) {
  if (mana_state == RESTART_REPLAY) {
    return;
  }
  current_phase = IS_READY;
  if (passthrough) {
    return;
  }
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
      MPI_Comm updated_comm =
        global_id_to_comm_table[(unsigned int) new_target[0]];
      unsigned long updated_target = new_target[1];
      std::unordered_map<MPI_Comm, unsigned long>::iterator it =
        target.find(updated_comm);
      if (it != target.end() && it->second < updated_target) {
        target[updated_comm] = updated_target;
#ifdef DEBUG_SEQ_NUM
        printf("rank %d received new target comm %u seq %lu target %lu\n",
            g_world_rank, updated_comm, seq_num[updated_comm],
            updated_target);
        fflush(stdout);
#endif
      }
    }
  }
}

void upload_seq_num() {
  MPI_Comm comm;
  unsigned int comm_id;
  unsigned int seq;
  for (comm_seq_pair_t pair : seq_num) {
    comm = pair.first;
    comm_id = get_global_id(comm);
    seq = pair.second;
    dmtcp_kvdb64(DMTCP_KVDB_MAX, "/mana/comm-seq-max", comm_id, seq);
  }
}

void download_targets(std::unordered_map<MPI_Comm, unsigned long> &target) {
  int64_t max_seq = 0;
  MPI_Comm comm;
  unsigned int comm_id;
  for (comm_seq_pair_t pair : seq_num) {
    comm = pair.first;
    comm_id = get_global_id(comm);
    dmtcp_kvdb64_get("/mana/comm-seq-max", comm_id, &max_seq);
    target[comm] = max_seq;
  }
}

void share_seq_nums(std::unordered_map<MPI_Comm, unsigned long> &target) {
  upload_seq_num();
  dmtcp_global_barrier("mana/share-seq-num");
  download_targets(target);
}

void drain_mpi_collective() {
  int round_num = 0;
  int64_t num_converged = 0;
  int64_t in_cs = 0;
  pthread_mutex_lock(&seq_num_lock);
  ckpt_pending = true;
  share_seq_nums(target);
  pthread_mutex_unlock(&seq_num_lock);
  while (1) {
    dmtcp::string barrierId = "MANA-PRESUSPEND-" + jalib::XToString(round_num);
    dmtcp::string csId = "MANA-CRITICAL-SECTION-" + jalib::XToString(round_num);
    dmtcp::string convergeId = "MANA-CONVERGE-" + jalib::XToString(round_num);
    dmtcp_kvdb64(DMTCP_KVDB_INCRBY, convergeId.c_str(), 0, check_seq_nums(false));
    dmtcp_kvdb64(DMTCP_KVDB_OR, csId.c_str(), 0, current_phase == IN_CS);
    dmtcp_global_barrier(barrierId.c_str());
    dmtcp_kvdb64_get(convergeId.c_str(), 0, &num_converged);
    dmtcp_kvdb64_get(csId.c_str(), 0, &in_cs);
    if (in_cs == 0 && num_converged == g_world_size) {
      break;
    }
    round_num++;
  }
}
