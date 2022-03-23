#pragma once
#ifndef MPI_VIRTUAL_IDS_H
#define MPI_VIRTUAL_IDS_H

#include <mpi.h>
#include <stdint.h>
#include <unordered_map>
#include "jassert.h"
#include "dmtcp.h"
#include "split_process.h"
#include "lower_half_api.h"

#define TABLE_SIZE 10000
#define MAX_TABLES 100

# define NEXT_FUNC(func)                                                       \
  ({                                                                           \
    static __typeof__(&MPI_##func)_real_MPI_## func =                          \
                                                (__typeof__(&MPI_##func)) - 1; \
    if (_real_MPI_ ## func == (__typeof__(&MPI_##func)) - 1) {                 \
      _real_MPI_ ## func = (__typeof__(&MPI_##func))pdlsym(MPI_Fnc_##func);    \
    }                                                                          \
    _real_MPI_ ## func;                                                        \
  })

#define VIRTUAL_TO_REAL_COMM(id) \
  (MPI_Comm) virtualToReal(&mpi_comm_table, id)
#define ADD_NEW_COMM(id) \
  (MPI_Comm) add_virtual_id(&mpi_comm_table, id)
#define REMOVE_OLD_COMM(id) \
  (MPI_Comm) remove_virtual_id(&mpi_comm_table, id)
#define UPDATE_COMM_MAP(v, r) \
  update_virtual_id(&mpi_comm_table, v, r)

#define VIRTUAL_TO_REAL_GROUP(id) \
  (MPI_Group) virtualToReal(&mpi_group_table, id)
#define ADD_NEW_GROUP(id) \
  (MPI_Group) add_virtual_id(&mpi_group_table, id)
#define REMOVE_OLD_GROUP(id) \
  (MPI_Group) remove_virtual_id(&mpi_group_table, id)
#define UPDATE_GROUP_MAP(v, r) \
  update_virtual_id(&mpi_group_table, v, r)

#define VIRTUAL_TO_REAL_TYPE(id) \
  (MPI_Datatype) virtualToReal(&mpi_type_table, id)
#define ADD_NEW_TYPE(id) \
  (MPI_Datatype) add_virtual_id(&mpi_type_table, id)
#define REMOVE_OLD_TYPE(id) \
  (MPI_Datatype) remove_virtual_id(&mpi_type_table, id)
#define UPDATE_TYPE_MAP(v, r) \
  update_virtual_id(&mpi_type_table, v, r)

#define VIRTUAL_TO_REAL_OP(id) \
  (MPI_Op) virtualToReal(&mpi_op_table, id)
#define ADD_NEW_OP(id) \
  (MPI_Op) add_virtual_id(&mpi_op_table, id)
#define REMOVE_OLD_OP(id) \
  (MPI_Op) remove_virtual_id(&mpi_op_table, id)
#define UPDATE_OP_MAP(v, r) \
  update_virtual_id(&mpi_op_table, v, r)

#define VIRTUAL_TO_REAL_COMM_KEYVAL(id) \
  (int) virtualToReal(&mpi_comm_keyval_table, id)
#define ADD_NEW_COMM_KEYVAL(id) \
  (int) add_virtual_id(&mpi_comm_keyval_table, id)
#define REMOVE_OLD_COMM_KEYVAL(id) \
  (int) remove_virtual_id(&mpi_comm_keyval_table, id)
#define UPDATE_COMM_KEYVAL_MAP(v, r) \
  update_virtual_id(&mpi_comm_keyval_table, v, r)

#define VIRTUAL_TO_REAL_REQUEST(id) \
  (MPI_Request) virtualToReal(&mpi_request_table, id)
#define ADD_NEW_REQUEST(id) \
  (MPI_Request) add_virtual_id(&mpi_request_table, id)
#define REMOVE_OLD_REQUEST(id) \
  (MPI_Request) remove_virtual_id(&mpi_request_table, id)
#define UPDATE_REQUEST_MAP(v, r) \
  update_virtual_id(&mpi_request_table, v, r)

typedef struct _id_table_slot_t {
  int in_use = 0;
  intptr_t real_id = 0;
} id_table_slot_t;

typedef struct _id_table_t {
  int slot_idx = 0;
  int count = 0;
  id_table_slot_t slots[TABLE_SIZE];
} id_table_t;

extern id_table_t mpi_comm_table;
extern id_table_t mpi_group_table;
extern id_table_t mpi_type_table;
extern id_table_t mpi_op_table;
extern id_table_t mpi_comm_keyval_table;
extern id_table_t mpi_request_table;

intptr_t add_virtual_id(id_table_t *table, intptr_t real);
intptr_t remove_virtual_id(id_table_t *table, intptr_t virt);
intptr_t virtualToReal(id_table_t *table, intptr_t virt);
void update_virtual_id(id_table_t *table, intptr_t virt, intptr_t real);

namespace dmtcp_mpi
{
  // FIXME: The new name should be: GlobalIdOfSimiliarComm
  class VirtualGlobalCommId {
    public:
      unsigned int createGlobalId(MPI_Comm comm) {
        if (comm == MPI_COMM_NULL) {
          return comm;
        }
        unsigned int gid = 0;
        int worldRank, commSize;
        int realComm = VIRTUAL_TO_REAL_COMM(comm);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(comm, &commSize);
        int rbuf[commSize];
        // FIXME: Use MPI_Group_translate_ranks instead of Allgather.
        // MPI_Group_translate_ranks only execute localy, so we can avoid
        // the cost of collective communication
        // FIXME: cray cc complains "catastrophic error" that can't find
        // split-process.h
#if 1
        DMTCP_PLUGIN_DISABLE_CKPT();
        JUMP_TO_LOWER_HALF(lh_info.fsaddr);
        NEXT_FUNC(Allgather)(&worldRank, 1, MPI_INT,
                             rbuf, 1, MPI_INT, realComm);
        RETURN_TO_UPPER_HALF();
        DMTCP_PLUGIN_ENABLE_CKPT();
#else
        MPI_Allgather(&worldRank, 1, MPI_INT, rbuf, 1, MPI_INT, comm);
#endif
        for (int i = 0; i < commSize; i++) {
          gid ^= hash(rbuf[i] + 1);
        }
#if 0
        printf("Global Comm ID %x:", gid);
        for (int i = 0; i < commSize; i++) {
          printf(" %d", rbuf[i]);
        }
        printf("\n");
#endif
        // FIXME: We assume the hash collision between communicators who
        // have different members is low.
        // FIXME: We want to prune virtual communicators to avoid long
        // restart time.
        // FIXME: In VASP we observed that for the same virtual communicator
        // (adding 1 to each new communicator with the same rank members),
        // the virtual group can change over time, using:
        // virtual Comm -> real Comm -> real Group -> virtual Group
        // We don't understand why since vasp does not seem to free groups.
#if 0
        // FIXME: Some code can create new communicators during execution,
        // and so hash conflict may occur later.
        // if the new gid already exists in the map, add one and test again
        while (1) {
          bool found = false;
          for (std::pair<MPI_Comm, unsigned int> idPair : globalIdTable) {
            if (idPair.second == gid) {
              found = true;
              break;
            }
          }
          if (found) {
            gid++;
          } else {
            break;
          }
        }
#endif
        globalIdTable[comm] = gid;
        return gid;
      }

      unsigned int getGlobalId(MPI_Comm comm) {
        std::unordered_map<MPI_Comm, unsigned int>::iterator it =
          globalIdTable.find(comm);
        JASSERT(it != globalIdTable.end())(comm)
          .Text("Can't find communicator in the global id table");
        return it->second;
      }

      static VirtualGlobalCommId& instance() {
        static VirtualGlobalCommId _vGlobalId;
        return _vGlobalId;
      }

    private:
      VirtualGlobalCommId()
      {
          globalIdTable[MPI_COMM_NULL] = MPI_COMM_NULL;
          globalIdTable[MPI_COMM_WORLD] = MPI_COMM_WORLD;
      }

      void printMap(bool flag = false) {
        for (std::pair<MPI_Comm, int> idPair : globalIdTable) {
          if (flag) {
            printf("virtual comm: %x, real comm: %x, global id: %x\n",
                   idPair.first, VIRTUAL_TO_REAL_COMM(idPair.first),
                   idPair.second);
            fflush(stdout);
          } else {
            JTRACE("Print global id mapping")((void*) (uint64_t) idPair.first)
                      ((void*) (uint64_t) VIRTUAL_TO_REAL_COMM(idPair.first))
                      ((void*) (uint64_t) idPair.second);
          }
        }
      }
      // from https://stackoverflow.com/questions/664014/
      // what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
      int hash(int i) {
        return i * 2654435761 % ((unsigned long)1 << 32);
      }
      std::unordered_map<MPI_Comm, unsigned int> globalIdTable;
  };
};  // namespace dmtcp_mpi

#endif // ifndef MPI_VIRTUAL_IDS_H
