#ifndef SEQ_NUM_H
#define SEQ_NUM_H

#include <mpi.h>
#include <pthread.h>

// The main functions of the sequence number algorithm for MPI collectives
void commit_begin(MPI_Comm comm, const char *name);
void commit_finish(MPI_Comm comm);

// Forces the current process to synchronize with the coordinator in order to
// get to a globally safe state for checkpointing
void drainMpiCollectives();
void seq_num_init();
void seq_num_reset();

#endif // ifndef SEQ_NUM_H
