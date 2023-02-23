#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MSG_SIZE 10

int main(int argc, char **argv) {
  int rank, size;
  MPI_Request req[2];
  int *buf = (int*) malloc (sizeof(int) * MSG_SIZE);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    fprintf(stderr, "This test programs requires 2 MPI processes.\n");
    MPI_Abort();
  }
  
  if (rank == 0) {
    MPI_Isend(buf, MSG_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(buf, MSG_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD, &req[1]);
  } else {
    MPI_Irecv(buf, MSG_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(buf, MSG_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &req[1]);
  }
  MPI_Waitall(req, 2, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
}

