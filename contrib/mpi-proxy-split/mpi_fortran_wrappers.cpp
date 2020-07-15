#include <mpi.h>
#include "dmtcp.h"
#include "jassert.h"
EXTERNC int mpi_init_ (int* argc,  char*** argv, int *ierr) {
  *ierr = MPI_Init(argc, argv);
  return *ierr;
}

EXTERNC int mpi_finalize_ (int* ierr) {
  *ierr = MPI_Finalize();
  return *ierr;
}

EXTERNC int mpi_finalized_ (int* flag, int *ierr) {
  *ierr = MPI_Finalized(flag);
  return *ierr;
}

EXTERNC int mpi_get_processor_name_ (char* name,  int* resultlen, int *ierr) {
  *ierr = MPI_Get_processor_name(name, resultlen);
  return *ierr;
}

EXTERNC double mpi_wtime_ () {
  return MPI_Wtime();
}

EXTERNC int mpi_initialized_ (int* flag, int *ierr) {
  *ierr = MPI_Initialized(flag);
  return *ierr;
}

EXTERNC int mpi_init_thread_ (int* argc,  char*** argv,  int* required,  int* provided, int *ierr) {
  *ierr = MPI_Init_thread(argc, argv, *required, provided);
  return *ierr;
}

EXTERNC int mpi_get_count_ (const MPI_Status* status,  MPI_Datatype* datatype,  int* count, int *ierr) {
  *ierr = MPI_Get_count(status, *datatype, count);
  return *ierr;
}

EXTERNC int mpi_bcast_ (void* buffer,  int* count,  MPI_Datatype* datatype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Bcast(buffer, *count, *datatype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_barrier_ (MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Barrier(*comm);
  return *ierr;
}

EXTERNC int mpi_allreduce_ (const void* sendbuf,  void* recvbuf,  int* count,  MPI_Datatype* datatype,  MPI_Op* op,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Allreduce(sendbuf, recvbuf, *count, *datatype, *op, *comm);
  return *ierr;
}

EXTERNC int mpi_reduce_ (const void* sendbuf,  void* recvbuf,  int* count,  MPI_Datatype* datatype,  MPI_Op* op,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Reduce(sendbuf, recvbuf, *count, *datatype, *op, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_alltoall_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Alltoall(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *comm);
  return *ierr;
}

EXTERNC int mpi_alltoallv_ (const void* sendbuf,  const int* sendcounts,  const int* sdispls,  MPI_Datatype* sendtype,  void* recvbuf,  const int* recvcounts,  const int* rdispls,  MPI_Datatype* recvtype,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Alltoallv(sendbuf, sendcounts, sdispls, *sendtype, recvbuf, recvcounts, rdispls, *recvtype, *comm);
  return *ierr;
}

EXTERNC int mpi_allgather_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Allgather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *comm);
  return *ierr;
}

EXTERNC int mpi_allgatherv_ (const void*  sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  const int* recvcount,  const int* displs,  MPI_Datatype* recvtype,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Allgatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcount, displs, *recvtype, *comm);
  return *ierr;
}

EXTERNC int mpi_gather_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Gather(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_gatherv_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  const int* recvcounts,  const int* displs,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Gatherv(sendbuf, *sendcount, *sendtype, recvbuf, recvcounts, displs, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_scatter_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Scatter(sendbuf, *sendcount, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_scatterv_ (const void* sendbuf,  const int* sendcounts,  const int* displs,  MPI_Datatype* sendtype,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* root,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Scatterv(sendbuf, sendcounts, displs, *sendtype, recvbuf, *recvcount, *recvtype, *root, *comm);
  return *ierr;
}

EXTERNC int mpi_scan_ (const void* sendbuf,  void* recvbuf,  int* count,  MPI_Datatype* datatype,  MPI_Op* op,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Scan(sendbuf, recvbuf, *count, *datatype, *op, *comm);
  return *ierr;
}

EXTERNC int mpi_comm_size_ (MPI_Comm* comm,  int* world_size, int *ierr) {
  *ierr = MPI_Comm_size(*comm, world_size);
  return *ierr;
}

EXTERNC int mpi_comm_rank_ (MPI_Group* group,  int* world_rank, int *ierr) {
  *ierr = MPI_Comm_rank(*group, world_rank);
  return *ierr;
}

EXTERNC int mpi_abort_ (MPI_Comm* comm,  int* errorcode, int *ierr) {
  *ierr = MPI_Abort(*comm, *errorcode);
  return *ierr;
}

EXTERNC int mpi_comm_split_ (MPI_Comm* comm,  int* color,  int* key,  MPI_Comm* newcomm, int *ierr) {
  *ierr = MPI_Comm_split(*comm, *color, *key, newcomm);
  return *ierr;
}

EXTERNC int mpi_comm_dup_ (MPI_Comm* comm,  MPI_Comm* newcomm, int *ierr) {
  *ierr = MPI_Comm_dup(*comm, newcomm);
  return *ierr;
}

EXTERNC int mpi_comm_create_ (MPI_Comm* comm,  MPI_Group* group,  MPI_Comm* newcomm, int *ierr) {
  *ierr = MPI_Comm_create(*comm, *group, newcomm);
  return *ierr;
}

EXTERNC int mpi_comm_compare_ (MPI_Comm* comm1,  MPI_Comm* comm2,  int* result, int *ierr) {
  *ierr = MPI_Comm_compare(*comm1, *comm2, result);
  return *ierr;
}

EXTERNC int mpi_comm_free_ (MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Comm_free(comm);
  return *ierr;
}

EXTERNC int mpi_comm_set_errhandler_ (MPI_Comm* comm,  MPI_Errhandler* errhandler, int *ierr) {
  *ierr = MPI_Comm_set_errhandler(*comm, *errhandler);
  return *ierr;
}

EXTERNC int mpi_topo_test_ (MPI_Comm* comm,  int* status, int *ierr) {
  *ierr = MPI_Topo_test(*comm, status);
  return *ierr;
}

EXTERNC int mpi_comm_split_type_ (MPI_Comm* comm,  int* split_type,  int* key,  MPI_Info* info,  MPI_Comm* newcomm, int *ierr) {
  *ierr = MPI_Comm_split_type(*comm, *split_type, *key, *info, newcomm);
  return *ierr;
}

EXTERNC int mpi_attr_get_ (MPI_Comm* comm,  int* keyval,  void* attribute_val,  int* flag, int *ierr) {
  *ierr = MPI_Attr_get(*comm, *keyval, attribute_val, flag);
  return *ierr;
}

EXTERNC int mpi_attr_delete_ (MPI_Comm* comm,  int* keyval, int *ierr) {
  *ierr = MPI_Attr_delete(*comm, *keyval);
  return *ierr;
}

EXTERNC int mpi_attr_put_ (MPI_Comm* comm,  int* keyval,  void* attribute_val, int *ierr) {
  *ierr = MPI_Attr_put(*comm, *keyval, attribute_val);
  return *ierr;
}

EXTERNC int mpi_cart_coords_ (MPI_Comm* comm,  int* rank,  int* maxdims,  int* coords, int *ierr) {
  *ierr = MPI_Cart_coords(*comm, *rank, *maxdims, coords);
  return *ierr;
}

EXTERNC int mpi_cart_create_ (MPI_Comm* old_comm,  int* ndims,  const int* dims,  const int* periods,  int* reorder,  MPI_Comm* comm_cart, int *ierr) {
  *ierr = MPI_Cart_create(*old_comm, *ndims, dims, periods, *reorder, comm_cart);
  return *ierr;
}

EXTERNC int mpi_cart_get_ (MPI_Comm* comm,  int* maxdims,  int* dims,  int* periods,  int* coords, int *ierr) {
  *ierr = MPI_Cart_get(*comm, *maxdims, dims, periods, coords);
  return *ierr;
}

EXTERNC int mpi_cart_map_ (MPI_Comm* comm,  int* ndims,  const int* dims,  const int* periods,  int* newrank, int *ierr) {
  *ierr = MPI_Cart_map(*comm, *ndims, dims, periods, newrank);
  return *ierr;
}

EXTERNC int mpi_cart_rank_ (MPI_Comm* comm,  const int* coords,  int* rank, int *ierr) {
  *ierr = MPI_Cart_rank(*comm, coords, rank);
  return *ierr;
}

EXTERNC int mpi_cart_shift_ (MPI_Comm* comm,  int* direction,  int* disp,  int* rank_source,  int* rank_dest, int *ierr) {
  *ierr = MPI_Cart_shift(*comm, *direction, *disp, rank_source, rank_dest);
  return *ierr;
}

EXTERNC int mpi_cart_sub_ (MPI_Comm* comm,  const int* remain_dims,  MPI_Comm* new_comm, int *ierr) {
  *ierr = MPI_Cart_sub(*comm, remain_dims, new_comm);
  return *ierr;
}

EXTERNC int mpi_cartdim_get_ (MPI_Comm* comm,  int* ndims, int *ierr) {
  *ierr = MPI_Cartdim_get(*comm, ndims);
  return *ierr;
}

EXTERNC int mpi_test_ (MPI_Request* request,  int* flag,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Test(request, flag, status);
  return *ierr;
}

EXTERNC int mpi_wait_ (MPI_Request* request,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Wait(request, status);
  return *ierr;
}

EXTERNC int mpi_iprobe_ (int* source,  int* tag,  MPI_Comm* comm,  int* flag,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Iprobe(*source, *tag, *comm, flag, status);
  return *ierr;
}

EXTERNC int mpi_probe_ (int* source,  int* tag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Probe(*source, *tag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_waitall_ (int* count,  MPI_Request* array_of_requests,  MPI_Status* array_of_statuses, int *ierr) {
  *ierr = MPI_Waitall(*count, array_of_requests, array_of_statuses);
  return *ierr;
}

EXTERNC int mpi_testall_ (int* count,  MPI_Request* requests,  int* flag,  MPI_Status* statuses, int *ierr) {
  *ierr = MPI_Testall(*count, requests, flag, statuses);
  return *ierr;
}

EXTERNC int mpi_comm_group_ (MPI_Comm* comm,  MPI_Group* group, int *ierr) {
  *ierr = MPI_Comm_group(*comm, group);
  return *ierr;
}

EXTERNC int mpi_group_size_ (MPI_Group* group,  int* size, int *ierr) {
  *ierr = MPI_Group_size(*group, size);
  return *ierr;
}

EXTERNC int mpi_group_free_ (MPI_Group* group, int *ierr) {
  *ierr = MPI_Group_free(group);
  return *ierr;
}

EXTERNC int mpi_group_compare_ (MPI_Group* group1,  MPI_Group* group2,  int* result, int *ierr) {
  *ierr = MPI_Group_compare(*group1, *group2, result);
  return *ierr;
}

EXTERNC int mpi_group_rank_ (MPI_Group* group,  int* rank, int *ierr) {
  *ierr = MPI_Group_rank(*group, rank);
  return *ierr;
}

EXTERNC int mpi_group_incl_ (MPI_Group* group,  int* n,  const int* ranks,  MPI_Group* newgroup, int *ierr) {
  *ierr = MPI_Group_incl(*group, *n, ranks, newgroup);
  return *ierr;
}

EXTERNC int mpi_type_size_ (MPI_Datatype* datatype,  int* size, int *ierr) {
  *ierr = MPI_Type_size(*datatype, size);
  return *ierr;
}

EXTERNC int mpi_type_commit_ (MPI_Datatype* type, int *ierr) {
  *ierr = MPI_Type_commit(type);
  return *ierr;
}

EXTERNC int mpi_type_contiguous_ (int* count,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_contiguous(*count, *oldtype, newtype);
  return *ierr;
}

EXTERNC int mpi_type_free_ (MPI_Datatype* type, int *ierr) {
  *ierr = MPI_Type_free(type);
  return *ierr;
}

EXTERNC int mpi_type_vector_ (int* count,  int* blocklength,  int* stride,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_vector(*count, *blocklength, *stride, *oldtype, newtype);
  return *ierr;
}

EXTERNC int mpi_type_indexed_ (int* count,  const int* array_of_blocklengths,  const int* array_of_displacements,  MPI_Datatype* oldtype,  MPI_Datatype* newtype, int *ierr) {
  *ierr = MPI_Type_indexed(*count, array_of_blocklengths, array_of_displacements, *oldtype, newtype);
  return *ierr;
}

EXTERNC int mpi_pack_size_ (int* incount,  MPI_Datatype* datatype,  MPI_Comm* comm,  int* size, int *ierr) {
  *ierr = MPI_Pack_size(*incount, *datatype, *comm, size);
  return *ierr;
}

EXTERNC int mpi_pack_ (const void* inbuf,  int* incount,  MPI_Datatype* datatype,  void* outbuf,  int* outsize,  int* position,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Pack(inbuf, *incount, *datatype, outbuf, *outsize, position, *comm);
  return *ierr;
}

EXTERNC int mpi_op_create_ (MPI_User_function* user_fn,  int* commute,  MPI_Op* op, int *ierr) {
  *ierr = MPI_Op_create(user_fn, *commute, op);
  return *ierr;
}

EXTERNC int mpi_op_free_ (MPI_Op* op, int *ierr) {
  *ierr = MPI_Op_free(op);
  return *ierr;
}

EXTERNC int mpi_send_ (const void* buf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* tag,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Send(buf, *count, *datatype, *dest, *tag, *comm);
  return *ierr;
}

EXTERNC int mpi_isend_ (const void* buf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* tag,  MPI_Comm* comm,  MPI_Request* request, int *ierr) {
  *ierr = MPI_Isend(buf, *count, *datatype, *dest, *tag, *comm, request);
  return *ierr;
}

EXTERNC int mpi_recv_ (void* buf,  int* count,  MPI_Datatype* datatype,  int* source,  int* tag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Recv(buf, *count, *datatype, *source, *tag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_irecv_ (void* buf,  int* count,  MPI_Datatype* datatype,  int* source,  int* tag,  MPI_Comm* comm,  MPI_Request* request, int *ierr) {
  *ierr = MPI_Irecv(buf, *count, *datatype, *source, *tag, *comm, request);
  return *ierr;
}

EXTERNC int mpi_sendrecv_ (const void* sendbuf,  int* sendcount,  MPI_Datatype* sendtype,  int* dest,  int* sendtag,  void* recvbuf,  int* recvcount,  MPI_Datatype* recvtype,  int* source,  int* recvtag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Sendrecv(sendbuf, *sendcount, *sendtype, *dest, *sendtag, recvbuf, *recvcount, *recvtype, *source, *recvtag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_sendrecv_replace_ (void* buf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* sendtag,  int* source,  int* recvtag,  MPI_Comm* comm,  MPI_Status* status, int *ierr) {
  *ierr = MPI_Sendrecv_replace(buf, *count, *datatype, *dest, *sendtag, *source, *recvtag, *comm, status);
  return *ierr;
}

EXTERNC int mpi_rsend_ (const void* ibuf,  int* count,  MPI_Datatype* datatype,  int* dest,  int* tag,  MPI_Comm* comm, int *ierr) {
  *ierr = MPI_Rsend(ibuf, *count, *datatype, *dest, *tag, *comm);
  return *ierr;
}

