#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <mpi.h>
#define NUM_PAGES 100

int main(){
  void *addr;
  int length=1024*1024*4;
  int fd;
  struct stat statbuf;
  fd=open("mmap_test_file",O_RDONLY);
  if(fd < 0){
    printf("open() failed\n");
  }
   /* find size of input file */
  if (fstat (fd,&statbuf) < 0)
  {
    printf ("fstat error");
    return 0;
  }
  MPI_Init(NULL,NULL);
  //printf("fstat.length: %d",statbuf.st_size);
  /*
  for(int i=0;i<NUM_PAGES;i++){
    addr = mmap(NULL, statbuf.st_size, PROT_READ,
    	        MAP_PRIVATE|MAP_HUGETLB, fd, 0);
    if (addr == MAP_FAILED){
      printf("mmap_failed\n");
    }
    else{
      printf("mmap successful: %d\n",i);
    }
    fflush(stdout);
    sleep(5);
  }
  */
}
