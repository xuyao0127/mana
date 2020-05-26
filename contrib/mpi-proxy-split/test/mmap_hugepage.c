#include <stdlib.h>
#include <sys/mman.h>
#include <stdio.h>
#include <unistd.h>
#define NUM_PAGES 100

int main(){
  void *addr;
  int length=1024*1024*4;
  for(int i=0;i<NUM_PAGES;i++){
    addr = mmap(NULL, length, PROT_READ|PROT_WRITE,
    	        MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
    if (addr == MAP_FAILED){
      printf("mmap_failed");
    }
  }
  sleep(600); 
}
