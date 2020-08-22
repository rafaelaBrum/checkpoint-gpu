#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>

#include "config.h"
#include "cuda_plugin.h"


// 0.
EXTERNC cudaError_t
cudaMallocManaged(void **pointer, size_t size, unsigned int flags)
{
  FILE *file = fopen("tracelog.txt", "a");
  fprintf(file, "Entering cudaMallocManaged() function\n");
  fclose(file);
  if (!initialized)
    proxy_initialize();

#ifdef USERFAULTFD_INITIALIZED
  if (!ufd_initialized)
    userfaultfd_initialize();
#else
  if (!segvfault_initialized)
    segvfault_initialize();
#endif

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMallocManaged;
  strce_to_send.syscall_type.cuda_malloc.pointer = *pointer;
  strce_to_send.syscall_type.cuda_malloc.size = size;
  // TODO: Add field for flags

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  JASSERT(ret_val == cudaSuccess)(ret_val).Text("Failed to create UVM region");

  // change the pointer to point the global memory (device memory)
  *pointer = create_shadow_pages(size, &rcvd_strce);

  if (should_log_cuda_calls()) {
    // record this system call to the log file
    memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
    strce_to_send.op = CudaMallocManaged;
    strce_to_send.syscall_type.cuda_malloc.pointer = pointer;
    strce_to_send.syscall_type.cuda_malloc.size = size;

    log_append(strce_to_send);
  }
  
  file = fopen("tracelog.txt", "a");
  fprintf(file, "Exiting cudaMallocManaged() function with %d value\n", ret_val);
  fclose(file);

  return ret_val;
}
