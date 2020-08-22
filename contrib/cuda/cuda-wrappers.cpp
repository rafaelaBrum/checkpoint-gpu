#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <limits.h>

#include "config.h"
#include "cuda_plugin.h"

struct MallocRegion {
  void *addr;
  void *host_addr;
  size_t len;
};

static dmtcp::vector<MallocRegion>&
allMallocRegions()
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering allMallocRegions() function");
  fclose(file);
  static dmtcp::vector<MallocRegion> *instance = NULL;
  if (instance == NULL) {
    void *buffer = JALLOC_MALLOC(1024 * 1024);
    instance = new (buffer)dmtcp::vector<MallocRegion>();
  }
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting allMallocRegions() function");
  fclose(file);
  return *instance;
}

void
copy_data_to_host()
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering copy_data_to_host() function");
  fclose(file);
  dmtcp::vector<MallocRegion>::iterator it;
  for (it = allMallocRegions().begin(); it != allMallocRegions().end(); it++) {
    void *page = mmap(NULL, it->len, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    JASSERT(page != MAP_FAILED)(JASSERT_ERRNO);
    it->host_addr = page;
    cudaMemcpy(page, it->addr, it->len, cudaMemcpyDeviceToHost);
  }
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting copy_data_to_host() function");
  fclose(file);
}

void
copy_data_to_device()
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering copy_data_to_device() function");
  fclose(file);
  dmtcp::vector<MallocRegion>::iterator it;
  for (it = allMallocRegions().begin(); it != allMallocRegions().end(); it++) {
    cudaMemcpy(it->addr, it->host_addr, it->len, cudaMemcpyHostToDevice);
  }
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting copy_data_to_device() function");
  fclose(file);
}

// 1.
EXTERNC cudaError_t
cudaMalloc(void **pointer, size_t size)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMalloc() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMalloc;
  strce_to_send.syscall_type.cuda_malloc.pointer = *pointer;
  strce_to_send.syscall_type.cuda_malloc.size = size;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  if (should_log_cuda_calls()) {
    void *record_pointer = *pointer;

    // change the pointer to point the global memory (device memory)
    *pointer = rcvd_strce.syscall_type.cuda_malloc.pointer;

    // record this system call to the log file
    memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
    strce_to_send.op = CudaMalloc;
    // FIXME: should we save record_pointer or pointer?
    strce_to_send.syscall_type.cuda_malloc.pointer = pointer;
    strce_to_send.syscall_type.cuda_malloc.size = size;

    log_append(strce_to_send);

    MallocRegion r =  {.addr = *pointer, .host_addr = NULL, .len = size};
    allMallocRegions().push_back(r);
  }

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMalloc() function");
  fclose(file);
  return ret_val;
}

// 2.
EXTERNC cudaError_t
cudaFree(void *pointer)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaFree() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaFree;
  strce_to_send.syscall_type.cuda_free.pointer = pointer;


  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaFree;
  strce_to_send.syscall_type.cuda_free.pointer = pointer;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaFree() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaPointerGetAttributes() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val = cudaSuccess;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaPointerGetAttributes;
  strce_to_send.syscall_type.cuda_pointer_get_attributes.ptr = (void*)ptr;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  *attributes = rcvd_strce.syscall_type.cuda_pointer_get_attributes.attributes;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaPointerGetAttributes() function");
  fclose(file);

  return ret_val;
}

// 3.
EXTERNC cudaError_t
cudaMemcpy(void *pointer1, const void *pointer2, size_t size,
           enum cudaMemcpyKind direction)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMemcpy() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val = cudaSuccess;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaMemcpy;
  strce_to_send.syscall_type.cuda_memcpy.destination = pointer1;
  strce_to_send.syscall_type.cuda_memcpy.source = (void*)pointer2;
  strce_to_send.syscall_type.cuda_memcpy.size = size;
  strce_to_send.syscall_type.cuda_memcpy.direction = direction;

  // FIXME:  Implement 'memory_type(ptr)' to call proxy and return
  //         either 'cudaMemoryYypeHost' or 'cudaMemoryYypeDevice'.
  // SEE:  http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html#group__CUDART__UNIFIED_1gd89830e17d399c064a2f3c3fa8bb4390
  //     Possible code on proxy:
  //     enum cudaMemoryType memory_type(const void* ptr) {
  //       cudaPointerAttributes attributes;
  //       cudaPointerGetAttributes(&attributes, ptr);
  //       return attributes.memoryType;
  //     }
  if (direction == cudaMemcpyDefault)
  {
    cudaPointerAttributes pointer1Attr, pointer2Attr;

    JASSERT(cudaPointerGetAttributes(&pointer1Attr, pointer1) == cudaSuccess)\
                                     .Text("Error getting pointer properties");
    JASSERT(cudaPointerGetAttributes(&pointer2Attr, pointer2) == cudaSuccess)\
                                     .Text("Error getting pointer properties");

    if (pointer1Attr.memoryType == cudaMemoryTypeHost &&
        pointer2Attr.memoryType == cudaMemoryTypeHost) {
      direction = cudaMemcpyHostToHost;
    } else if (pointer1Attr.memoryType == cudaMemoryTypeHost &&
               pointer2Attr.memoryType == cudaMemoryTypeDevice) {
      direction = cudaMemcpyDeviceToHost;
    } else if (pointer1Attr.memoryType == cudaMemoryTypeDevice &&
               pointer2Attr.memoryType == cudaMemoryTypeHost) {
      direction = cudaMemcpyHostToDevice;
    } else if (pointer1Attr.memoryType == cudaMemoryTypeDevice &&
               pointer2Attr.memoryType == cudaMemoryTypeDevice) {
      direction = cudaMemcpyDeviceToDevice;
    } else {
      JASSERT(false).Text("DMTCP/CUDA internal error");
    }
  }

  switch(direction)
  {
    case cudaMemcpyHostToHost:
    {
      memcpy(pointer1, pointer2, size);
      return ret_val;
    }


    case cudaMemcpyHostToDevice:
      strce_to_send.payload = pointer2;
      strce_to_send.payload_size = size;
      send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
      break;

    case cudaMemcpyDeviceToHost:
      // send the structure
      JASSERT(write(skt_master, &strce_to_send, sizeof(strce_to_send)) != -1)
             (JASSERT_ERRNO);

      // get the payload: part of the GPU computation actually
      JASSERT(read(skt_master, pointer1, size) != -1)(JASSERT_ERRNO);

      // receive the result
      memset(&ret_val, 0, sizeof(ret_val));

      JASSERT(read(skt_master, &ret_val, sizeof(ret_val)) != -1)(JASSERT_ERRNO);

      JASSERT(ret_val == cudaSuccess)(ret_val).Text("cudaMemcpy failed");

      // get the structure back
      memset(&rcvd_strce, 0, sizeof(rcvd_strce));
      JASSERT(read(skt_master, &rcvd_strce, sizeof(rcvd_strce)) != -1)
             (JASSERT_ERRNO);
      break;

    case cudaMemcpyDeviceToDevice:
      send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

      break;

    case cudaMemcpyDefault:
      JASSERT(false).
        Text("DMTCP/CUDA internal error: cudaMemcpyDefault not translated");

    default:
      JASSERT(false)(direction).Text("Unknown direction for memcpy");
  }

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaMemcpy;
  strce_to_send.syscall_type.cuda_memcpy.destination = pointer1;
  strce_to_send.syscall_type.cuda_memcpy.source = pointer2;
  strce_to_send.syscall_type.cuda_memcpy.size = size;
  strce_to_send.syscall_type.cuda_memcpy.direction = direction;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMemcpy() function");
  fclose(file);
  
  return ret_val;
}


EXTERNC cudaError_t
cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
             size_t width, size_t height, enum cudaMemcpyKind direction)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMemcpy2D() function");
  fclose(file);
  
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val = cudaSuccess;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaMemcpy2D;
  strce_to_send.syscall_type.cuda_memcpy2_d.dst = dst;
  strce_to_send.syscall_type.cuda_memcpy2_d.dpitch = dpitch;
  strce_to_send.syscall_type.cuda_memcpy2_d.src = src;
  strce_to_send.syscall_type.cuda_memcpy2_d.spitch = spitch;
  strce_to_send.syscall_type.cuda_memcpy2_d.width = width;
  strce_to_send.syscall_type.cuda_memcpy2_d.height = height;
  strce_to_send.syscall_type.cuda_memcpy2_d.kind = direction;

  switch(direction)
  {
    case cudaMemcpyHostToHost:
    {
      // height: number of rows
      int i;
      for (i=0; i< height; ++i){
        memcpy((void*)((char*)dst+(i*dpitch)), \
                                      (void*)((char*)src+ (i*spitch)), width);
      }
      return ret_val;
    }

    case cudaMemcpyHostToDevice:
    {
      strce_to_send.payload = src;
      // the size includes padding.
      strce_to_send.payload_size = (spitch * height);
      send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
    }
    break;

    case cudaMemcpyDeviceToHost:
      // send the structure
      JASSERT(write(skt_master, &strce_to_send, sizeof(strce_to_send)) != -1)
             (JASSERT_ERRNO);

      // get the payload: part of the GPU computation actually
      // size includes padding.
      JASSERT(read(skt_master, dst, (dpitch * height)) != -1)(JASSERT_ERRNO);

      // receive the result
      memset(&ret_val, 0, sizeof(ret_val));

      JASSERT(read(skt_master, &ret_val, sizeof(ret_val)) != -1)(JASSERT_ERRNO);

      JASSERT(ret_val == cudaSuccess)(ret_val).Text("cudaMemcpy failed");

      // get the structure back
      memset(&rcvd_strce, 0, sizeof(rcvd_strce));
      JASSERT(read(skt_master, &rcvd_strce, sizeof(rcvd_strce)) != -1)
             (JASSERT_ERRNO);
      break;

    case cudaMemcpyDeviceToDevice:
      send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
      break;

    default:
      JASSERT(false)(direction).Text("Unknown direction for memcpy");
  }

//  memset(&strce_to_send, 0, sizeof(strce_to_send));
//  strce_to_send.op = CudaMemcpy;
//  strce_to_send.syscall_type.cuda_memcpy.destination = pointer1;
//  strce_to_send.syscall_type.cuda_memcpy.source = pointer2;
//  strce_to_send.syscall_type.cuda_memcpy.size = size;
//  strce_to_send.syscall_type.cuda_memcpy.direction = direction;

  log_append(strce_to_send);
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMemcpy2D() function");
  fclose(file);
  return ret_val;
}



// 4.
EXTERNC cudaError_t
cudaMallocArray(struct cudaArray **array,
                const struct cudaChannelFormatDesc *desc,
                size_t width, size_t height, unsigned int flags)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMallocArray() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaMallocArray;
  strce_to_send.syscall_type.cuda_malloc_array.array = *array;
  strce_to_send.syscall_type.cuda_malloc_array.desc = *desc;
  strce_to_send.syscall_type.cuda_malloc_array.width = width;
  strce_to_send.syscall_type.cuda_malloc_array.height = height;
  strce_to_send.syscall_type.cuda_malloc_array.flags = flags;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  struct cudaArray *record_array = *array;

  // change the pointer to point the global memory (device memory)
  *array = rcvd_strce.syscall_type.cuda_malloc_array.array;

  // record this system call to the log file
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaMallocArray;
  strce_to_send.syscall_type.cuda_malloc_array.array = record_array;
  strce_to_send.syscall_type.cuda_malloc_array.desc = *desc;
  strce_to_send.syscall_type.cuda_malloc_array.width = width;
  strce_to_send.syscall_type.cuda_malloc_array.height = height;
  strce_to_send.syscall_type.cuda_malloc_array.flags = flags;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMallocArray() function");
  fclose(file);

  return ret_val;
}

// 5.
EXTERNC cudaError_t
cudaFreeArray(struct cudaArray *array)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaFreeArray() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaFreeArray;
  strce_to_send.syscall_type.cuda_free_array.array = array;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaFreeArray;
  strce_to_send.syscall_type.cuda_free_array.array = array;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaFreeArray() function");
  fclose(file);

  return ret_val;
}



//
EXTERNC cudaError_t
cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                  size_t sharedMem, cudaStream_t stream)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaConfigureCall() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaConfigureCall;
  strce_to_send.syscall_type.cuda_configure_call.gridDim[0] = gridDim.x;
  strce_to_send.syscall_type.cuda_configure_call.gridDim[1] = gridDim.y;
  strce_to_send.syscall_type.cuda_configure_call.gridDim[2] = gridDim.z;
  strce_to_send.syscall_type.cuda_configure_call.blockDim[0] = blockDim.x;
  strce_to_send.syscall_type.cuda_configure_call.blockDim[1] = blockDim.y;
  strce_to_send.syscall_type.cuda_configure_call.blockDim[2] = blockDim.z;
  strce_to_send.syscall_type.cuda_configure_call.sharedMem = sharedMem;
  strce_to_send.syscall_type.cuda_configure_call.stream = stream;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaConfigureCall() function");
  fclose(file);

  return ret_val;
}

//
EXTERNC cudaError_t
cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaSetupArgument() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

#if USE_SHM
  memcpy(shmaddr, arg, size);
#endif

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaSetupArgument;
  strce_to_send.syscall_type.cuda_setup_argument.size = size;
  strce_to_send.syscall_type.cuda_setup_argument.offset = offset;
  strce_to_send.payload = arg;
  strce_to_send.payload_size = size;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaSetupArgument;
  strce_to_send.syscall_type.cuda_setup_argument.arg = arg;
  strce_to_send.syscall_type.cuda_setup_argument.size = size;
  strce_to_send.syscall_type.cuda_setup_argument.offset = offset;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaSetupArgument() function");
  fclose(file);

  return ret_val;
}

//
EXTERNC cudaError_t
cudaLaunch(const void *func)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaLaunch() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  // TODO: Ideally, we should flush only when the function uses the
  // data from the managed regions
  if (haveDirtyPages)
    flushDirtyPages();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaLaunch;
  strce_to_send.syscall_type.cuda_launch.func_addr = func;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaLaunch;

  log_append(strce_to_send);
  
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaLaunch() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaThreadSynchronize(void)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaThreadSynchronize() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  memset(&strce_to_send, 0, sizeof(strce_to_send));
  cudaError_t ret_val;

  strce_to_send.op = CudaThreadSync;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaThreadSync;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaThreadSynchronize() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaGetLastError(void)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaGetLastError() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcv_strce;
  memset(&strce_to_send, 0, sizeof(strce_to_send));
  cudaError_t ret_val;

  strce_to_send.op = CudaGetLastError;
  send_recv(skt_master, &strce_to_send, &rcv_strce, &ret_val);

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaGetLastError;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaGetLastError() function");
  fclose(file);

  return ret_val;
}

EXTERNC const char*
cudaGetErrorString(cudaError_t errorCode)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaGetErrorString() function");
  fclose(file);

  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcv_strce;
  memset(&strce_to_send, 0, sizeof(strce_to_send));
  cudaError_t ret_val;
  char *ret_string;

  strce_to_send.op = CudaGetErrorString;
  strce_to_send.syscall_type.cuda_get_error_string.errorCode = errorCode;
  send_recv(skt_master, &strce_to_send, &rcv_strce, &ret_val);

  ret_string = rcv_strce.syscall_type.cuda_get_error_string.error_string;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  strce_to_send.op = CudaGetLastError;

  log_append(strce_to_send);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaGetErrorString() function");
  fclose(file);

  return (const char *) ret_string;
}

EXTERNC cudaError_t
cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMallocPitch() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMallocPitch;
  strce_to_send.syscall_type.cuda_malloc_pitch.devPtr = *devPtr;
  strce_to_send.syscall_type.cuda_malloc_pitch.width = width;
  strce_to_send.syscall_type.cuda_malloc_pitch.height = height;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  // change the pointer to point the global memory (device memory)
  *devPtr = rcvd_strce.syscall_type.cuda_malloc_pitch.devPtr;
  *pitch = rcvd_strce.syscall_type.cuda_malloc_pitch.pitch;

  // useful for restart.
  strce_to_send.syscall_type.cuda_malloc_pitch.devPtr = devPtr;
  strce_to_send.syscall_type.cuda_malloc_pitch.pitchp = pitch;
  log_append(strce_to_send); // FIXME: Not sure it is sufficient for restart.

  MallocRegion r =  {.addr = *devPtr, .host_addr = NULL, .len = width*height};
  allMallocRegions().push_back(r);
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMallocPitch() function");
  fclose(file);

  return ret_val;
}


EXTERNC cudaError_t
cudaDeviceReset( void)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaDeviceReset() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaDeviceReset;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(rcvd_strce);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaDeviceReset() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, \
                               size_t offset, enum cudaMemcpyKind kind)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMemcpyToSymbol() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaMemcpyToSymbol;
  strce_to_send.syscall_type.cuda_memcpy_to_symbol.symbol = symbol;
  strce_to_send.syscall_type.cuda_memcpy_to_symbol.src = src;
  strce_to_send.syscall_type.cuda_memcpy_to_symbol.count = count;
  strce_to_send.syscall_type.cuda_memcpy_to_symbol.offset = offset;
  strce_to_send.syscall_type.cuda_memcpy_to_symbol.kind = kind;

  if (kind == cudaMemcpyHostToDevice){
    strce_to_send.payload = src;
    strce_to_send.payload_size = count;
    send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  }
  else if (kind == cudaMemcpyDeviceToDevice){
    send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  }
  else{
    JASSERT(false)(kind).Text("Unknown direction for cudaMemcpyToSymbol");
  }

  log_append(rcvd_strce);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMemcpyToSymbol() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaCreateChannelDesc() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  size_t size;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaCreateChannelDesc;
  strce_to_send.syscall_type.cuda_create_channel_desc.x = x;
  strce_to_send.syscall_type.cuda_create_channel_desc.y = y;
  strce_to_send.syscall_type.cuda_create_channel_desc.z = z;
  strce_to_send.syscall_type.cuda_create_channel_desc.w = w;
  strce_to_send.syscall_type.cuda_create_channel_desc.f = f;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  JASSERT(ret_val == cudaSuccess)(ret_val).Text("cudaMemcpy failed");


  log_append(rcvd_strce);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaCreateChannelDesc() function");
  fclose(file);

  return (rcvd_strce.syscall_type).cuda_create_channel_desc.ret_val;
}

EXTERNC cudaError_t
cudaBindTexture2D(size_t * offset, const struct textureReference * texref, \
                  const void * devPtr, const cudaChannelFormatDesc * desc, \
                  size_t width, size_t height, size_t pitch)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaBindTexture2D() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;
  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));

  strce_to_send.op = CudaBindTexture2D;
  if (offset == NULL)
    strce_to_send.syscall_type.cuda_bind_texture2_d.offset = 0;
  else
    strce_to_send.syscall_type.cuda_bind_texture2_d.offset = *offset;
  // a texture reference must be a global variable, hence
  // the pointer in the proxy process is valid as well.
  // strce_to_send.syscall_type.cuda_bind_texture2_d.offset = *offset;
  strce_to_send.syscall_type.cuda_bind_texture2_d.texref = \
                                     (struct textureReference*)texref;
  // devPtr is a pointer to memory on the device, which makes
  // it (devPtr) valid in the proxy process as well.
  strce_to_send.syscall_type.cuda_bind_texture2_d.devPtr = devPtr;
  strce_to_send.syscall_type.cuda_bind_texture2_d.desc =  *desc;
  strce_to_send.syscall_type.cuda_bind_texture2_d.width = width;
  strce_to_send.syscall_type.cuda_bind_texture2_d.height = height;
  strce_to_send.syscall_type.cuda_bind_texture2_d.pitch = pitch;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  // offset is an "out" parameter
  if (offset != NULL)
    *offset = (rcvd_strce.syscall_type).cuda_bind_texture2_d.offset;

  log_append(rcvd_strce);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaBindTexture2D() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaBindTexture(size_t * offset, const textureReference * texref, \
const void * devPtr, const cudaChannelFormatDesc * desc, size_t size)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaBindTexture() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaBindTexture;
  if (offset == NULL)
    strce_to_send.syscall_type.cuda_bind_texture.offset = 0;
  else{
    // 1 to indicates that offset != NULL
    strce_to_send.syscall_type.cuda_bind_texture.offset = 1;
  }
  // a texture reference must be a global variable, hence
  // the pointer in the proxy process is valid as well.
  strce_to_send.syscall_type.cuda_bind_texture.texref = texref;
  // devPtr is a pointer to memory on the device, which makes
  // it (devPtr) valid in the proxy process as well.
  strce_to_send.syscall_type.cuda_bind_texture.devPtr = devPtr;
  strce_to_send.syscall_type.cuda_bind_texture.desc = *desc;
  strce_to_send.syscall_type.cuda_bind_texture.size = size;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  // offset is an "out" parameter
  if (offset != NULL)
    *offset = (rcvd_strce.syscall_type).cuda_bind_texture.offset;

  // needed for start
  strce_to_send.syscall_type.cuda_bind_texture.offsetp = offset;
  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaBindTexture() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t cudaCreateTextureObject (cudaTextureObject_t * pTexObject, \
  const struct cudaResourceDesc * pResDesc, \
  const struct cudaTextureDesc *pTexDesc, \
  const struct cudaResourceViewDesc * pResViewDesc)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaCreateTextureObject() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaCreateTextureObject;
  strce_to_send.syscall_type.cuda_create_texture_object.pResDesc = *pResDesc;
  strce_to_send.syscall_type.cuda_create_texture_object.pTexDesc =  *pTexDesc;
  strce_to_send.syscall_type.cuda_create_texture_object.pResViewDesc = \
                                                    *pResViewDesc;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  *pTexObject = rcvd_strce.syscall_type.cuda_create_texture_object.pTexObject;
  // will be useful on restart.
  strce_to_send.syscall_type.cuda_create_texture_object.\
                                            pTexObjectp = pTexObject;
  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaCreateTextureObject() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaPeekAtLastError(void)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaPeekAtLastError() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaPeekAtLastError;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaPeekAtLastError() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaProfilerStart(void)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaProfilerStart() function");
  fclose(file);

  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaProfilerStart;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaProfilerStart() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaProfilerStop(void)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaProfilerSop() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaProfilerStop;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaProfilerSop() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaStreamSynchronize(cudaStream_t stream)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaStreamSynchronize() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaStreamSynchronize;
  strce_to_send.syscall_type.cuda_stream_synchronize.stream = stream;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaStreamSynchronize() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaUnbindTexture (const textureReference* texref)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaUnbindTexture() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaUnbindTexture;
  strce_to_send.syscall_type.cuda_unbind_texture.texref = texref;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaUnbindTexture() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaDestroyTextureObject (cudaTextureObject_t texObject)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaDestroyTextureObject() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaDestroyTextureObject;
  strce_to_send.syscall_type.cuda_destroy_texture_object.texObject = texObject;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaDestroyTextureObject() function");
  fclose(file);
  return ret_val;
}


EXTERNC cudaError_t
cudaEventDestroy (cudaEvent_t event)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaEventDestroy() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaEventDestroy;
  strce_to_send.syscall_type.cuda_event_destroy.event = event;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaEventDestroy() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaEventQuery (cudaEvent_t event)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaEventQuery() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(cudaSyscallStructure));
  strce_to_send.op = CudaEventQuery;
  strce_to_send.syscall_type.cuda_event_query.event = event;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaEventQuery() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaFreeHost(void *ptr)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaFreeHost() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaFreeHost;
  strce_to_send.syscall_type.cuda_free_host.ptr = ptr;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaFreeHost() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaDeviceCanAccessPeer (int* canAccessPeer, int device, int peerDevice)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaFreeHost() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaDeviceCanAccessPeer;
  strce_to_send.syscall_type.cuda_device_can_access_peer.device = device;
  strce_to_send.syscall_type.cuda_device_can_access_peer.\
                                       peerDevice = peerDevice;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  *canAccessPeer = strce_to_send.syscall_type.cuda_device_can_access_peer.\
                                                             canAccessPeer;

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaFreeHost() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaDeviceGetAttribute (int* value, cudaDeviceAttr attr, int device)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaDeviceGetAttribute() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaDeviceGetAttribute;
  strce_to_send.syscall_type.cuda_device_get_attribute.attr = attr;
  strce_to_send.syscall_type.cuda_device_get_attribute.device = device;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  *value = strce_to_send.syscall_type.cuda_device_get_attribute.value;

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaDeviceGetAttribute() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaDeviceSetCacheConfig (cudaFuncCache cacheConfig)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaDeviceSetCacheConfig() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaDeviceSetCacheConfig;
  strce_to_send.syscall_type.cuda_deviceSetCacheConfig.\
                                  cacheConfig = cacheConfig;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaDeviceSetCacheConfig() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaDeviceSetSharedMemConfig (cudaSharedMemConfig config)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaDeviceSetSharedMemConfig() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaDeviceSetSharedMemConfig;
  strce_to_send.syscall_type.cuda_deviceSetSharedMemConfig.config = config;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaDeviceSetSharedMemConfig() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaDeviceSynchronize ( void )
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaDeviceSynchronize() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaDeviceSynchronize;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaDeviceSynchronize() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaEventCreateWithFlags ( cudaEvent_t* event, unsigned int  flags )
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaEventCreateWithFlags() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaEventCreateWithFlags;
  strce_to_send.syscall_type.cuda_eventCreateWithFlags.flags = flags;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  *event = rcvd_strce.syscall_type.cuda_eventCreateWithFlags.event;

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaEventCreateWithFlags() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaEventRecord() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaEventRecord;
  strce_to_send.syscall_type.cuda_eventRecord.event = event;
  strce_to_send.syscall_type.cuda_eventRecord.stream = stream;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaEventRecord() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaFuncGetAttributes() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaFuncGetAttributes;
  // func(pointer) is valid in the proxy as well.
  strce_to_send.syscall_type.cuda_funcGetAttributes.func = func;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  // attr is an "out" parameter.
  *attr = rcvd_strce.syscall_type.cuda_funcGetAttributes.attr;

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaFuncGetAttributes() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaGetDevice ( int* device )
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaFuncGetAttributes() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaGetDevice;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  // device is an "out" parameter.
  *device = rcvd_strce.syscall_type.cuda_getDevice.device;

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaFuncGetAttributes() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaGetDeviceCount(int* count)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaGetDeviceCount() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaGetDeviceCount;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  // count is an "out" parameter.
  *count = rcvd_strce.syscall_type.cuda_getDeviceCount.count;

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaGetDeviceCount() function");
  fclose(file);
  return ret_val;
}

EXTERNC cudaError_t
cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaGetDeviceProperties() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaGetDeviceProperties;
  strce_to_send.syscall_type.cuda_getDeviceProperties.device = device;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);
  // prop is an "out" parameter.
  *prop = rcvd_strce.syscall_type.cuda_getDeviceProperties.prop;
  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaGetDeviceProperties() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaMemset(void* devPtr, int  value, size_t count)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMemset() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaMemset;
  strce_to_send.syscall_type.cuda_memset.devPtr = devPtr;
  strce_to_send.syscall_type.cuda_memset.value = value;
  strce_to_send.syscall_type.cuda_memset.count = count;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMemset() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaSetDevice (int device)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaSetDevice() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaSetDevice;
  strce_to_send.syscall_type.cuda_setDevice.device = device;
  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaSetDevice() function");
  fclose(file);

  return ret_val;
}

EXTERNC cudaError_t
cudaMallocHost ( void** ptr, size_t size )
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMallocHost() function");
  fclose(file);
  // FIXME: need to be implemented later.
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMallocHost() function");
  fclose(file);
  return cudaMalloc(ptr, size);
}

EXTERNC cudaError_t
cudaMemcpyAsync (void* dst, const void* src, size_t count, \
        cudaMemcpyKind kind, cudaStream_t stream)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMemcpyAsync() function");
  fclose(file);
  // FIXME: need to be implemented later.
  
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMemcpyAsync() function");
  fclose(file);
  return cudaMemcpy(dst, src, count, kind);
}

EXTERNC cudaError_t
cudaMemsetAsync (void* devPtr, int  value, size_t count, cudaStream_t stream)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaMemsetAsync() function");
  fclose(file);
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaMemsetAsync() function");
  fclose(file);
  return cudaMemset(devPtr, value, count);
}


EXTERNC cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessor (int *numBlocks, \
           const void *func, int blockSize, size_t dynamicSMemSize)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cudaOccupancyMaxActiveBlocksPerMultiprocessor() function");
  fclose(file);
  if (!initialized)
    proxy_initialize();

  cudaSyscallStructure strce_to_send, rcvd_strce;
  cudaError_t ret_val;

  memset(&strce_to_send, 0, sizeof(strce_to_send));
  memset(&rcvd_strce, 0, sizeof(rcvd_strce));

  strce_to_send.op = CudaOccupancyMaxActiveBlocksPerMultiprocessor;
  strce_to_send.syscall_type.\
      cuda_occupancy_max_active_blocks_per_multiprocessor.func = func;
  strce_to_send.syscall_type.\
      cuda_occupancy_max_active_blocks_per_multiprocessor.blockSize \
        = blockSize;
  strce_to_send.syscall_type.\
      cuda_occupancy_max_active_blocks_per_multiprocessor.dynamicSMemSize \
        = dynamicSMemSize;

  send_recv(skt_master, &strce_to_send, &rcvd_strce, &ret_val);

  *numBlocks = rcvd_strce.\
      syscall_type.cuda_occupancy_max_active_blocks_per_multiprocessor.\
      numBlocks;

  log_append(strce_to_send);

  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cudaOccupancyMaxActiveBlocksPerMultiprocessor() function");
  fclose(file);

  return ret_val;
}
