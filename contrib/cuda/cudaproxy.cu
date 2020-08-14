#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>
#include <cuda_profiler_api.h>

#ifdef USE_SHM
# include <sys/ipc.h>
# include <sys/shm.h>
#endif

// Definitions of common structs shared with the main process
#include "cuda_plugin.h"
#include "trampolines.h"

#define SKTNAME "proxy"
#define specialCudaReturnValue  60000

#ifndef EXTERNC
# ifdef __cplusplus
#  define EXTERNC extern "C"
# else // ifdef __cplusplus
#  define EXTERNC
# endif // ifdef __cplusplus
#endif // ifndef EXTERNC

#ifdef USE_SHM
int shmID;
void *shmaddr;
#endif

static trampoline_info_t main_trampoline_info;

static int compute(int fd, cudaSyscallStructure *structure);
static int start_proxy(void);

// This is the trampoline destination for the user main; this does not return
// to the user main function.
int main_wrapper()
{
  start_proxy();
  return 0;
}

__attribute__((constructor))
void proxy_init()
{
  void *handle = dlopen(NULL, RTLD_NOW);
  void *addr = dlsym(handle, "main");
  assert(addr != NULL);
  dmtcp_setup_trampoline_by_addr(addr, (void*)&main_wrapper, &main_trampoline_info);
}

static int start_proxy(void)
{
  // set up the server
  int skt_proxy, skt_accept;
  struct sockaddr_un sa_proxy;
  const char *sktname = getenv("CUDA_PROXY_SOCKET");
  if (!sktname) {
    sktname = SKTNAME;
  }

  (void) unlink(sktname);
  memset(&sa_proxy, 0, sizeof(sa_proxy));
  strcpy(sa_proxy.sun_path, sktname);
  sa_proxy.sun_family = AF_UNIX;

  if ((skt_proxy = socket(AF_UNIX, SOCK_STREAM, 0)) == -1)
  {
    perror("socket()");
    exit(EXIT_FAILURE);
  }

  if (bind(skt_proxy, (struct sockaddr *)&sa_proxy, sizeof(sa_proxy)) == -1)
  {
    perror("bind()");
    exit(EXIT_FAILURE);
  }

  if (listen(skt_proxy, SOMAXCONN) == -1)
  {
    perror("listen()");
    exit(EXIT_FAILURE);
  }
 
  if ((skt_accept = accept(skt_proxy, NULL, 0)) == -1)
  {
    perror("accept()");
    exit(EXIT_FAILURE);
  }

#ifdef USE_SHM
  // read the shmID
  if (read(skt_accept, &shmID, sizeof(shmID)) == -1)
  {
    perror("read()");
    exit(EXIT_FAILURE);
  }


  if ((shmaddr = shmat(shmID, NULL, 0)) == (void *) -1)
  {
    perror("shmat()");
    exit(EXIT_FAILURE);
  }
#endif

  int return_val;
  cudaSyscallStructure structure;

  while(1)
  {
    // read the structure
    // At this stage the GPU does the computation as well.

    if (read(skt_accept, &structure, sizeof(structure)) == -1)
    {
      perror("read()");
      exit(EXIT_FAILURE);
    }
    return_val = compute(skt_accept, &structure);

    // send the result
    if (write(skt_accept, &return_val, sizeof(return_val)) == -1)
    {
      perror("write()");
      exit(EXIT_FAILURE);
    }

    // send the datastructure back
    if (write(skt_accept, &structure, sizeof(structure)) == -1)
    {
      perror("write()");
      exit(EXIT_FAILURE);
    }
    // "cudaGetErrorString" is a special case, its return type
    // is "const char *"
    if (structure.op == CudaGetErrorString)
    {
      // send the error string to master.
      char *error_string = structure.syscall_type.cuda_get_error_string.error_string;
      size_t size = structure.syscall_type.cuda_get_error_string.size;
      if (write(skt_accept, error_string, size) == -1)
      {
        perror("write()");
        exit(EXIT_FAILURE);
      }
    }

   }
}


int compute(int fd, cudaSyscallStructure *structure)
{
  int return_val;
  enum cuda_syscalls op = structure->op;

  enum cudaMemcpyKind direction = (structure->syscall_type).cuda_memcpy.direction;
  size_t size = (structure->syscall_type).cuda_memcpy.size;
  void *new_source, *new_destination;

  switch (op)
  {
    //
    case CudaDeviceSynchronize:
      return_val = cudaDeviceSynchronize();
      break;

    case CudaThreadSync:
      return_val = cudaThreadSynchronize();
      break;

    case CudaGetLastError:
      return_val = cudaGetLastError();
      break;

    case CudaGetErrorString:
    {
      cudaError_t errorCode = (cudaError_t)(structure->syscall_type).cuda_get_error_string.errorCode;
      char *error_string = (char *)cudaGetErrorString(errorCode);
      (structure->syscall_type).cuda_get_error_string.error_string = error_string;
      (structure->syscall_type).cuda_get_error_string.size = strlen(error_string);
      return_val = cudaSuccess;
    }
      break;

    case CudaMalloc:
     {
      return_val =  cudaMalloc(&((structure->syscall_type).cuda_malloc.pointer),
        (structure->syscall_type).cuda_malloc.size);
     }
     break;


    case CudaMallocPitch:
     {
      // void **devPtr = &((structure->syscall_type).cuda_malloc_pitch.devPtr);
      // size_t *pitch = &((structure->syscall_type).cuda_malloc_pitch.pitch);
      size_t width = (structure->syscall_type).cuda_malloc_pitch.width;
      size_t height = (structure->syscall_type).cuda_malloc_pitch.height;
      return_val =  cudaMallocPitch(&((structure->syscall_type).cuda_malloc_pitch.devPtr), \
                    &((structure->syscall_type).cuda_malloc_pitch.pitch), width, height);
     }
     break;

    case CudaDeviceReset:
    {
      return_val = cudaDeviceReset();
    }
    break;

    case CudaMemcpyToSymbol:
    {
      const void * symbol = (structure->syscall_type).cuda_memcpy_to_symbol.symbol;
      const void * src = (structure->syscall_type).cuda_memcpy_to_symbol.src;
      void *new_src;
      size_t count = (structure->syscall_type).cuda_memcpy_to_symbol.count;
      size_t offset = (structure->syscall_type).cuda_memcpy_to_symbol.offset;
      enum cudaMemcpyKind kind = (structure->syscall_type).cuda_memcpy_to_symbol.kind;

      if (kind == cudaMemcpyHostToDevice){
        // read the payload.
        new_src = malloc(count);
        if (new_src == NULL){
          perror("malloc()");
          exit(EXIT_FAILURE);
        }
        if (read(fd, new_src, count) == -1){
          perror("read()");
          exit(EXIT_FAILURE);
        }
        return_val = cudaMemcpyToSymbol(symbol, new_src, count, offset, kind);
      }
      else if (kind == cudaMemcpyDeviceToDevice){
        return_val = cudaMemcpyToSymbol(symbol, src, count, offset, kind);
      }
      else{
        perror("bad direction value");
        exit(EXIT_FAILURE);
      }
    }
    break;


    case CudaCreateChannelDesc:
    {
      int x = (structure->syscall_type).cuda_create_channel_desc.x;
      int y = (structure->syscall_type).cuda_create_channel_desc.y;
      int z = (structure->syscall_type).cuda_create_channel_desc.z;
      int w = (structure->syscall_type).cuda_create_channel_desc.w;
      enum cudaChannelFormatKind f = (structure->syscall_type).cuda_create_channel_desc.f;
      (structure->syscall_type).cuda_create_channel_desc.ret_val = cudaCreateChannelDesc(x, y, z, w, f);

      return_val = cudaSuccess;
    }
    break;


    case CudaBindTexture2D:
    {
      // offset is an "out" parameter.
      size_t offset = (structure->syscall_type).cuda_bind_texture2_d.offset;
      // size_t * offset = (structure->syscall_type).cuda_bind_texture2_d.offset;
      struct textureReference *texref = (structure->syscall_type).cuda_bind_texture2_d.texref;
      const void * devPtr = (structure->syscall_type).cuda_bind_texture2_d.devPtr;
      const cudaChannelFormatDesc desc = (structure->syscall_type).cuda_bind_texture2_d.desc;
      size_t width = (structure->syscall_type).cuda_bind_texture2_d.width;
      size_t height = (structure->syscall_type).cuda_bind_texture2_d.height;
      size_t pitch = (structure->syscall_type).cuda_bind_texture2_d.pitch;
      if (offset == 0)
        return_val = cudaBindTexture2D(NULL, texref, devPtr, &desc, width, height, pitch);
      else
        return_val = cudaBindTexture2D(&offset, texref, devPtr, &desc, width, height, pitch);
        (structure->syscall_type).cuda_bind_texture2_d.offset = offset;
    }
    break;

    case CudaBindTexture:
    {
      // offset is an "out" parameter.
      size_t offset = (structure->syscall_type).cuda_bind_texture.offset;
      const textureReference * texref = (structure->syscall_type).cuda_bind_texture.texref;
      const void * devPtr = (structure->syscall_type).cuda_bind_texture.devPtr;
      const cudaChannelFormatDesc desc = (structure->syscall_type).cuda_bind_texture.desc;
      size_t size = (structure->syscall_type).cuda_bind_texture.size;
      if (offset == 0)
        return_val = cudaBindTexture(NULL, texref, devPtr, &desc, size);
      else
        return_val = cudaBindTexture(&offset, texref, devPtr, &desc, size);
      (structure->syscall_type).cuda_bind_texture.offset = offset;
    }
    break;

    case CudaMallocManaged:
     {
      return_val =  cudaMallocManaged(&((structure->syscall_type).cuda_malloc.pointer),
                           (structure->syscall_type).cuda_malloc.size);
     }
     break;

    case CudaFree:
      return_val = cudaFree((structure->syscall_type).cuda_free.pointer);
      break;

    case CudaMallocManagedMemcpy:
      switch (direction) {
        case cudaMemcpyDeviceToHost:
          // send data to the master
          if (write(fd,
                    (structure->syscall_type).cuda_memcpy.source,
                    (structure->syscall_type).cuda_memcpy.size) == -1) {
            perror("write()");
            exit(EXIT_FAILURE);
          }

          break;

        case cudaMemcpyHostToDevice:
          // read the data from the master directly into the UVM region
          if (read(fd,
                   (structure->syscall_type).cuda_memcpy.destination,
                   (structure->syscall_type).cuda_memcpy.size) == -1) {
            perror("read()");
            exit(EXIT_FAILURE);
          }

          break;

        default:
          printf("bad direction value: %d\n", direction);
          exit(EXIT_FAILURE);
      }
      break;

    //
    case CudaMemcpy:
      switch(direction)
      {
        case cudaMemcpyHostToDevice:
          // receive payload

          // we need a new pointer for source
          new_source = malloc(size);
          if (new_source == NULL)
          {
            printf("malloc() failed\n");
            exit(EXIT_FAILURE);
          }

          if (read(fd, new_source, size) == -1)
          {
            perror("read()");
            exit(EXIT_FAILURE);
          }

          // GPU computation
          return_val = cudaMemcpy((structure->syscall_type).cuda_memcpy.destination,
                                    new_source, size, direction);

          free(new_source);
          break;


        case cudaMemcpyDeviceToHost:
          // we need a new pointer for destination
          new_destination = malloc(size);
          if (new_destination == NULL)
          {
            printf("malloc() failed\n");
            exit(EXIT_FAILURE);
          }

          // GPU computation
          return_val = cudaMemcpy(new_destination,
                    (structure->syscall_type).cuda_memcpy.source, size, direction);

          // send data to the master
          if (write(fd, new_destination, size) == -1)
          {
            perror("write()");
            exit(EXIT_FAILURE);
          }

          free(new_destination);
          break;

        case cudaMemcpyDeviceToDevice:
          // GPU computation
          return_val = cudaMemcpy((structure->syscall_type).cuda_memcpy.destination,
           (structure->syscall_type).cuda_memcpy.source, size, direction);
          break;

        default:
          printf("bad direction value: %d\n", direction);
          exit(EXIT_FAILURE);

      }
      break;

    case CudaMemcpy2D:
    {
      void *dst = (structure->syscall_type).cuda_memcpy2_d.dst;
      void *new_dst;
      size_t dpitch = (structure->syscall_type).cuda_memcpy2_d.dpitch;
      const void *src =  (structure->syscall_type).cuda_memcpy2_d.src;
      void *new_src;
      size_t spitch =  (structure->syscall_type).cuda_memcpy2_d.spitch;
      size_t width =  (structure->syscall_type).cuda_memcpy2_d.width;
      size_t height =  (structure->syscall_type).cuda_memcpy2_d.height;
      enum cudaMemcpyKind kind =  (structure->syscall_type).cuda_memcpy2_d.kind;

      switch(kind)
      {
        case cudaMemcpyHostToDevice:
          // receive payload
          // we need a new memory space for source
          new_src = malloc(spitch * height);
          if (new_src == NULL)
          {
            printf("malloc() failed\n");
            exit(EXIT_FAILURE);
          }

          if (read(fd, new_src, spitch * height) == -1)
          {
            perror("read()");
            exit(EXIT_FAILURE);
          }

          // GPU computation
          return_val = cudaMemcpy2D(dst, dpitch, new_src, spitch, width, height, kind);

          free(new_src);
          break;


        case cudaMemcpyDeviceToHost:
          // we need a new pointer for destination
          new_dst = malloc(dpitch * height);
          if (new_destination == NULL)
          {
            printf("malloc() failed\n");
            exit(EXIT_FAILURE);
          }

          // GPU computation
          return_val = cudaMemcpy2D(new_dst, dpitch, src, spitch, width, height, kind);

          // send data to the master
          if (write(fd, new_dst, (dpitch * height)) == -1)
          {
            perror("write()");
            exit(EXIT_FAILURE);
          }

          free(new_dst);
          break;

        case cudaMemcpyDeviceToDevice:
          // GPU computation
          return_val = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
          break;

        default:
          printf("bad direction value: %d\n", direction);
          exit(EXIT_FAILURE);

      }
    }
    break;


    case CudaMallocArray:
      return_val = cudaMallocArray(&((structure->syscall_type).cuda_malloc_array.array),
                       &((structure->syscall_type).cuda_malloc_array.desc),
                       (structure->syscall_type).cuda_malloc_array.width,
                       (structure->syscall_type).cuda_malloc_array.height,
                       (structure->syscall_type).cuda_malloc_array.flags);

      // send shmid

      break;

    case CudaFreeArray:
      return_val = cudaFreeArray((structure->syscall_type).cuda_free_array.array);
      break;

  //  case CudaHostAlloc:
  //    return_val = cudaHostAlloc(&((structure->syscall_type).cuda_host_alloc.pHost),
  //                                (structure->syscall_type).cuda_host_alloc.size,
  //                                (structure->syscall_type).cuda_host_alloc.flags);
  //    break;

    case CudaConfigureCall:
    {
      int *gridDim = (structure->syscall_type).cuda_configure_call.gridDim;
      dim3 gDim(gridDim[0], gridDim[1], gridDim[2]);      

      int *blockDim = (structure->syscall_type).cuda_configure_call.blockDim;
      dim3 bDim(blockDim[0], blockDim[1], blockDim[2]);
      size_t sharedMem = (structure->syscall_type).cuda_configure_call.sharedMem;
      cudaStream_t stream = (structure->syscall_type).cuda_configure_call.stream;
      return_val = cudaConfigureCall(gDim, bDim, sharedMem, stream);
    }

    break;

    case CudaSetupArgument:
    {
      size_t size = (structure->syscall_type).cuda_setup_argument.size;
      size_t offset = (structure->syscall_type).cuda_setup_argument.offset;

      void *arg = malloc(size);
#ifdef USE_SHM
      memcpy(arg, shmaddr, size);
#endif
      if (read(fd, arg, size) == -1)
      {
        perror("read()");
        exit(EXIT_FAILURE);
      }
      return_val = cudaSetupArgument(arg, size, offset);
      break; 
    }

    case CudaLaunch:
     {
      const void *func = (structure->syscall_type).cuda_launch.func_addr;
      return_val = cudaLaunch(func);
     }
     break;

    case CudaPeekAtLastError:
    {
      return_val = cudaPeekAtLastError();
    }
    break;

    case CudaProfilerStart:
    {
      return_val = cudaProfilerStart();
    }
    break;

    case CudaProfilerStop:
    {
      return_val = cudaProfilerStop();
    }
    break;

    case CudaStreamSynchronize:
    {
      cudaStream_t stream;
      stream = (structure->syscall_type).cuda_stream_synchronize.stream;
      return_val = cudaStreamSynchronize(stream);
    }
    break;

    case CudaUnbindTexture:
    {
      const textureReference* texref;
      texref = (structure->syscall_type).cuda_unbind_texture.texref;
      return_val = cudaUnbindTexture(texref);
    }
    break;


    case CudaCreateTextureObject:
     {
       cudaTextureObject_t pTexObject;
       struct cudaResourceDesc pResDesc = (structure->syscall_type).cuda_create_texture_object.pResDesc;
       struct cudaTextureDesc pTexDesc = (structure->syscall_type).cuda_create_texture_object.pTexDesc;
       struct cudaResourceViewDesc pResViewDesc = (structure->syscall_type).cuda_create_texture_object.pResViewDesc;
       return_val = cudaCreateTextureObject(&pTexObject, &pResDesc, &pTexDesc, &pResViewDesc);
       // pTextObject is an "out" parameter
       structure->syscall_type.cuda_create_texture_object.pTexObject = pTexObject;
     }
     break;

     case CudaDestroyTextureObject:
     {
       return_val = cudaDestroyTextureObject((structure->syscall_type).cuda_destroy_texture_object.texObject);
     }
     break;

     case CudaEventDestroy:
     {
       return_val = cudaEventDestroy((structure->syscall_type).cuda_event_destroy.event);
     }
     break;

     case CudaEventQuery:
     {
       return_val = cudaEventQuery((structure->syscall_type).cuda_event_query.event);
     }
     break;

    case CudaFreeHost:
    {
      return_val = cudaFreeHost((structure->syscall_type).cuda_free_host.ptr);
    }
    break;

    case CudaDeviceCanAccessPeer:
    {
      int canAccessPeer;
      int device = (structure->syscall_type).cuda_device_can_access_peer.device;
      int peerDevice = (structure->syscall_type).cuda_device_can_access_peer.peerDevice;
      return_val = cudaDeviceCanAccessPeer (&canAccessPeer, device, peerDevice);
      (structure->syscall_type).cuda_device_can_access_peer.canAccessPeer = canAccessPeer;
    }
    break;

    case CudaDeviceGetAttribute:
    {
      int value;
      cudaDeviceAttr attr = (structure->syscall_type).cuda_device_get_attribute.attr;
      int device = (structure->syscall_type).cuda_device_get_attribute.device;
      return_val = cudaDeviceGetAttribute(&value, attr, device);
      (structure->syscall_type).cuda_device_get_attribute.value = value;
    }
    break;

    case CudaDeviceSetCacheConfig:
    {
      cudaFuncCache cacheConfig = (structure->syscall_type).cuda_deviceSetCacheConfig.cacheConfig;
      return_val = cudaDeviceSetCacheConfig(cacheConfig);
    }
    break;

    case CudaDeviceSetSharedMemConfig:
    {
      cudaSharedMemConfig config = (structure->syscall_type).cuda_deviceSetSharedMemConfig.config;
      return_val = cudaDeviceSetSharedMemConfig(config);
    }
    break;

    case CudaEventCreateWithFlags:
    {
      cudaEvent_t event;
      unsigned int flags = (structure->syscall_type).cuda_eventCreateWithFlags.flags;
      return_val = cudaEventCreateWithFlags(&event, flags);
      (structure->syscall_type).cuda_eventCreateWithFlags.event = event;
    }
    break;

    case CudaEventRecord:
    {
      cudaEvent_t event = (structure->syscall_type).cuda_eventRecord.event;
      cudaStream_t stream = (structure->syscall_type).cuda_eventRecord.stream;

      return_val = cudaEventRecord(event, stream);
    }
    break;

    case CudaFuncGetAttributes:
    {
      cudaFuncAttributes attr;
      const void *func = (structure->syscall_type).cuda_funcGetAttributes.func;
      return_val = cudaFuncGetAttributes(&attr, func);
      (structure->syscall_type).cuda_funcGetAttributes.attr = attr;
    }
    break;

    case CudaGetDevice:
    {
      int device;
      return_val = cudaGetDevice(&device);
      (structure->syscall_type).cuda_getDevice.device = device;
    }
    break;

    case CudaGetDeviceCount:
    {
      int count;
      return_val = cudaGetDeviceCount(&count);
      (structure->syscall_type).cuda_getDeviceCount.count = count;
    }
    break;

    case CudaGetDeviceProperties:
    {
      cudaDeviceProp prop;
      int device = (structure->syscall_type).cuda_getDeviceProperties.device;
      return_val = cudaGetDeviceProperties(&prop, device);
      (structure->syscall_type).cuda_getDeviceProperties.prop = prop;
    }
    break;

    case CudaMemset:
    {
      void *devPtr = (structure->syscall_type).cuda_memset.devPtr;
      int value = (structure->syscall_type).cuda_memset.value;
      size_t count = (structure->syscall_type).cuda_memset.count;
      return_val = cudaMemset(devPtr, value, count);
    }
    break;

    case CudaSetDevice:
    {
      int device = (structure->syscall_type).cuda_setDevice.device;
      return_val = cudaSetDevice(device);
    }
    break;

    case CudaPointerGetAttributes:
    {
      return_val = cudaPointerGetAttributes(&((structure->syscall_type).cuda_pointer_get_attributes.attributes), (structure->syscall_type).cuda_pointer_get_attributes.ptr);
    }
    break;

    case CudaOccupancyMaxActiveBlocksPerMultiprocessor:
    {
      return_val = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&((structure->syscall_type).cuda_occupancy_max_active_blocks_per_multiprocessor.numBlocks), \
		      (structure->syscall_type).cuda_occupancy_max_active_blocks_per_multiprocessor.func, \
		      (structure->syscall_type).cuda_occupancy_max_active_blocks_per_multiprocessor.blockSize, \
		      (structure->syscall_type).cuda_occupancy_max_active_blocks_per_multiprocessor.dynamicSMemSize);
    }
    break;

    default:
      printf("bad op value: %d\n", (int) op);
      exit(EXIT_FAILURE);

  }

  return return_val;
}
