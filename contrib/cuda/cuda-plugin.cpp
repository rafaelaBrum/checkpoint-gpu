#include <stdio.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "config.h"
#include "cuda_plugin.h"


/* Globals */
int logFd = -1;

static void
cuda_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering cuda_event_hook() function");
  fclose(file);
  /* NOTE:  See warning in plugin/README about calls to printf here. */
  switch (event) {
  case DMTCP_EVENT_INIT:
  {
    JTRACE("The CUDA plugin has been initialized.");
    // create the log file
    logFd = open(LOGFILE, O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);
    if (logFd == -1)
    {
      perror("open()");
      exit(EXIT_FAILURE);
    }
    close(logFd);
    logFd = open(LOGFILE, O_APPEND|O_WRONLY);
    if (logFd == -1)
    {
      perror("open()");
      exit(EXIT_FAILURE);
    }
    break;
  }
  case DMTCP_EVENT_EXIT:
    JTRACE("The plugin is being called before exiting.");
    break;
  default:
    break;
  }
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting cuda_event_hook() function");
  fclose(file);
}

/*
 * Global barriers
 */

static void
pre_ckpt()
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering pre_ckpt() function");
  fclose(file);
  unregister_all_pages();
  copy_data_to_host();
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting pre_ckpt() function");
  fclose(file);
}

static void
resume()
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering resume() function");
  fclose(file);
  register_all_pages();
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting resume() function");
  fclose(file);
}


static void
restart()
{
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Entering restart() function");
  fclose(file);
  JTRACE("Trying to re-init the CUDA driver");
  close(skt_master);
  proxy_initialize();
  reset_uffd();
  register_all_pages();
  logFd = open(LOGFILE, O_APPEND|O_RDWR);
  if (logFd == -1)
  {
    perror("open()");
    exit(EXIT_FAILURE);
  }
  disable_cuda_call_logging();
  cudaSyscallStructure rec;
  memset(&rec, 0, sizeof(rec));
  // Replay calls from the log
  bool ret = log_read(&rec);
  while (ret) {
    // TODO: Add cases for other calls
    if (rec.op == CudaMalloc) {
      cudaMalloc((void**)rec.syscall_type.cuda_malloc.pointer,
                 rec.syscall_type.cuda_malloc.size);
    }
    else if (rec.op == CudaMallocManaged) {
      cudaMallocManaged((void**)rec.syscall_type.cuda_malloc.pointer,
                        rec.syscall_type.cuda_malloc.size);
    }
    else if (rec.op == CudaMallocPitch) {
      cudaMallocPitch((void**)rec.syscall_type.cuda_malloc_pitch.devPtr, \
                              rec.syscall_type.cuda_malloc_pitch.pitchp, \
                              rec.syscall_type.cuda_malloc_pitch.width,  \
                              rec.syscall_type.cuda_malloc_pitch.height);
    }

    ret = log_read(&rec);
  }
  copy_data_to_device();
  enable_cuda_call_logging();
  FILE *file = fopen("tracelog.txt", "w"");
  fprintf(file, "Exiting restart() function");
  fclose(file);
}

static DmtcpBarrier cudaPluginBarriers[] = {
  { DMTCP_GLOBAL_BARRIER_PRE_CKPT, pre_ckpt, "checkpoint" },
  { DMTCP_GLOBAL_BARRIER_RESUME, resume, "resume" },
  { DMTCP_GLOBAL_BARRIER_RESTART, restart, "restart" }
};

DmtcpPluginDescriptor_t cuda_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  PACKAGE_VERSION,
  "cuda",
  "DMTCP",
  "dmtcp@ccs.neu.edu",
  "CUDA plugin",
  DMTCP_DECL_BARRIERS(cudaPluginBarriers),
  cuda_event_hook
};

DMTCP_DECL_PLUGIN(cuda_plugin);
