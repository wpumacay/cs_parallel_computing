
#pragma once

#include <cuda_runtime.h>

struct GpuTimer
{

private :

  cudaEvent_t m_start;
  cudaEvent_t m_stop;

public :

  GpuTimer()
  {
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
  }

  void start()
  {
    cudaEventRecord(m_start, 0);
  }

  void stop()
  {
    cudaEventRecord(m_stop, 0);
  }

  float elapsed()
  {
    float _elapsed;
    cudaEventSynchronize(m_stop);
    cudaEventElapsedTime(&_elapsed, m_start, m_stop);
    return _elapsed;
  }
};