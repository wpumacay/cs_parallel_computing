

#include <cmath>
#include "cuda_vectorsum.h"

__global__ void vectorSum( int* d_arr, int size )
{
    int tId = threadIdx.x;

    int Nlevels = ( int ) log2( ( double ) size );

    for ( int q = 0; q < Nlevels; q++ )
    {
        int _dStep = ( int ) pow( 2.0, ( double ) ( q + 1 ) );
        if ( tId % _dStep == 0 )
        {
            int _nextIndx = ( ( int ) ( pow( 2.0, ( double ) q ) ) );
            d_arr[tId] += d_arr[tId + _nextIndx];
        }
        __syncthreads();
    }
}


void calculateSum( int* h_arr, int size )
{
    int *d_arr;
    
    dim3 dimBlock( size, 1 );
    dim3 dimGrid( 1, 1 );

    const int arr_size = size * sizeof( int );

    cudaMalloc( ( void** ) &d_arr, arr_size );
    cudaMemcpy( d_arr, h_arr, arr_size, cudaMemcpyHostToDevice );

    vectorSum<<< dimGrid, dimBlock >>>( d_arr, size );
    
    cudaMemcpy( h_arr, d_arr, arr_size, cudaMemcpyDeviceToHost );

    cudaFree( d_arr );
}
