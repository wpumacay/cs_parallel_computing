
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;



__global__ void vectorSum( int* d_arr, int size )
{
    int tId = threadIdx.x;

    int Nlevels = ( int ) log2( ( double ) size );

    for ( int q = 0; q <= Nlevels; q++ )
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


#define SIZE 20

int main()
{
    int n_arr[SIZE];
    for ( int q = 0; q < SIZE; q++ )
    {
        n_arr[q] = q;
    }

    int *d_n_arr;
    
    dim3 dimBlock( SIZE, 1 );
    dim3 dimGrid( 1, 1 );

    const int arr_size = SIZE * sizeof( int );

    cudaMalloc( ( void** ) &d_n_arr, arr_size );
    cudaMemcpy( d_n_arr, n_arr, arr_size, cudaMemcpyHostToDevice );

    vectorSum<<< dimGrid, dimBlock >>>( d_n_arr, SIZE );

    cudaMemcpy( n_arr, d_n_arr, arr_size, cudaMemcpyDeviceToHost );

    cudaFree( d_n_arr );

    cout << "nSum: " << n_arr[0] << endl;

    return 0;
}
