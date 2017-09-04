
#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

#define VECT_SIZE 999
#define RAND_INT(x) rand() % x

#define CUDA_THREADS_PER_BLOCK 1000

long long int h_arr[VECT_SIZE];
long long int h_arrOut[VECT_SIZE];

//#define TEST_SCAN 1

void printArray( long long int* arr, int size, int offset = 0 )
{
    cout << "{ ";
    for ( int q = 0; q < size; q++ )
    {
        cout << arr[q + offset] << " ";
    }
    cout << "}" << endl;
}


__device__ void kernel_ex_sum_scan_blelloch( int *d_inData, 
                                             int *d_outData, 
                                             int size, int wsize )
{
    int tIndx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tIndx < size )
    {
        d_outData[tIndx] = d_inData[tIndx];
    }

    __syncthreads();

    int _offset = 1;

    // Reduce phase
    for ( int d = wsize >> 1; d > 0; d >>= 1 )
    {
        __syncthreads();
        
        if ( tIndx < d )
        {
            int _ai = _offset * ( 2 * tIndx + 1 ) - 1;
            int _bi = min( _offset * ( 2 * tIndx + 2 ) - 1, size - 1 );

            if ( _ai < size - 1 )
            {
                d_outData[_bi] += d_outData[_ai];
            }
        }

        _offset *= 2;
    }
    
    if ( tIndx == 0 )
    {
        d_outData[size - 1] = 0;
    }

    for ( int d = 1; d < size; d *= 2 )
    {
        _offset >>= 1;
        __syncthreads();

        if ( tIndx < d )
        {
            int _ai = _offset * ( 2 * tIndx + 1 ) - 1;
            int _bi = min( _offset * ( 2 * tIndx + 2 ) - 1, size - 1 );

            if ( _ai < size - 1 )
            {
                int _tmp = d_outData[_ai];
                d_outData[_ai] = d_outData[_bi];
                d_outData[_bi] += _tmp;
            }
        }
    }

    __syncthreads();
    
}


__global__ void kernel_radix_sort_partition( long long int *d_inData,
                                             long long int *d_outData, 
                                             int *d_scanBuff,
                                             int *d_indxBuff,
                                             int bit, int size, int wsize )
{
    int tIndx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tIndx >= size )
    {
        return;
    }

    int zeroPresent = ( ( d_inData[tIndx] >> bit ) & 1 ) == 0 ? 1 : 0;
    d_indxBuff[tIndx] = zeroPresent;

    __syncthreads();

    kernel_ex_sum_scan_blelloch( d_indxBuff, d_scanBuff, size, wsize );

    __syncthreads();

    int _numZeros = d_scanBuff[size - 1] + d_indxBuff[size - 1];

    int _indxOpt1 = d_scanBuff[tIndx];
    int _indxOpt2 = _numZeros + tIndx - _indxOpt1;

    int _indxOpt = ( zeroPresent == 1 ) ? _indxOpt1 : _indxOpt2;

    d_outData[_indxOpt] = d_inData[tIndx];

    __syncthreads();

    d_inData[tIndx] = d_outData[tIndx];
}


int main()
{


    for ( int q = 0; q < VECT_SIZE; q++ )
    {
        //h_arr[q] = h_arrOut[q] = RAND_INT( VECT_SIZE );
        h_arr[q] = h_arrOut[q] = VECT_SIZE - q;
    }

    long long int* d_arr;
    cudaMalloc( &d_arr , sizeof( long long int ) * VECT_SIZE );
    cudaMemcpy( d_arr, h_arr, sizeof( long long int ) * VECT_SIZE, cudaMemcpyHostToDevice );

    long long int* d_arrOut;
    cudaMalloc( &d_arrOut, sizeof( long long int ) * VECT_SIZE );
    cudaMemcpy( d_arrOut, h_arrOut, sizeof( long long int ) * VECT_SIZE, cudaMemcpyHostToDevice );

    int* d_indxBuff;
    cudaMalloc( &d_indxBuff , sizeof( int ) * VECT_SIZE );
    cudaMemset( d_indxBuff, 0, sizeof( int ) * VECT_SIZE );

    int* d_scan;
    cudaMalloc( &d_scan, sizeof( int ) * VECT_SIZE );
    cudaMemset( d_scan, 0, sizeof( int ) * VECT_SIZE );

    int nThreads = ( VECT_SIZE < CUDA_THREADS_PER_BLOCK ) ? VECT_SIZE : CUDA_THREADS_PER_BLOCK;
    int nBlocks = ceil( ( ( float ) VECT_SIZE ) / CUDA_THREADS_PER_BLOCK );

    cout << "nBlocks: " << nBlocks << endl;
    cout << "nThreads: " << nThreads << endl;


    int size = VECT_SIZE;
    int nbits = ceil( log2( size ) );
    int wsize = 1 << nbits;

    cout << "size: " << size << endl;
    cout << "wsize: " << wsize << endl;
    cout << "nbits: " << nbits << endl;

#ifdef TEST_SCAN

    for ( int q = 0; q < VECT_SIZE; q++ )
    {
        h_arr[q] = VECT_SIZE - q;
    }    

    cudaMemcpy( d_arr, h_arr, sizeof( long long int ) * VECT_SIZE, cudaMemcpyHostToDevice );


    // test blelloch scan
    kernel_ex_sum_scan_blelloch<<<nBlocks, nThreads>>>( d_arr, d_scan, size, wsize );

    cudaMemcpy( h_arr, d_scan, sizeof( long long int ) * VECT_SIZE, cudaMemcpyDeviceToHost );

    cout << "scan: " << endl;
    printArray( h_arr, VECT_SIZE );

#else

    for ( int bit = 0; bit < nbits; bit++ )
    {
        // Compute exclusive-scan of predicate array
        kernel_radix_sort_partition<<<nBlocks, nThreads>>>( d_arr, 
                                                            d_arrOut,
                                                            d_scan, 
                                                            d_indxBuff,
                                                            bit, size, wsize );
    }

    cudaMemcpy( h_arrOut, d_arrOut, sizeof( long long int ) * VECT_SIZE, cudaMemcpyDeviceToHost );

    cout << "scan: " << endl;
    printArray( h_arrOut, VECT_SIZE );

#endif

    cudaFree( d_arr );
    cudaFree( d_arrOut );
    cudaFree( d_scan );
    cudaFree( d_indxBuff );

    return 0;
}