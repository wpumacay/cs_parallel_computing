
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "timer.h"

using namespace std;

#define VECT_SIZE 50000000
#define MAX_NUMBER 100000000.0
#define RAND_INT(x) rand() % x

#define CUDA_THREADS_PER_BLOCK 1024

#define NUM_SUMS VECT_SIZE / CUDA_THREADS_PER_BLOCK

int h_arr[VECT_SIZE];
int h_arrOut[VECT_SIZE];
int h_arrScan[VECT_SIZE];
int h_arrSums[NUM_SUMS];

//#define TEST_SCAN 1

#ifndef TEST_SCAN
    #define USE_THRUST 1
#endif

template<class T>
void printArray( T* arr, int size, int offset = 0 )
{
    cout << "{ ";
    for ( int q = 0; q < size; q++ )
    {
        cout << arr[q + offset] << " ";
    }
    cout << "}" << endl;
}

template<class T>
bool isSorted( T* arr, int size )
{
    for ( int q = 1; q < size; q++ )
    {
        if ( arr[q] < arr[q - 1] )
        {
            cout << "indx: " << q << endl;
            cout << "arr[q]: " << arr[q] << endl;
            return false;
        }
    }

    return true;
}

__global__ void kernel_ex_sum_scan_blelloch_block_stage( int *d_inData, 
                                                         int *d_outData, 
                                                         int *d_blockSum,
                                                         int size )
{
    int localIndx = threadIdx.x;
    int blockIndx = blockIdx.x;
    int globalIndx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int sd_workData[CUDA_THREADS_PER_BLOCK];

    if ( globalIndx < size )
    {
        sd_workData[localIndx] = d_inData[globalIndx];
    }
    else
    {
        sd_workData[localIndx] = 0;
    }

    __syncthreads();

    int _offset = 1;

    // Calculate according to the work size
    int _wsize = CUDA_THREADS_PER_BLOCK;

    // Reduce phase
    for ( int d = _wsize >> 1; d > 0; d >>= 1 )
    {
        if ( localIndx < d )
        {
            int _ai = _offset * ( 2 * localIndx + 1 ) - 1;
            int _bi = _offset * ( 2 * localIndx + 2 ) - 1;

            sd_workData[_bi] += sd_workData[_ai];
        }

        _offset *= 2;
        __syncthreads();
    }
    
    if ( localIndx == 0 )
    {
        d_blockSum[blockIndx] = sd_workData[_wsize - 1];
        sd_workData[_wsize - 1] = 0;
    }

    __syncthreads();

    for ( int d = 1; d < _wsize; d *= 2 )
    {
        _offset >>= 1;
        

        if ( localIndx < d )
        {
            int _ai = _offset * ( 2 * localIndx + 1 ) - 1;
            int _bi = _offset * ( 2 * localIndx + 2 ) - 1;

            int _tmp = sd_workData[_ai];
            sd_workData[_ai] = sd_workData[_bi];
            sd_workData[_bi] += _tmp;
        }

        __syncthreads();
    }

    if ( globalIndx < size )
    {
        d_outData[globalIndx] = sd_workData[localIndx];
    }
}

__global__ void kernel_ex_sum_scan_blelloch_block_sum( int *d_inData,
                                                       int *d_sums,
                                                       int size )
{
    int blockIndx = blockIdx.x;
    int globalIndx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( globalIndx  >= size )
    {
        return;
    }
    int res = 0;
    for ( int q = 0; q < blockIndx; q++ )
    {
        res += d_sums[q];
    }
    d_inData[globalIndx] += res;
}

__global__ void kernel_radix_sort_predicate( int *d_inData,
                                             int *d_indxBuff,
                                             int bit, int size )
{
    int tIndx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tIndx >= size )
    {
        return;
    }

    int zeroPresent = ( ( d_inData[tIndx] >> bit ) & 1 ) == 0 ? 1 : 0;
    d_indxBuff[tIndx] = zeroPresent;
}

__global__ void kernel_radix_sort_partition( int *d_inData,
                                             int *d_outData, 
                                             int *d_indxBuff,
                                             int *d_scanBuff,
                                             int bit, int size )
{
    int tIndx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tIndx >= size )
    {
        return;
    }

    int zeroPresent = ( ( d_inData[tIndx] >> bit ) & 1 ) == 0 ? 1 : 0;

    int _numZeros = d_scanBuff[size - 1] + d_indxBuff[size - 1];

    int _indxOpt1 = d_scanBuff[tIndx];
    int _indxOpt2 = _numZeros + tIndx - _indxOpt1;

    int _indxOpt = ( zeroPresent == 1 ) ? _indxOpt1 : _indxOpt2;

    d_outData[_indxOpt] = d_inData[tIndx];
}

__global__ void kernel_radix_sort_back_swap( int *d_inData,
                                             int *d_outData,
                                             int size)
{
    int tIndx = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tIndx >= size )
    {
        return;
    }

    d_inData[tIndx] = d_outData[tIndx];
}

int main()
{
    cudaDeviceReset();

    // Read input file ******************************

    ifstream _file;
    _file.open( "../../../list.txt" );
    string _line;
    int _count = 0;
    int _millions = 0;
    if ( _file.is_open() )
    {
        cout << "file openened" << endl;
        while( getline( _file, _line ) )
        {
            float _numf = std::stof( _line );
            h_arr[_count] = ( ( int ) ( _numf * MAX_NUMBER ) );
            _count++;
            if ( _count % 1000000 == 0 )
            {
                cout << "millions read: " << _millions++ << endl;
            }
            if ( _count == VECT_SIZE )
            {
                break;
            }
        }
        _file.close();
    }
    else
    {
        cout << "couldn't open file" << endl;
    }

    // **********************************************

    //for ( int q = 0; q < VECT_SIZE; q++ )
    //{
    //    //h_arr[q] = h_arrOut[q] = RAND_INT( VECT_SIZE );
    //    h_arr[q] = h_arrOut[q] = VECT_SIZE - q;
    //}

    int* d_arr;
    cudaMalloc( &d_arr , sizeof( int ) * VECT_SIZE );
    cudaMemcpy( d_arr, h_arr, sizeof( int ) * VECT_SIZE, cudaMemcpyHostToDevice );

    int* d_arrOut;
    cudaMalloc( &d_arrOut, sizeof( int ) * VECT_SIZE );
    cudaMemcpy( d_arrOut, h_arrOut, sizeof( int ) * VECT_SIZE, cudaMemcpyHostToDevice );

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
#ifndef USE_THRUST
    int* d_sums;
    cudaMalloc( &d_sums, sizeof( int ) * nBlocks );
    cudaMemset( d_sums, 0, sizeof( int ) * nBlocks );
#endif
    int size = VECT_SIZE;
    int nbits = ceil( log2( size ) );
    int wsize = 1 << nbits;
    if ( size % wsize == 0 )
    {
        nbits++;
    }
    nbits = ceil( log2( MAX_NUMBER ) );

    cout << "size: " << size << endl;
    cout << "wsize: " << wsize << endl;
    cout << "nbits: " << nbits << endl;

#ifdef TEST_SCAN

    for ( int q = 0; q < VECT_SIZE; q++ )
    {
        h_arrScan[q] = q + 1;
    }    


    cudaMemcpy( d_indxBuff, h_arrScan, sizeof( int ) * VECT_SIZE, cudaMemcpyHostToDevice );

    // test blelloch scan
    kernel_ex_sum_scan_blelloch_block_stage<<<nBlocks, nThreads>>>( d_indxBuff, 
                                                                    d_scan, 
                                                                    d_sums, 
                                                                    size );

    kernel_ex_sum_scan_blelloch_block_sum<<<nBlocks, nThreads>>>( d_scan,
                                                                  d_sums,
                                                                  size );

    cudaMemcpy( h_arrScan, d_scan, sizeof( int ) * VECT_SIZE, cudaMemcpyDeviceToHost );
    cudaMemcpy( h_arrSums, d_sums, sizeof( int ) * nBlocks, cudaMemcpyDeviceToHost );

    //cout << "scan: " << endl;
    //printArray<int>( h_arrScan, VECT_SIZE );
    //printArray<int>( h_arrSums, NUM_SUMS );

#else

    size_t _mem_free;
    size_t _mem_total;

    cudaMemGetInfo( &_mem_free, &_mem_total );

    cout << "free memory: " << _mem_free << endl;
    cout << "total memory: " << _mem_total << endl;

    GpuTimer _timer;
    _timer.start();

    for ( int bit = 0; bit < nbits; bit++ )
    {
        // Compute exclusive-scan of predicate array
        //cout << "bit: " << bit << endl;
        // Compute the predicate
        kernel_radix_sort_predicate<<<nBlocks, nThreads>>>( d_arr,
                                                            d_indxBuff,
                                                            bit, size );
        
        
        thrust::exclusive_scan( thrust::device, d_indxBuff, d_indxBuff + size, d_scan );
        

        /*
        // Compute ex-sum scan of the indexes generated by the predicate
        kernel_ex_sum_scan_blelloch_block_stage<<<nBlocks, nThreads>>>( d_indxBuff,
                                                                        d_scan,
                                                                        d_sums,
                                                                        size );

        
        kernel_ex_sum_scan_blelloch_block_sum<<<nBlocks, nThreads>>>( d_scan,
                                                                      d_sums,
                                                                      size );
        */
        
        //cudaMemcpy( h_arrScan, d_scan, sizeof( int ) * VECT_SIZE, cudaMemcpyDeviceToHost );
        //cudaMemcpy( h_arrSums, d_sums, sizeof( int ) * nBlocks, cudaMemcpyDeviceToHost );

        //cout << "scan - bit: " << bit << endl;
        //printArray<int>( h_arrScan, VECT_SIZE );
        //printArray<int>( h_arrSums, NUM_SUMS );
        
        // sort
           
        kernel_radix_sort_partition<<<nBlocks, nThreads>>>( d_arr,
                                                            d_arrOut,
                                                            d_indxBuff,
                                                            d_scan,
                                                            bit, size );

        kernel_radix_sort_back_swap<<<nBlocks, nThreads>>>( d_arr,
                                                            d_arrOut,
                                                            size );
        
    }

    _timer.stop();

    float _elapsedTime = _timer.elapsed();

    cudaMemcpy( h_arrOut, d_arrOut, sizeof( int ) * VECT_SIZE, cudaMemcpyDeviceToHost );

    
    bool _isOk = isSorted<int>( h_arrOut, VECT_SIZE );
    cout << "isSorted: " << ( _isOk ? "yes" : "no" ) << endl;

    printArray<int>( h_arrOut, 100, 0 );
    printArray<int>( h_arrOut, 100, VECT_SIZE - 101 );
    
    cout << "time: " << _elapsedTime << endl;

    cudaMemGetInfo( &_mem_free, &_mem_total );

    cout << "free memory: " << _mem_free << endl;
    cout << "total memory: " << _mem_total << endl;
    
#endif

    cudaFree( d_arr );
    cudaFree( d_arrOut );
    cudaFree( d_scan );
    cudaFree( d_indxBuff );
#ifndef USE_THRUST
    cudaFree( d_sums );
#endif

    return 0;
}