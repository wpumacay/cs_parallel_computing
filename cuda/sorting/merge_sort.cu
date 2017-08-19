

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

using namespace std;

#define VECT_SIZE 100000000
#define THREADS_PER_BLOCK 500
#define MIN_WORK_SIZE 10
#define BLOCK_MEM_SIZE 5000 /* THREADS_PER_BLOCK * MIN_WORK_SIZE */

#define DEBUG 1
#define ENABLE_SECOND_STAGE 1

float g_arr[VECT_SIZE];

void applyWorkDivision( int& nBlocks, int& nThreads, int& curr_size, int& step );
bool hasFinished( int step );

__global__ void kernel_sort( float* d_data )
{
	// Each thread should work in this stage in MIN_WORK_SIZE ...
	// items ( 10 in this case ). 
	// For each block, lets copy the corresponding working part into ...
	// shared data

	__shared__ float _blockWorkMemory[BLOCK_MEM_SIZE];

	int _globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int _localIdx = threadIdx.x;

	// Each thread copies its 10 elements into the shared block
	for ( int q = 0; q < MIN_WORK_SIZE; q++ )
	{
		_blockWorkMemory[_localIdx * MIN_WORK_SIZE + q] = d_data[_globalIdx * MIN_WORK_SIZE + q];
	}

	__syncthreads();

	// Do a simple insertion sort here
	for ( int q = 0; q < MIN_WORK_SIZE; q++ )
	{
		for ( int p = q; p > 0; p-- )
		{
			if ( _blockWorkMemory[_localIdx * MIN_WORK_SIZE + p] < 
				 _blockWorkMemory[_localIdx * MIN_WORK_SIZE + p - 1] )
			{
				float _tmp = _blockWorkMemory[_localIdx * MIN_WORK_SIZE + p];
				_blockWorkMemory[_localIdx * MIN_WORK_SIZE + p] = _blockWorkMemory[_localIdx * MIN_WORK_SIZE + p - 1];
				_blockWorkMemory[_localIdx * MIN_WORK_SIZE + p - 1] = _tmp;
			}
			else
			{
				break;
			}
		}
	}

	// Copy the data back to the global buffer
	for ( int q = 0; q < MIN_WORK_SIZE; q++ )
	{
		d_data[_globalIdx * MIN_WORK_SIZE + q] = _blockWorkMemory[_localIdx * MIN_WORK_SIZE + q];
	}
	__syncthreads();
}

__global__ void kernel_merge( float* d_data, float* d_data_aux, int curr_size )
{
	long int _globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

	long int _left = _globalIdx * 2 * curr_size;
	long int _right = min( (long int)(_left + 2 * curr_size - 1), (long int )(VECT_SIZE - 1) );
	long int _mid = _left + curr_size - 1;

	//if ( _right < _mid )
	//{
	//	_mid = ( _left + _right ) / 2;
	//}
	if ( _right >= _mid )
	{
		long int q;

		// Copy the necessary data into the aux array
		for ( q = _left; q <= _right; q++ )
		{
			d_data_aux[q] = d_data[q];
		}

	    long int p1 = _left;
	    long int p2 = _mid + 1;

	    for ( q = _left; q <=_right; q++ )
	    {
	        if ( p1 > _mid )
	        {
	            d_data[q] = d_data_aux[p2];
	            p2++;
	        }
	        else if ( p2 > _right )
	        {
	            d_data[q] = d_data_aux[p1];
	            p1++;
	        }
	        else if ( d_data_aux[p2] < d_data_aux[p1] )
	        {
	            d_data[q] = d_data_aux[p2];
	            p2++;
	        }
	        else
	        {
	            d_data[q] = d_data_aux[p1];
	            p1++;
	        }
	    }
	}

}

__global__ void kernel_merge_n2( float* d_data, int curr_size )
{
	long int _globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

	long int _left = _globalIdx * 2 * curr_size;
	long int _right = min( (long int)(_left + 2 * curr_size - 1), (long int )(VECT_SIZE - 1) );
	long int _mid = _left + curr_size - 1;

	if ( _right > _mid )
	{
		if ( d_data[_mid] < d_data[_right] )
		{
			return;
		}

		long int q;

	    long int p1 = _left;
	    long int p2 = _mid + 1;

		while( p1 <= _mid && p2 <= _right )
		{
			if ( d_data[p1] < d_data[p2] )
			{
				p1++;
			}
			else
			{
				float _tmp = d_data[p2];

				for ( q = p2; q > p1; q-- )
				{
					float _tmp2 = d_data[q];
					d_data[q] = d_data[q - 1];
					d_data[q - 1] = _tmp2;
				}

				d_data[p1] = _tmp;

				p1++;p2++;_mid++;
			}
		}
			
	}
}

void printArray( float* arr, int size = VECT_SIZE, int offset = 0 )
{
    for ( int q = 0; q < size; q++ )
    {
        cout << arr[q + offset] << " ";
    }

    cout << endl;
}

bool isSorted( float* vect, int size = VECT_SIZE, int offset = 0 )
{
    for ( int q = 0; q < size - 1; q++ )
    {
        if ( vect[q + offset] > vect[q + offset + 1] )
        {
            cout << "q: " << vect[q + offset] << " q + 1: " << vect[q + offset + 1] << endl;
            return false;
        }
    }
    return true;
}


int main()
{

	cout << "sizeof(float): " << sizeof( float ) << endl;
	cout << "sizeof(double): " << sizeof( double ) << endl;

	// Read input file ******************************

    ifstream _file;
    _file.open( "../../list.txt" );
    string _line;
    int _count = 0;
    int _millions = 0;
    if ( _file.is_open() )
    {
        cout << "file openened" << endl;
        while( getline( _file, _line ) )
        {
            g_arr[_count] = std::stof( _line );
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

    cout << "creating buffers ..." << endl;

    float* d_buff;
    cudaMalloc( ( void ** ) &d_buff, sizeof( float ) * VECT_SIZE );
    cudaMemcpy( d_buff, g_arr, sizeof( float ) * VECT_SIZE, cudaMemcpyHostToDevice );

    float* d_buff_aux;
    cudaMalloc( ( void ** ) &d_buff_aux, sizeof( float ) * VECT_SIZE );

    int _step = 0;
    int _nBlocks, _nThreads;

    // First stage, sort leafs of size 10 ***************************************************
    {
    	cout << "first stage ..." << endl;

    	_nThreads = THREADS_PER_BLOCK; // 500
    	_nBlocks = ( VECT_SIZE / THREADS_PER_BLOCK ) / MIN_WORK_SIZE; // 20000
    	// Each thread should work on 10 elements
    	kernel_sort<<<_nBlocks, _nThreads>>>( d_buff );

    	_step++;

    	cout << "finished first stage" << endl;
    }
    // End first stage **********************************************************************
    #ifdef ENABLE_SECOND_STAGE
    // Second stage, merge parts of increasing size *****************************************
    cout << "started second stage ..." << endl;

    int _curr_size = MIN_WORK_SIZE;
    _nThreads = THREADS_PER_BLOCK;
    _nBlocks = ( VECT_SIZE / THREADS_PER_BLOCK ) / MIN_WORK_SIZE;

    while( true )
    {
    	applyWorkDivision( _nBlocks, _nThreads, _curr_size, _step );

    	//kernel_merge<<<_nBlocks,_nThreads>>>( d_buff, d_buff_aux, _curr_size );
    	kernel_merge_n2<<<_nBlocks,_nThreads>>>( d_buff, _curr_size );

		_curr_size *= 2;
		_step++;
		cout << "step: " << _step << endl;
    	//if ( _curr_size >= VECT_SIZE )
    	if ( _step >= 1 )
    	{
    		break;
    	}
    }

    cout << "finished second stage" << endl;
    // **************************************************************************************

	cout << "_curr_size: " << _curr_size << endl;
	cout << "_nThreads: " << _nThreads << endl;
	cout << "_nBlocks: " << _nBlocks << endl;

    #endif
	/*
	cout << "copying back the data ..." << endl;
	cudaMemcpy( g_arr, d_buff, sizeof( float ) * VECT_SIZE, cudaMemcpyDeviceToHost );
	cout << "finished copying back the data" << endl;
	printArray( g_arr, 20, 0 );
	printArray( g_arr, 20, VECT_SIZE - 21 );
	bool _ok = isSorted( g_arr, _curr_size );
	cout << "issorted: " << ( _ok ? "yes" : "no" ) << endl;
	*/

   	cudaFree( d_buff );
   	cudaFree( d_buff_aux );


	return 0;
}


void applyWorkDivision( int& nBlocks, int& nThreads, int& curr_size, int& step )
{
	long int _size = ( (long int) ( 2 * curr_size ) ) * ((long int) THREADS_PER_BLOCK);
	if ( _size < (long int)VECT_SIZE )
	{
		// Need more than THREADS_PER_BLOCK threads to merge this
		nBlocks = ceil( ( VECT_SIZE / THREADS_PER_BLOCK ) / ( float ) ( 2 * curr_size ) );
		nThreads = THREADS_PER_BLOCK;
	}
	else
	{
		// Just use one block and some threads in block 1 are needed to merge this
		nBlocks = 1;
		nThreads = ceil( VECT_SIZE / ( float ) ( 2 * curr_size  ) );
	}
}