


#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <pthread.h>
#include <omp.h>
#include <fstream>
#include <string>
#include <cstdlib>

#define RANDOM() ( rand() / ( float )RAND_MAX )

using namespace std;

#define NUM_THREADS 4
#define VECT_SIZE 100000000

// GLOBAL ARRAY TO SORT

double g_array[VECT_SIZE];
double g_array_aux[VECT_SIZE];
double g_array_serial[VECT_SIZE];

template<class T>
void merge( long int _left, long int _mid, long int _right );

/*
* Function to check if a vector is correctly sorted
*/
template<class T>
bool isSorted( T vect )
{
    for ( long int q = 0; q < VECT_SIZE - 1; q++ )
    {
        if ( vect[q] > vect[q+1] )
        {
            cout << "q: " << vect[q] << " q + 1: " << vect[q+1] << endl;
            return false;
        }
    }
    return true;
}

/*
* Function to check if a vector is correctly sorted
*/
template<class T>
void merge_sort()
{
    long int _curr_size;
    long int _left_start;
    
    for ( _curr_size = 1; _curr_size <= VECT_SIZE - 1; _curr_size = 2 * _curr_size )
    {
        for ( _left_start = 0; _left_start < VECT_SIZE - 1; _left_start += 2 * _curr_size )
        {
            long int _mid = _left_start + _curr_size - 1;
            
            long int _right_end = min( _left_start + 2 * _curr_size - 1, (long int)( VECT_SIZE - 1 ) );
            if ( _right_end < _mid )
            {
                _mid = ( _left_start + _right_end ) / 2;
            }
            merge<T>( _left_start, _mid, _right_end );
        }
    }
}

struct WorkStruct
{
    int left_start;
    int curr_size;
    int right_max;
};

void* Pth_merge_sort_work_chunk( void* working_struct )
{
    WorkStruct* _wchunk = ( WorkStruct* ) working_struct;

    long int _left_start = _wchunk->left_start;
    long int _curr_size  = _wchunk->curr_size;
    long int _right_max = _wchunk->right_max;

    for ( _left_start = _wchunk->left_start; _left_start <= _right_max; _left_start += 2 * _curr_size )
    {
        long int _mid = _left_start + _curr_size - 1;
            
        long int _right_end = min( _left_start + 2 * _curr_size - 1, ( ( long int ) VECT_SIZE - 1 ) );

        if ( _right_end < _mid )
        {
            _mid = ( _left_start + _right_end ) / 2;
        }
        merge<double*>( _left_start, _mid, _right_end );
    }
}

template<class T>
void merge_sort_parallel()
{
    long int _curr_size;
    long int len_vect = VECT_SIZE;

    pthread_t* _thread_handles = new pthread_t[NUM_THREADS];

    for ( _curr_size = 1; _curr_size <= len_vect - 1; _curr_size = 2 * _curr_size )
    {

        long int n_chunks_per_thread = ceil( ( (float)len_vect ) / ( NUM_THREADS * 2 * _curr_size ) );
        long q;
        WorkStruct _wchunks[NUM_THREADS];

        for ( q = 0; q < NUM_THREADS; q++ )
        {
            long int _left_start = 0 + q * n_chunks_per_thread * 2 * _curr_size;
            if ( _left_start > VECT_SIZE - 1 )
            {
            	continue;
            }
            long int _right_max = min( _left_start + n_chunks_per_thread * 2 * _curr_size - 1, ( ( long int ) VECT_SIZE - 1 ) );

            _wchunks[q].left_start = _left_start;
            _wchunks[q].curr_size = _curr_size;
            _wchunks[q].right_max = _right_max;
        }

        for ( q = 0; q < NUM_THREADS; q++ )
        {
            pthread_create( &_thread_handles[q], NULL, Pth_merge_sort_work_chunk, ( void * )&_wchunks[q]);
        }

        for ( q = 0; q < NUM_THREADS; q++ )
        {
            pthread_join( _thread_handles[q], NULL );
        }
    }

    delete[] _thread_handles;
}

template<class T>
void merge( long int _left, long int _mid, long int _right )
{
    long int q;

    // prepare the aux array
    // #pragma omp parallel for shared( g_array, g_array_aux, _left, _right ) private( q )
    for ( q = _left; q <= _right; q++  )
    {
        g_array_aux[q] = g_array[q];
    }

    long int p1 = _left;
    long int p2 = _mid + 1;

    //cout << "foo?" << endl;
    for ( q = _left; q <=_right; q++ )
    {
        if ( p1 > _mid )
        {
            g_array[q] = g_array_aux[p2];
            p2++;
        }
        else if ( p2 > _right )
        {
            g_array[q] = g_array_aux[p1];
            p1++;
        }
        else if ( g_array_aux[p2] < g_array_aux[p1] )
        {
            g_array[q] = g_array_aux[p2];
            p2++;
        }
        else
        {
            g_array[q] = g_array_aux[p1];
            p1++;
        }
    }
}


template<class T>
void print_vector( T vect )
{
    // return;
    cout << "[ ";
    for ( long int q = 0; q < VECT_SIZE; q++ )
    {
        cout << vect[q] << " ";
    }
    cout << "]" << endl;
}

int main()
{ 
    double _t1, _t2;

    cout << "reading from file ... " << endl;

    ifstream _file;
    _file.open( "../list.txt" );
    string _line;
    int _count = 0;
    if ( _file.is_open() )
    {
        cout << "file openened" << endl;
        while( getline( _file, _line ) )
        {
            g_array[_count] = std::stod( _line );
            g_array_aux[_count] = g_array_serial[_count] = g_array[_count];
            _count++;
        }
        _file.close();
    }
    else
    {
        cout << "couldn't open file" << endl;
    }

    cout << "finished reading from file, size: " << VECT_SIZE << endl;

    cout << "sorting serial   ..." << endl;

    _t1 = omp_get_wtime();
    merge_sort<double*>();
    _t2 = omp_get_wtime();

    double serialtime = _t2 - _t1;

    cout << "time: " << serialtime << endl;

    cout << "sorting parallel ..." << endl;

    for ( long int q = 0; q < VECT_SIZE; q++ )
    {
        g_array[q] = g_array_serial[q];
    }

    omp_set_num_threads( NUM_THREADS );

    _t1 = omp_get_wtime();
    merge_sort_parallel<double*>();
    _t2 = omp_get_wtime();

    double paralleltime = _t2 - _t1;

    cout << "time: " << paralleltime << endl;

    bool isok = isSorted<double*>( g_array );

    cout << "isok: " << ( isok ? "yes" : "no" ) << endl;

    cout << "speedup: " << serialtime / paralleltime << endl;
    cout << "efficiency: " << serialtime / ( paralleltime * NUM_THREADS ) << endl;

    cout << "finished sorting ****************" << endl;
    
    return 0;
}
