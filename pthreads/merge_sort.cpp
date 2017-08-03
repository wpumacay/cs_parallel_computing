


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
#define VECT_SIZE 10000000

// GLOBAL ARRAY TO SORT
vector<double> g_arr;
vector<double> g_arr_serial;
vector<double> g_arr_aux;

template<class T>
void merge( vector<T> &vect, vector<T> &vect_aux, long int _left, long int _mid, long int _right );

/*
* Function to check if a vector is correctly sorted
*/
template<class T>
bool isSorted( vector<T> &vect )
{
    for ( long int q = 0; q < vect.size() - 1; q++ )
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
void merge_sort( vector<T> &vect, vector<T> &vect_aux )
{
    long int _curr_size;
    long int _left_start;
    
    for ( _curr_size = 1; _curr_size <= vect.size() - 1; _curr_size = 2 * _curr_size )
    {
        for ( _left_start = 0; _left_start < vect.size() - 1; _left_start += 2 * _curr_size )
        {
            long int _mid = _left_start + _curr_size - 1;
            
            long int _right_end = min( _left_start + 2 * _curr_size - 1, (long int)( vect.size() - 1 ) );
            if ( _right_end < _mid )
            {
                _mid = ( _left_start + _right_end ) / 2;
            }
            merge( vect, vect_aux, _left_start, _mid, _right_end );
        }
    }
}

struct WorkStruct
{
    long rank;
    int left_start;
    int curr_size;
    int right_max;
    int vect_size;
};

void* Pth_merge_sort_work_chunk( void* working_struct )
{
    WorkStruct* _wchunk = ( WorkStruct* ) working_struct;

    long int _left_start = _wchunk->left_start;
    long int _curr_size  = _wchunk->curr_size;
    long int _right_max = _wchunk->right_max;
    long int _vect_size  = _wchunk->vect_size;
    long _rank = _wchunk->rank;

    for ( _left_start = _wchunk->left_start; _left_start <= _right_max; _left_start += 2 * _curr_size )
    {
        long int _mid = _left_start + _curr_size - 1;
            
        long int _right_end = min( _left_start + 2 * _curr_size - 1, _vect_size - 1 );

        if ( _right_end < _mid )
        {
            _mid = ( _left_start + _right_end ) / 2;
        }
        merge( g_arr, g_arr_aux, _left_start, _mid, _right_end );
    }
}

template<class T>
void merge_sort_parallel( vector<T> &vect, vector<T> &vect_aux )
{
    long int _curr_size;
    long int len_vect = vect.size();

    pthread_t* _thread_handles = new pthread_t[NUM_THREADS];

    for ( _curr_size = 1; _curr_size <= len_vect - 1; _curr_size = 2 * _curr_size )
    {

        long int n_chunks_per_thread = ceil( ( (float)len_vect ) / ( NUM_THREADS * 2 * _curr_size ) );
        //cout << "n_chunks_per_thread: " << n_chunks_per_thread << endl;
        //cout << "chunk_size: " << 2 * _curr_size << endl;
        long q;
        WorkStruct _wchunks[NUM_THREADS];

        for ( q = 0; q < NUM_THREADS; q++ )
        {
            long int _left_start = 0 + q * n_chunks_per_thread * 2 * _curr_size;
            if ( _left_start > len_vect - 1 )
            {
            	continue;
            }
            long int _right_max = min( _left_start + n_chunks_per_thread * 2 * _curr_size - 1, len_vect - 1 );

            _wchunks[q].rank = q;
            _wchunks[q].left_start = _left_start;
            _wchunks[q].curr_size = _curr_size;
            _wchunks[q].right_max = _right_max;
            _wchunks[q].vect_size = vect.size();
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
void merge( vector<T> &vect, vector<T> &vect_aux, long int _left, long int _mid, long int _right )
{
    long int q;

    // prepare the aux array
    for ( q = _left; q <= _right; q++  )
    {
        vect_aux[q] = vect[q];
    }

    long int p1 = _left;
    long int p2 = _mid + 1;

    //cout << "foo?" << endl;
    for ( q = _left; q <=_right; q++ )
    {
        if ( p1 > _mid )
        {
            vect[q] = vect_aux[p2];
            p2++;
        }
        else if ( p2 > _right )
        {
            vect[q] = vect_aux[p1];
            p1++;
        }
        else if ( vect_aux[p2] < vect_aux[p1] )
        {
            vect[q] = vect_aux[p2];
            p2++;
        }
        else
        {
            vect[q] = vect_aux[p1];
            p1++;
        }
    }
}


template<class T>
void print_vector( vector<T> &vect )
{
    // return;
    cout << "[ ";
    for ( long int q = 0; q < vect.size(); q++ )
    {
        cout << vect[q] << " ";
    }
    cout << "]" << endl;
}

int main()
{ 

    for ( int q = 0; q < VECT_SIZE; q++ )
    {
        g_arr.push_back( q );
    }
    std::random_shuffle( g_arr.begin(), g_arr.end() );

    double _t1, _t2;

    cout << "reading from file ... " << endl;

    ifstream _file;
    _file.open( "list.txt" );
    string _line;
    int _count = 0;
    if ( _file.is_open() )
    {
        while( getline( _file, _line ) )
        {
            g_arr.push_back( std::stod( _line ) );
        }
        _file.close();
    }
    else
    {
        cout << "couldn't open file" << endl;
    }

    g_arr_aux = g_arr;
    g_arr_serial = g_arr;

    cout << "finished reading from file, size: " << g_arr.size() << endl;

    cout << "sorting serial   ..." << endl;

    _t1 = omp_get_wtime();
    merge_sort<double>( g_arr_serial, g_arr_aux );
    _t2 = omp_get_wtime();

    cout << "time: " << _t2 - _t1 << endl;

    cout << "sorting parallel ..." << endl;
    _t1 = omp_get_wtime();
    merge_sort_parallel<double>( g_arr, g_arr_aux );
    _t2 = omp_get_wtime();

    cout << "time: " << _t2 - _t1 << endl;

    bool isok = isSorted( g_arr );

    cout << "isok: " << ( isok ? "yes" : "no" ) << endl;

    cout << "finished sorting ****************" << endl;
    
    return 0;
}
