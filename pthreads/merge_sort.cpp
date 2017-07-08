


#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <pthread.h>
#include <omp.h>

using namespace std;

#define VECT_SIZE 10000000
#define NUM_THREADS 4

// GLOBAL ARRAY TO SORT
vector<int> g_arr;

void merge( vector<int> &vect, int _left, int _mid, int _right );

void merge_sort( vector<int> &vect )
{
    int _curr_size;
    int _left_start;
    
    for ( _curr_size = 1; _curr_size <= vect.size() - 1; _curr_size = 2 * _curr_size )
    {
        for ( _left_start = 0; _left_start < vect.size() - 1; _left_start += 2 * _curr_size )
        {
            int _mid = _left_start + _curr_size - 1;
            
            int _right_end = min( _left_start + 2 * _curr_size - 1, (int)( vect.size() - 1 ) );
            merge( vect, _left_start, _mid, _right_end );
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

    int _left_start = _wchunk->left_start;
    int _curr_size  = _wchunk->curr_size;
    int _right_max = _wchunk->right_max;
    int _vect_size  = _wchunk->vect_size;
    long _rank = _wchunk->rank;

    for ( _left_start = _wchunk->left_start; _left_start <= _right_max; _left_start += 2 * _curr_size )
    {
        int _mid = _left_start + _curr_size - 1;
            
        int _right_end = min( _left_start + 2 * _curr_size - 1, _vect_size - 1 );
        merge( g_arr, _left_start, _mid, _right_end );
    }
}

void merge_sort_parallel( vector<int> &vect )
{
    int _curr_size;
    int len_vect = vect.size();

    pthread_t* _thread_handles = new pthread_t[NUM_THREADS];

    for ( _curr_size = 1; _curr_size <= len_vect - 1; _curr_size = 2 * _curr_size )
    {

        int n_chunks_per_thread = ceil( ( (float)len_vect ) / ( NUM_THREADS * 2 * _curr_size ) );
        //cout << "n_chunks_per_thread: " << n_chunks_per_thread << endl;
        //cout << "chunk_size: " << 2 * _curr_size << endl;
        long q;
        WorkStruct _wchunks[NUM_THREADS];

        for ( q = 0; q < NUM_THREADS; q++ )
        {
            int _left_start = 0 + q * n_chunks_per_thread * 2 * _curr_size;
            if ( _left_start > len_vect - 1 )
            {
            	continue;
            }
            int _right_max = min( _left_start + n_chunks_per_thread * 2 * _curr_size - 1, len_vect - 1 );

            _wchunks[q].rank = q;
            _wchunks[q].left_start = _left_start;
            _wchunks[q].curr_size = _curr_size;
            _wchunks[q].right_max = _right_max;
            _wchunks[q].vect_size = VECT_SIZE;

            //cout << "ls: " << _left_start << endl;
            //cout << "cs: " << _curr_size << endl;
            //cout << "rm: " << _right_max << endl;
            //cout << "----" << endl;

            pthread_create( &_thread_handles[q], NULL, Pth_merge_sort_work_chunk, ( void * )&_wchunks[q]);
        }

        for ( q = 0; q < NUM_THREADS; q++ )
        {
            pthread_join( _thread_handles[q], NULL );
        }
    }

    delete[] _thread_handles;
}



void merge( vector<int> &vect, int _left, int _mid, int _right )
{
    int q;

    int _n1 = _mid - _left + 1;// size of left array
    int _n2 = _right - _mid;// size of right array

    vector<int> _L, _R;

    for ( q = 0; q < _n1; q++ )
    {
        _L.push_back( vect[_left + q] );
    }
    for ( q = 0; q < _n2; q++ )
    {
        _R.push_back( vect[_mid + q + 1] );
    }

    int p1 = 0;
    int p2 = 0;
    q = _left;

    while( p1 < _n1 && p2 < _n2 )
    {
        if ( _L[p1] <= _R[p2] )
        {
            vect[q] = _L[p1];
            p1++;
        }
        else
        {
            vect[q] = _R[p2];
            p2++;
        }
        q++;
    }

    while ( p1 < _n1 )
    {
        vect[q] = _L[p1];
        p1++;
        q++;
    }

    while ( p2 < _n2 )
    {
        vect[q] = _R[p2];
        p2++;
        q++;
    }
}

void print_vector( vector<int> &vect )
{
    // return;
    cout << "[ ";
    for ( int q = 0; q < vect.size(); q++ )
    {
        cout << vect[q] << " ";
    }
    cout << "]" << endl;
}

int main()
{
    cout << "testing serial merge_sort ******" << endl;
    for ( int q = 1; q <= VECT_SIZE; q++ )
    {
        g_arr.push_back( q );
    }
    random_shuffle( g_arr.begin(), g_arr.end() );

    //print_vector( g_arr );
    double _t1 = omp_get_wtime();
    merge_sort( g_arr );
    double _t2 = omp_get_wtime();
    //print_vector( g_arr );

    double serial_time = _t2 - _t1;

    cout << "time: " << _t2 - _t1 << endl;
    cout << "********************************" << endl;
    cout << "testing parallel merge_sort ****" << endl;

    random_shuffle( g_arr.begin(), g_arr.end() );

    //print_vector( g_arr );
    _t1 = omp_get_wtime();
    merge_sort_parallel( g_arr );
    _t2 = omp_get_wtime();
    //print_vector( g_arr );
   
    double parallel_time = _t2 - _t1;

    cout << "time: " << _t2 - _t1 << endl;

    cout << "speedup: " << serial_time / parallel_time << endl;
    cout << "efficiency: " << ( serial_time / parallel_time ) / ( NUM_THREADS ) << endl;
    cout << "********************************" << endl;


    return 0;
}
