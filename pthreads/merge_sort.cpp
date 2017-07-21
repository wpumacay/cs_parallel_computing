


#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <pthread.h>
#include <omp.h>
#include <fstream>
#include <string>
#include <cstdlib>

using namespace std;

#define VECT_SIZE 10000000
#define NUM_THREADS 7

#define SERIAL_TIME 43.218

// #define CUT_SIZE 1000000
#define CUT_SIZE -1

// GLOBAL ARRAY TO SORT
vector<double> g_arr;
vector<double> g_arr_db;

template<class T>
void merge( vector<T> &vect, long int _left, long int _mid, long int _right );

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

template<class T>
void merge_sort( vector<T> &vect )
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
                // cout << "???: " << _left_start << " - " << _mid << " - " << _right_end << endl;
                _mid = ( _left_start + _right_end ) / 2;
            }
            if ( _right_end > vect.size() )
            {
                cout << "????: " << _right_end << endl;
            }
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
    bool test;
};

void* Pth_merge_sort_work_chunk( void* working_struct )
{
    WorkStruct* _wchunk = ( WorkStruct* ) working_struct;

    long int _left_start = _wchunk->left_start;
    long int _curr_size  = _wchunk->curr_size;
    long int _right_max = _wchunk->right_max;
    long int _vect_size  = _wchunk->vect_size;
    long _rank = _wchunk->rank;
    bool _test = _wchunk->test;
    // cout << "vect size: " << _vect_size << endl;
    for ( _left_start = _wchunk->left_start; _left_start <= _right_max; _left_start += 2 * _curr_size )
    {
        long int _mid = _left_start + _curr_size - 1;
            
        long int _right_end = min( _left_start + 2 * _curr_size - 1, _vect_size - 1 );

        if ( _right_end < _mid )
        {
            // cout << "???: " << _left_start << " - " << _mid << " - " << _right_end << endl;
            _mid = ( _left_start + _right_end ) / 2;
        }
        if ( _right_end > _vect_size )
        {
            cout << "????: " << _right_end << endl;
        }
        //if ( _test )
        //{
        //    merge( g_arr_db, _left_start, _mid, _right_end );
        //}
        //else
        //{
            merge( g_arr, _left_start, _mid, _right_end );
        //}
    }
}

template<class T>
void merge_sort_parallel( vector<T> &vect, bool test = false )
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
            _wchunks[q].test = test;

            if ( _right_max > len_vect )
            {
                cout << "?????????" << endl;
            }
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


template<class T>
void merge( vector<T> &vect, long int _left, long int _mid, long int _right )
{
    long int q;

    long int _n1 = _mid - _left + 1;// size of left array
    long int _n2 = _right - _mid;// size of right array

    vector<T> _L, _R;

    for ( q = 0; q < _n1; q++ )
    {
        if ( _left + q < 0 || _left + q >= vect.size() )
        {
            cout << "_left: " << _left << endl;
            cout << "_mid: " << _mid << endl;
            cout << "_right: " << _right << endl;
            cout << "q: " << q << endl;
            cout << "?: " << _left + q << " - " << vect.size() << " - " << vect[_left + q] << endl;
        }
        // cout << "?" << endl;
    	T _elem = vect[_left + q];
        // cout << "????" << endl;
        _L.push_back( _elem );
    }
    for ( q = 0; q < _n2; q++ )
    {
        if ( _mid + q + 1 < 0 || _mid + q + 1 >= vect.size() )
        {
            cout << "_left: " << _left << endl;
            cout << "_mid: " << _mid << endl;
            cout << "_right: " << _right << endl;
            cout << "??: " << _mid + q + 1 << " - " << vect.size() << " - " << vect[_mid + q + 1] << endl;
        }
    	T _elem = vect[_mid + q + 1];
        _R.push_back( _elem );
    }

    long int p1 = 0;
    long int p2 = 0;
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
    cout << "testing serial merge_sort ******" << endl;
    for ( int q = 1; q <= VECT_SIZE; q++ )
    {
        g_arr.push_back( q );
    }
    random_shuffle( g_arr.begin(), g_arr.end() );

    //print_vector( g_arr );
    double _t1 = omp_get_wtime();
    merge_sort<double>( g_arr );
    double _t2 = omp_get_wtime();
    //print_vector( g_arr );

    double serial_time = _t2 - _t1;

    cout << "time: " << _t2 - _t1 << endl;
    cout << "isok: " << ( isSorted<double>( g_arr ) ? "yes" : "no" ) << endl;
    cout << "********************************" << endl;
    cout << "testing parallel merge_sort ****" << endl;

    random_shuffle( g_arr.begin(), g_arr.end() );

    //print_vector( g_arr );
    _t1 = omp_get_wtime();
    merge_sort_parallel<double>( g_arr );
    _t2 = omp_get_wtime();
    //print_vector( g_arr );
   
    double parallel_time = _t2 - _t1;

    cout << "time: " << _t2 - _t1 << endl;

    cout << "speedup: " << serial_time / parallel_time << endl;
    cout << "efficiency: " << ( serial_time / parallel_time ) / ( NUM_THREADS ) << endl;
    cout << "isok: " << ( isSorted<double>( g_arr ) ? "yes" : "no" ) << endl;
    cout << "********************************" << endl;
    
    cout << "reading from file ... " << endl;
    
    g_arr = vector<double>();

    ifstream _file;
    _file.open( "list.txt" );
    string _line;
    int _count = 0;
    if ( _file.is_open() )
    {
        while( getline( _file, _line ) )
        {
            if ( CUT_SIZE != -1 )
            {
                _count++;
                if ( _count > CUT_SIZE )
                {
                    _file.close();
                    break;
                }
            }
            g_arr.push_back( std::stod( _line ) );
        }
    }
    else
    {
        cout << "couldn't open file" << endl;
    }

    vector<double> _arr_db_cpy( g_arr );

    cout << "finished reading from file, size: " << g_arr.size() << endl;
    cout << "sorting serial ..." << endl;

    _t1 = omp_get_wtime();
    merge_sort<double>( _arr_db_cpy );
    _t2 = omp_get_wtime();

    serial_time = _t2 - _t1;
    cout << "time: " << _t2 - _t1 << endl;
    cout << "isok: " << ( isSorted<double>( _arr_db_cpy ) ? "yes" : "no" ) << endl;
    cout << "sorting parallel ..." << endl;
    _t1 = omp_get_wtime();
    merge_sort_parallel<double>( g_arr, true );
    _t2 = omp_get_wtime();

    /// print_vector<double>( g_arr );

    parallel_time = _t2 - _t1;
    cout << "time: " << _t2 - _t1 << endl;

    cout << "speedup: " << serial_time / parallel_time << endl;
    cout << "efficiency: " << ( serial_time / parallel_time ) / ( NUM_THREADS ) << endl;

    cout << "isok?" << endl;
    bool isok = isSorted( g_arr );

    cout << "isok: " << ( isok ? "yes" : "no" ) << endl;

    //cout << "speedup: " << SERIAL_TIME / parallel_time << endl;
    //cout << "efficiency: " << ( SERIAL_TIME / parallel_time ) / ( NUM_THREADS ) << endl;

    cout << "finished sorting ****************" << endl;
    
    return 0;
}
