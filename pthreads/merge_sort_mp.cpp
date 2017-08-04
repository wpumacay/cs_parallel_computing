


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
void merge( T vect, T vect_aux, long int _left, long int _mid, long int _right );

enum DataType
{
    ARRAY,
    VECTOR
};

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
void merge_sort( T vect, T vect_aux )
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
            merge<T>( vect, vect_aux, _left_start, _mid, _right_end );
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

}

template<class T>
void merge_sort_parallel( T vect, T vect_aux, DataType type )
{
    long int _curr_size;
    long int _left_start;
    #pragma omp parallel shared( vect,vect_aux ) private( _curr_size, _left_start )
    for ( _curr_size = 1; _curr_size <= VECT_SIZE - 1; _curr_size = 2 * _curr_size )
    {
        #pragma omp for
        for ( _left_start = 0; _left_start < VECT_SIZE - 1; _left_start += 2 * _curr_size )
        {
            long int _mid = _left_start + _curr_size - 1;
            
            long int _right_end = min( _left_start + 2 * _curr_size - 1, (long int)( VECT_SIZE - 1 ) );
            if ( _right_end < _mid )
            {
                _mid = ( _left_start + _right_end ) / 2;
            }
            merge<T>( vect, vect_aux, _left_start, _mid, _right_end );
        }
    }

}

template<class T>
void merge( T vect, T vect_aux, long int _left, long int _mid, long int _right )
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
    _file.open( "list.txt" );
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

    cout << "sorting parallel ..." << endl;

    _t1 = omp_get_wtime();
    merge_sort<double*>( g_array_serial, g_array_aux );
    _t2 = omp_get_wtime();

    double serialtime = _t2 - _t1;

    cout << "time: " << serialtime << endl;

    cout << "sorting parallel ..." << endl;

    _t1 = omp_get_wtime();
    merge_sort_parallel<double*>( g_array, g_array_aux, ARRAY );
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
