
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <fstream>
#include <string>
#include <algorithm>
#include <pthread.h>
#include <vector>

using namespace std;

#define VECT_SIZE 100000000
#define NUM_THREADS 2

double g_arr[VECT_SIZE];
double g_arr_bp[VECT_SIZE];

double g_median;
int g_median_indx;

struct PthData
{
    int left;
    int right;
};

void printArray( double* arr, int size )
{
    for ( int q = 0; q < size; q++ )
    {
        cout << arr[q] << " ";
    }

    cout << endl;
}

int partitionHoare( int &left, int &right, int medianIndx = -1 )
{   
    if ( medianIndx != -1 )
    {
        g_arr[medianIndx] = g_arr[left];
        g_arr[left] = g_median;
    }

    int p1 = left + 1;
    int p2 = right;

    while ( true )
    {

        while ( g_arr[p1] < g_arr[left] )
        {
            p1++;
            if ( p1 >= right )
            {
                break;
            }
        }

        while( g_arr[p2] > g_arr[left] )
        {
            p2--;
            if ( p2 <= left )
            {
                break;
            }
        }

        if ( p1 >= p2 )
        {
            break;
        }
        
        // Swap
        double _tmp = g_arr[p1];
        g_arr[p1] = g_arr[p2];
        g_arr[p2] = _tmp;
    }

    // Swap the pivot
    double _tmp = g_arr[left];
    g_arr[left] = g_arr[p2];
    g_arr[p2] = _tmp;

    return p2;
}

void quicksort( int left, int right )
{
    if ( right <= left )
    {
        return;
    }

//    int _mid = partitionLomuto( arr, left, right );
    int _mid = partitionHoare( left, right );
    quicksort( left, _mid - 1 );
    quicksort( _mid + 1, right );
}

void quicksort_parallel_normal( int left, int right )
{
    if ( right <= left )
    {
        return;
    }

    int _mid = partitionHoare( left, right );
    quicksort_parallel_normal( left, _mid - 1 );
    quicksort_parallel_normal( _mid + 1, right );
}

void* Pth_quicksort( void* data )
{
    PthData* _data = ( PthData* ) data;

    quicksort_parallel_normal( _data->left, _data->right );
    
    return NULL;
}

void quicksort_parallel_stage1( int left, int right )
{
    // Create pthreads for each partition
    
    int _mid = partitionHoare( left, right, g_median_indx );

    pthread_t* thread_handles = new pthread_t[2];

    PthData _pt1;
    _pt1.left = left;
    _pt1.right = _mid - 1;

    PthData _pt2;
    _pt2.left = _mid + 1;
    _pt2.right = right;

    pthread_create( &thread_handles[0], NULL, Pth_quicksort, ( void* )&_pt1 );
    pthread_create( &thread_handles[1], NULL, Pth_quicksort, ( void* )&_pt2 );

    pthread_join( thread_handles[0], NULL );
    pthread_join( thread_handles[1], NULL );
    
}

bool isSorted( double* vect )
{
    for ( int q = 0; q < VECT_SIZE - 1; q++ )
    {
        if ( vect[q] > vect[q+1] )
        {
            cout << "q: " << vect[q] << " q + 1: " << vect[q+1] << endl;
            return false;
        }
    }
    return true;
}

int main()
{
    ifstream _file;
    _file.open( "../list.txt" );
    string _line;
    int _count = 0;
    if ( _file.is_open() )
    {
        cout << "file openened" << endl;
        while( getline( _file, _line ) )
        {
            g_arr[_count] = g_arr_bp[_count] = std::stod( _line );
            _count++;
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

    cout << "finished reading file" << endl;

    cout << "sorting serial ************" << endl;

    double t1 = omp_get_wtime();

    //printArray( g_arr, VECT_SIZE );
    quicksort( 0, VECT_SIZE - 1 );
    //printArray( g_arr, VECT_SIZE );

    double t2 = omp_get_wtime();

    double serial_time = t2 - t1;

    cout << "time: " << t2 - t1 << endl;

    bool isOk = isSorted( g_arr );
    cout << "isOk? " << ( isOk ? "yes" : "no" ) << endl;

    cout << "sorting parallel **********" << endl;

    for ( int q = 0; q < VECT_SIZE; q++ )
    {
        g_arr[q] = g_arr_bp[q];
    }    

    t1 = omp_get_wtime();

    nth_element( g_arr_bp, g_arr_bp + VECT_SIZE / 2, g_arr_bp + VECT_SIZE - 1 );
    g_median = g_arr_bp[VECT_SIZE / 2];

    for ( int q = 0; q < VECT_SIZE; q++ )
    {
        if ( g_arr[q] == g_median )
        {
            g_median_indx = q;
            break;
        }
    }

    //printArray( g_arr, VECT_SIZE );
    quicksort_parallel_stage1( 0, VECT_SIZE - 1 );
    //printArray( g_arr, VECT_SIZE );

    t2 = omp_get_wtime();

    double parallel_time = t2 - t1;

    cout << "time: " << t2 - t1 << endl;

    isOk = isSorted( g_arr );
    cout << "isOk? " << ( isOk ? "yes" : "no" ) << endl;

    cout << "finished sorting" << endl;

    cout << "speedup: " << serial_time / parallel_time << endl;
    cout << "efficiency: " << ( serial_time / parallel_time ) / NUM_THREADS << endl;

    return 0;
}

