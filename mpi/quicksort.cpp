
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>

#include <pthread.h>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define VECT_SIZE   10000000
#define BUCKET_SIZE 5000000
#define NUM_PROCESSES 2
#define NUM_THREADS 2

#define MASTER_RANK 0

double g_arr[VECT_SIZE];
double g_arr_aux[VECT_SIZE];
double g_bucket[BUCKET_SIZE];
double g_bucket_bp[BUCKET_SIZE];

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

void merge( long int _left, long int _mid, long int _right )
{
    long int q;

    // prepare the aux array
    for ( q = _left; q <= _right; q++  )
    {
        g_arr_aux[q] = g_arr[q];
    }

    long int p1 = _left;
    long int p2 = _mid + 1;

    //cout << "foo?" << endl;
    for ( q = _left; q <=_right; q++ )
    {
        if ( p1 > _mid )
        {
            g_arr[q] = g_arr_aux[p2];
            p2++;
        }
        else if ( p2 > _right )
        {
            g_arr[q] = g_arr_aux[p1];
            p1++;
        }
        else if ( g_arr_aux[p2] < g_arr_aux[p1] )
        {
            g_arr[q] = g_arr_aux[p2];
            p2++;
        }
        else
        {
            g_arr[q] = g_arr_aux[p1];
            p1++;
        }
    }
}

int partitionHoare( int &left, int &right, int medianIndx = -1 )
{   
    if ( medianIndx != -1 )
    {
        g_bucket[medianIndx] = g_bucket[left];
        g_bucket[left] = g_median;
    }

    int p1 = left + 1;
    int p2 = right;

    while ( true )
    {

        while ( g_bucket[p1] < g_bucket[left] )
        {
            p1++;
        }

        while( g_bucket[p2] > g_bucket[left] )
        {
            p2--;
        }

        if ( p1 >= p2 )
        {
            break;
        }
        
        // Swap
        double _tmp = g_bucket[p1];
        g_bucket[p1] = g_bucket[p2];
        g_bucket[p2] = _tmp;
    }

    // Swap the pivot
    double _tmp = g_bucket[left];
    g_bucket[left] = g_bucket[p2];
    g_bucket[p2] = _tmp;

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
    double t1,t2;

    int _my_rank;
    int _comm_size;

    MPI_Init( NULL, NULL );
    MPI_Comm_size( MPI_COMM_WORLD, &_comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &_my_rank );

    int N = VECT_SIZE / _comm_size;

    if ( N != BUCKET_SIZE || NUM_PROCESSES != _comm_size )
    {
        cout << "wrong bucket configuration" << endl;
        return -1;
    }

    //double* _my_arr = new double[N];
    //double* _my_arr_bp = new double[N];

    double* _my_arr_bp = new double[N];

    if ( _my_rank == MASTER_RANK )
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
                g_arr[_count] = std::stod( _line );
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

        MPI_Scatter( g_arr, N, MPI_DOUBLE,
                     g_bucket, N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );

        // printArray( g_arr, VECT_SIZE );
    }
    else
    {
        MPI_Scatter( NULL, N, MPI_DOUBLE,
                     g_bucket, N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );
    }

    memcpy( g_bucket_bp, g_bucket, BUCKET_SIZE * sizeof( double ) );
    //memcpy( _my_arr_bp, g_bucket, BUCKET_SIZE * sizeof( double ) );
    

    //if ( _my_rank == MASTER_RANK )
    //{
    //    printArray( g_bucket, N );
    //}
        

    if ( _my_rank == MASTER_RANK )
    {
        t1 = omp_get_wtime();
    }


    nth_element( g_bucket_bp, g_bucket_bp + N / 2, g_bucket_bp + N - 1 );
    g_median = g_bucket_bp[N / 2];

    //nth_element( _my_arr_bp, _my_arr_bp + N / 2, _my_arr_bp + N - 1 );
    //g_median = _my_arr_bp[N / 2];

    for ( int q = 0; q < N; q++ )
    {
        if ( g_bucket[q] == g_median )
        {
            g_median_indx = q;
            break;
        }
    }

    // cout << "started sort" << endl;
    quicksort_parallel_stage1( 0, N - 1 );
    // cout << "finished sort" << endl;
        
    MPI_Gather( g_bucket, N, MPI_DOUBLE,
                g_arr, N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );

    if ( _my_rank == MASTER_RANK )
    {
        //printArray( g_arr, VECT_SIZE );
        merge( 0, N - 1, VECT_SIZE - 1 );
        // printArray( g_arr, VECT_SIZE );
    }
    


    if ( _my_rank == MASTER_RANK )
    {
        t2 = omp_get_wtime();

        double parallel_time = t2 - t1;

        cout << "time: " << t2 - t1 << endl;

        bool isOk = isSorted( g_arr );
        cout << "isOk? " << ( isOk ? "yes" : "no" ) << endl;

        cout << "finished sorting" << endl;
    }


    MPI_Finalize();

    return 0;
}

