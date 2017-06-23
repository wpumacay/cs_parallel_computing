
#include <iostream>
#include <pthread.h>

using namespace std;

#define MAT_DIM 6
#define THREADS_COUNT 3

void* Pth_mat_vect( void* rank );

float A[MAT_DIM][MAT_DIM] = { {1.,2.,3.,4.,5.,6.},
                              {2.,3.,4.,5.,6.,7.},
                              {3.,4.,5.,6.,7.,8.},
                              {4.,5.,6.,7.,8.,9.},
                              {5.,6.,7.,8.,9.,10.},
                              {6.,7.,8.,9.,10.,11.} };

float v[MAT_DIM] = { 1.,1.,1.,1.,1.,1. };
float y[MAT_DIM] = { 0.,0.,0.,0.,0.,0. };
    

int main()
{
    cout << "serial version **********" << endl;

    int i,j;
    for ( i = 0; i < MAT_DIM; i++ )
    {
        y[i] = 0.0;        
        for ( j = 0; j < MAT_DIM; j++ )
        {
            y[i] += A[i][j] * v[j];
        }

    }
    
    cout << "y=Av-> ";
    for ( i = 0; i < MAT_DIM; i++ )
    {
        cout << y[i] << " ";
    }
    cout << "*************************" << endl;

    cout << "pthreads version ********" << endl;

    pthread_t* _thread_handles = new pthread_t[THREADS_COUNT];
    long q;
    for( q = 0; q < THREADS_COUNT; q++ )
    {
        pthread_create( &_thread_handles[q], NULL, Pth_mat_vect, ( void* ) q );
    }


    for( q = 0; q < THREADS_COUNT; q++ )
    {
        pthread_join( _thread_handles[q], NULL );
    }

    delete[] _thread_handles;

    cout << "y=Av-> ";
    for ( i = 0; i < MAT_DIM; i++ )
    {
        cout << y[i] << " ";
    }
    cout << "*************************" << endl;
   

    return 0;
}


void* Pth_mat_vect( void* rank )
{
    long local_id = ( long ) rank;
    int local_delta = MAT_DIM / THREADS_COUNT;
    int local_first_indx = ( local_id ) * local_delta;
    int local_last_indx = ( local_id + 1 ) * local_delta - 1;

    int _i,_j;

    for ( _i = local_first_indx; _i <= local_last_indx; _i++ )
    {
        y[_i] = 0.0;
        for ( _j = 0; _j < MAT_DIM; _j++ )
        {
            y[_i] += A[_i][_j] * v[_j];
        }
    }
    
    return NULL;
}
