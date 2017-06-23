

#include <iostream>
#include <pthread.h>

using namespace std;

void* Thread_sum( void* rank );

double sum = 0.0;
double pi = 0.0;

int current_executor = 0;

#define N 1000000
#define NUM_THREADS 10

int main()
{
    pthread_t* _thread_handles = new pthread_t[NUM_THREADS];

    current_executor = 0;

    long q;
    for ( q = 0; q < NUM_THREADS; q++ )
    {
        pthread_create( &_thread_handles[q], NULL, Thread_sum, ( void* )q );
    }


    for ( q = 0; q < NUM_THREADS; q++ )
    {
        pthread_join( _thread_handles[q], NULL );
    }
    
    delete[] _thread_handles;

    pi = 4.0 * sum;

    cout << "pi: " << pi << endl;

    return 0;
}



void* Thread_sum( void* rank )
{
    long local_id = ( long )rank;
    
    int local_chunk = N / NUM_THREADS;
    int local_start_indx = local_id * local_chunk;
    int local_final_indx = ( local_id + 1 ) * local_chunk - 1;
        
    int _q;
    float factor;
    factor = ( local_start_indx % 2 ) ? 1 : -1;
    double local_sum = 0.0;
    for ( _q = local_start_indx; _q <= local_final_indx; _q++ )
    {
        factor = -factor;
        local_sum += factor * ( 1. / ( 2 * _q + 1 ) );
    }

    while( current_executor != local_id );// busy waiting
    sum += local_sum;
    current_executor = ( current_executor + 1 ) % NUM_THREADS;

    return NULL;
}
