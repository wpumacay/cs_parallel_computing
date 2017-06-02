
#include <iostream>
#include <omp.h>

#define NUM_THREADS 10

void calcChunk( int numSteps, double *sum );

int main()
{
    
    double sign = 1.0;
    double sum = 0.0;
    int numSteps = 1000;
    // Version 1 : Decoupling ****************
    omp_set_num_threads( NUM_THREADS );
    #pragma omp parallel
    {
        calcChunk( numSteps, &sum );
    }

    double pi = 4 * sum;
    std::cout << "pi_v1: " << pi << std::endl;
    // ***************************************
    // Version 2 : parallel for construct ****
    int k;
    sum = 0.0;
    pi = 0.0;
    #pragma omp parallel for num_threads( NUM_THREADS ) \
    reduction( +: sum ) private( k, sign )
    for( k = 0; k < numSteps; k++ )
    {
        sign = ( k % 2 == 0 ) ? 1.0 : -1.0;
        sum += sign / ( 2 * k + 1 );
    }
    pi = 4 * sum;
    std::cout << "pi_v2: " << pi << std::endl;
    // ***************************************
    return 0;
}

void calcChunk( int numSteps, double *sum )
{
    int t_indx = omp_get_thread_num();
    int t_total = omp_get_num_threads();
    
    // zone to work with
    double _result = 0.0;
    int _chunkSize = numSteps / t_total;
    int _i_start = 0 + t_indx * _chunkSize;
    int _i;
    double _frac = 0.0;
    double _sign = 1.0;
    int _current_pos = 0;
    for ( _i = 0; _i < _chunkSize; _i++ )
    {
        _current_pos = _i_start + _i;
        _sign = ( _current_pos % 2 == 0 ? 1.0 : -1.0 );
        _frac = _sign / ( 2 * ( _current_pos ) + 1 );
        _result += _frac;
    }
    
    #pragma omp critical
    *sum += _result;

}
