
#include <iostream>
#include <omp.h>

void integrateZone( double a, double b, int numSteps, double *sum );

struct ThreadInfo
{
    int id;
    int total;
};

#define NUM_THREADS 4

int main()
{
    // sequential calculation
    double sum = 0;
    double area = 0;
    int numSteps = 100000;
    double a = 0.0;
    double b = 1.0;
    double dx = ( b - a ) / numSteps;
    
    int q;
    for ( q = 0; q < numSteps; q++ )
    {
        double x_q = a + ( q + 0.5 ) * dx;
        double f_xq = 4.0 / ( 1 + x_q * x_q );
        sum += f_xq;
    }

    area = sum * dx;

    std::cout << "SEQUENTIAL CALCULATION *********" << std::endl;
    std::cout << "pi: " << area << std::endl;
    std::cout << "********************************" << std::endl;

    // parallel calculation version 1 ( cummulative into one implementation )

    sum = 0;
    area = 0;
    
    omp_set_num_threads( NUM_THREADS );

    #pragma omp parallel
    {
        integrateZone( a, b, numSteps, &area );
    }

    std::cout << "PARALLEL CALCULATION V1 *******" << std::endl;
    std::cout << "pi: " << area << std::endl;
    std::cout << "*******************************" << std::endl;

    // parallel calculation version 2 ( separate calculation using a buffer )
    
    int i, nthreads;
    double pi,g_sum[NUM_THREADS];

    double step = ( b - a ) / numSteps;
    omp_set_num_threads( NUM_THREADS );

    #pragma omp parallel
    {
        int k, local_id, local_nthrds;
        double x;
        local_id = omp_get_thread_num();
        local_nthrds = omp_get_num_threads();
        if ( local_id == 0 )
        {
            // only the master will set the global nthreads
            nthreads = local_nthrds;
        }
        // make every thread work on part of the data
        for ( k = local_id, g_sum[local_id] = 0.0;
              k < numSteps; k = k + local_nthrds )
        {
            x = a + ( k + 0.5 ) * step;
            g_sum[local_id] += 4.0 / ( 1.0 + x * x );
        }
    }

    for ( i = 0, pi = 0.0; i < nthreads; i++ )
    {
        pi += g_sum[i] * step;
    }
    std::cout << "PARALLEL CALCULATION V2 *******" << std::endl;
    std::cout << "pi: " << pi << std::endl;
    std::cout << "*******************************" << std::endl;

    // parallel calculation version 3 ( using parallel for )

    sum = 0;
    area = 0;
    numSteps = 100000;
    a = 0.0;
    b = 1.0;
    dx = ( b - a ) / numSteps;
    
    #pragma omp parallel for num_threads( NUM_THREADS ) private( q ) reduction(+:sum)
    for ( q = 0; q < numSteps; q++ )
    {
        double x_q = a + ( q + 0.5 ) * dx;
        double f_xq = 4.0 / ( 1 + x_q * x_q );
        sum += f_xq;
    }

    area = sum * dx;

    std::cout << "PARALLEL CALCULATION V3 ********" << std::endl;
    std::cout << "pi: " << area << std::endl;
    std::cout << "********************************" << std::endl;

    return 0;
}


void integrateZone( double a, double b, int numSteps, double *totalArea )
{
    // Each one of these blocks is a thread ...
    // which should carry out some zone integration
    // This implementation generates a race condition, so ...
    // must use a critical section for the global sum variable

    // Extract the thread information
    ThreadInfo tInfo;
    tInfo.id = omp_get_thread_num();
    tInfo.total = omp_get_num_threads();

    // Calculate some info required
    double step = ( b - a ) / numSteps;

    // calculate the local zone parameters
    int local_n = numSteps / tInfo.total;
    double local_a = a + tInfo.id * ( local_n * step );
    
    int q;
    double local_area = 0.0;

    for ( q = 0; q < local_n; q++ )
    {
        double x_q = local_a + q * step;
        double f_xq = 4 / ( 1 + x_q * x_q );
        local_area += f_xq;
    }
    local_area *= step;

    #pragma omp critical
    *( totalArea ) += local_area;
}
