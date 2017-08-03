

#include <iostream>
#include <string>
#include <mpi.h>


using namespace std;

#define MASTER_RANK 0

#define VECTOR_SIZE 100

void printvector( double* vect, int size )
{
    cout << "[";
    for ( int q = 0; q < size; q++ )
    {
        cout << vect[q] << " ";
    }
    cout << "]" << endl;
}

int main()
{

    int _my_rank;
    int _comm_size;


    MPI_Init( NULL, NULL );
    MPI_Comm_size( MPI_COMM_WORLD, &_comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &_my_rank );

    double* x;
    double* y;
    double* z;

    int _local_n = VECTOR_SIZE / _comm_size;

    x = new double[_local_n];
    y = new double[_local_n];
    z = new double[_local_n];

    // Lets simulate that only the master read the data
    if ( _my_rank == MASTER_RANK )
    {   
        double *xData = new double[VECTOR_SIZE];
        double *yData = new double[VECTOR_SIZE];

        for ( int q = 0; q < VECTOR_SIZE; q++ )
        {
            xData[q] = ( double ) ( 2 * q + 1 );
            yData[q] = ( double ) ( 2 * q + 2 );
        }
        MPI_Scatter( xData, _local_n, MPI_DOUBLE,
                     x, _local_n, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );
        MPI_Scatter( yData, _local_n, MPI_DOUBLE,
                     y, _local_n, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );

        delete[] xData;
        delete[] yData;
    }
    else
    {
        MPI_Scatter( NULL, _local_n, MPI_DOUBLE,
                     x, _local_n, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );
        MPI_Scatter( NULL, _local_n, MPI_DOUBLE,
                     y, _local_n, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );
   }

    for ( int q = 0; q < _local_n; q++ )
    {
        z[q] = x[q] + y[q];
    }
    
    if ( _my_rank == MASTER_RANK )
    {
        double *zData = new double[VECTOR_SIZE];
        MPI_Gather( z,
                    _local_n,
                    MPI_DOUBLE,
                    zData,
                    _local_n,
                    MPI_DOUBLE,
                    MASTER_RANK,
                    MPI_COMM_WORLD );

        printvector( zData, VECTOR_SIZE );
        delete[] zData;
    }
    else
    {
        MPI_Gather( z,
                    _local_n,
                    MPI_DOUBLE,
                    NULL,
                    _local_n,
                    MPI_DOUBLE,
                    MASTER_RANK,
                    MPI_COMM_WORLD );
    }

    delete[] x;
    delete[] y;
    delete[] z;

    MPI_Finalize();
    return 0;
}
















