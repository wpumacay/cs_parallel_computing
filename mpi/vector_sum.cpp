

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
    double* res;
    x = new double[VECTOR_SIZE];
    y = new double[VECTOR_SIZE];
    z = new double[VECTOR_SIZE];
    res = new double[VECTOR_SIZE];

    // Lets simulate that only the master read the data
    if ( _my_rank == MASTER_RANK )
    {
        for ( int q = 0; q < VECTOR_SIZE; q++ )
        {
            x[q] = ( double ) ( 2 * q + 1 );
            y[q] = ( double ) ( 2 * q + 2 );
        }
    }
    for ( int q = 0; q < VECTOR_SIZE; q++ )
    {
        res[q] = 0.0;
    }

    MPI_Bcast( x, VECTOR_SIZE, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );
    MPI_Bcast( y, VECTOR_SIZE, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );

    int _local_n = VECTOR_SIZE / _comm_size;
    for ( int q = 0; q < _local_n; q++ )
    {
        z[q + _my_rank * _local_n] = x[q + _my_rank * _local_n] + y[q + _my_rank * _local_n];
    }
    
    MPI_Reduce( z,
                res,
                VECTOR_SIZE,
                MPI_DOUBLE,
                MPI_SUM,
                MASTER_RANK,
                MPI_COMM_WORLD );
    
    if ( _my_rank == MASTER_RANK )
    {
        printvector( res, VECTOR_SIZE );
    }

    delete[] x;
    delete[] y;
    delete[] z;

    MPI_Finalize();
    return 0;
}
















