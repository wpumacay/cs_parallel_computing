
#include <mpi.h>
#include <iostream>

using namespace std;

#define MASTER_RANK 0
#define MATRIX_DIM 20

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
    
    int _comm_sz;
    int _my_rank;


    MPI_Init( NULL, NULL );
    MPI_Comm_size( MPI_COMM_WORLD, &_comm_sz );
    MPI_Comm_rank( MPI_COMM_WORLD, &_my_rank );
    
    // Each process will handle MATRIX_DIM / _comm_sz rows
    // And doesn't need all the matrix, just some rows
    int _local_n = MATRIX_DIM / _comm_sz;

    double *lA = new double[_local_n * MATRIX_DIM];
    double *v = new double[MATRIX_DIM];
    double *w = new double[_local_n];

    // Master process loads the matrix A and vector v
    if ( _my_rank == MASTER_RANK )
    {
        double *A = new double[MATRIX_DIM * MATRIX_DIM];
        for ( int x = 0; x < MATRIX_DIM; x++ )
        {
            for ( int y = 0; y < MATRIX_DIM; y++ )
            {
                A[x + y * MATRIX_DIM] = x + y + 1;
            }
        }

        v = new double[MATRIX_DIM];
        for ( int q = 0; q < MATRIX_DIM; q++ )
        {
            v[q] = ( ( double )( q + 1 ) );
        }

        MPI_Scatter( A, _local_n * MATRIX_DIM, MPI_DOUBLE,
                     lA, _local_n * MATRIX_DIM, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );

        delete[] A;
    }
    else
    {
        MPI_Scatter( NULL, _local_n * MATRIX_DIM, MPI_DOUBLE,
                     lA, _local_n * MATRIX_DIM, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );
    }

    MPI_Bcast( v, MATRIX_DIM, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD );
 
    for ( int q = 0; q < _local_n; q++ )
    {
        w[q] = 0.0;
        for ( int k = 0; k < MATRIX_DIM; k++ )
        {
            w[q] += lA[k + q * MATRIX_DIM] * v[k];
        }
    }

    double *wData = new double[MATRIX_DIM];

    MPI_Allgather( w, _local_n, MPI_DOUBLE, wData, _local_n, MPI_DOUBLE, MPI_COMM_WORLD );

    if ( _my_rank == MASTER_RANK )
    {
        printvector( wData, MATRIX_DIM );
    }

    delete[] lA;
    delete[] v;
    delete[] wData;

    MPI_Finalize();
    return 0;
}
