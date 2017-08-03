

#include <iostream>
#include <string>
#include <mpi.h>


using namespace std;

#define MASTER_RANK 0

double Trap( double a, double b, int n, double h )
{
    double _res = 0.0;
    for ( int q = 0; q < n; q++ )
    {
        double x = a + q * h;
        _res += x * x;
    }
    _res *= h;

    return _res;
}

int main()
{

    int _my_rank;
    int _comm_size;
    int n = 100000000;
    int _local_n;

    double a = 0.0;
    double b = 1.0;
    double h, _local_a, _local_b;
    double _local_res, _total_res;
    

    MPI_Init( NULL, NULL );
    MPI_Comm_size( MPI_COMM_WORLD, &_comm_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &_my_rank );

    h = ( b - a ) / n;
    _local_n = n / _comm_size;

    _local_a = a + _my_rank * _local_n * h;
    _local_b = _local_a + _local_n * h;
    _local_res = Trap( _local_a, _local_b, _local_n, h );

    if ( _my_rank != MASTER_RANK )
    {
        MPI_Send( &_local_res, 1, MPI_DOUBLE, MASTER_RANK, 0, MPI_COMM_WORLD );
    }
    else
    {
        _total_res += _local_res;
    
        for ( int q = 1; q < _comm_size; q++ )
        {
            MPI_Recv( &_local_res, 1, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            _total_res += _local_res;
        }
    }


    if ( _my_rank == MASTER_RANK )
    {
        cout << "area: " << _total_res << endl;
    }


    MPI_Finalize();
    return 0;
}
















