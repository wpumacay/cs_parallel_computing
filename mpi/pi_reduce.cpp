

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

    MPI_Reduce( &_local_res, &_total_res, 1, MPI_DOUBLE, MPI_SUM,
                MASTER_RANK, MPI_COMM_WORLD );

    if ( _my_rank == MASTER_RANK )
    {
        cout << "area: " << _total_res << endl;
    }


    MPI_Finalize();
    return 0;
}
















