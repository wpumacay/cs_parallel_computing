

#include <mpi.h>
#include <iostream>
#include <string>

using namespace std;


#define MASTER_ID 0
#define MAX_STR_LEN 100


int main()
{
    string _greeting;
    char _greetingBuff[100];
    int _comm_sz;
    int _my_rank;

    MPI_Init( NULL, NULL );
    MPI_Comm_size( MPI_COMM_WORLD, &_comm_sz );
    MPI_Comm_rank( MPI_COMM_WORLD, &_my_rank );

    cout << "nProcesses: " << _comm_sz << endl;
    
    if ( _my_rank != MASTER_ID )
    {
        _greeting += "Greetings from process ";
        _greeting += std::to_string( _my_rank );
        _greeting += " of ";
        _greeting += std::to_string( _comm_sz );
        _greeting += "!";
    
        MPI_Send( _greeting.c_str(), _greeting.size() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD );
    }
    else
    {
        cout << "Greetings from master" << endl;
        
        for ( int q = 1; q < _comm_sz; q++ )
        {
            MPI_Recv( _greetingBuff, MAX_STR_LEN, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            _greeting = string( _greetingBuff );

            cout << _greeting << endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
