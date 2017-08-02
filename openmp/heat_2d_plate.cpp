


#include <iostream>
#include <omp.h>
#include <iomanip>

#define XDIM 4.0
#define YDIM 4.0
#define DH 0.01

#define SIMT 1000.0
#define DT 0.1

#define K 0.1

#define NUM_THREADS 4

void printGrid( double* pGrid, int xDim, int yDim );

using namespace std;

int main()
{

    double t1, t2;

    // cout << std::fixed;
    // cout << std::setprecision( 3 );

    int Nx = XDIM / DH + 1;
    int Ny = YDIM / DH + 1;
    int Nt = SIMT / DT;

    double* u_t = new double[Nx * Ny];
    double* u_t_1 = new double[Nx * Ny];
    
    // Initialize the profile at u_t(x,y,0) = 0.0625 - ( x - 0.5 )**2 - ( y - 0.5 )**2

    for ( int p = 0; p < Nx; p++ )
    {
        for ( int q = 0; q < Ny; q++ )
        {
            double _x = p * DH;
            double _y = q * DH;
            double _uVal  = 0.25 - ( ( _x - 0.5 ) * ( _x - 0.5 ) +
                                     ( _y - 0.5 ) * ( _y - 0.5 ) );
            _uVal = ( _uVal < 0.0001 ) ? 0 : _uVal;
            u_t[q * Nx + p] = _uVal;
            u_t_1[q * Nx + p] = _uVal;
        }
    }

    // printGrid( u_t, Nx, Ny );
    
    // Serial version ************************************
    cout << "calculating serial ... " << endl;
    t1 = omp_get_wtime();
    for ( int t = 0; t < Nt; t++ )
    {
        for ( int y = 0; y < Ny; y++ )
        {
            for ( int x = 0; x < Nx; x++ )
            {
                if ( x == 0 || y == 0 || x == Nx - 1 || y == Ny - 1 )
                {
                    u_t[x + y * Nx] = 0.0;
                }
                else
                {
                    u_t[x + y * Nx] = ( 1 - 4 * K ) * u_t_1[x + y * Nx] +
                                      K * ( u_t_1[( x - 1 ) + y * Nx] + u_t_1[( x + 1 ) + y * Nx] +
                                            u_t_1[x + ( y - 1 ) * Nx] + u_t_1[x + ( y + 1 ) * Nx] );
                }
            }
        }

        // Save back buffer
        for ( int y = 0; y < Ny; y++ )
        {
            for ( int x = 0; x < Nx; x++ )
            {
                u_t_1[x + y * Nx] = u_t[x + y * Nx];
            }
        }
        // cin.get();
        // printGrid( u_t, Nx, Ny );
    }
    t2 = omp_get_wtime();
    double sTime = t2 - t1;
    cout << "time: " << t2 - t1 << endl;
    // printGrid( u_t, Nx, Ny );
    cout << "finished calculating" << endl;
    // ***************************************************

    // Parallel version **********************************
    cout << "calculating parallel ... " << endl;
    omp_set_num_threads( NUM_THREADS );
    t1 = omp_get_wtime();
    for ( int t = 0; t < Nt; t++ )
    {
        #pragma omp parallel for shared(u_t,u_t_1,Nx,Ny)
        for ( int y = 0; y < Ny; y++ )
        {
            for ( int x = 0; x < Nx; x++ )
            {
                if ( x == 0 || y == 0 || x == Nx - 1 || y == Ny - 1 )
                {
                    u_t[x + y * Nx] = 0.0;
                }
                else
                {
                    u_t[x + y * Nx] = ( 1 - 4 * K ) * u_t_1[x + y * Nx] +
                                      K * ( u_t_1[( x - 1 ) + y * Nx] + u_t_1[( x + 1 ) + y * Nx] +
                                            u_t_1[x + ( y - 1 ) * Nx] + u_t_1[x + ( y + 1 ) * Nx] );
                }
            }
        }

        // Save back buffer
        #pragma omp parallel for shared(u_t,u_t_1,Nx,Ny)
        for ( int y = 0; y < Ny; y++ )
        {
            for ( int x = 0; x < Nx; x++ )
            {
                u_t_1[x + y * Nx] = u_t[x + y * Nx];
            }
        }
        // cin.get();
        // printGrid( u_t, Nx, Ny );
    }
    t2 = omp_get_wtime();
    double pTime = t2 - t1;
    cout << "time: " << t2 - t1 << endl;
    // printGrid( u_t, Nx, Ny );
    cout << "finished calculating" << endl;
    // ***************************************************

    cout << "speedup: " << sTime / pTime << endl;
    cout << "efficienty: " << sTime / ( pTime * NUM_THREADS ) << endl;


    delete[] u_t;
    delete[] u_t_1;


    return 0;
}


void printGrid( double *pGrid, int xDim, int yDim )
{

    for ( int y = 0; y < yDim; y++ )
    {
        for ( int x = 0; x < xDim; x++ )
        {
            cout << pGrid[x + y * xDim] << "\t";
        }
        cout << endl;
    }
}

