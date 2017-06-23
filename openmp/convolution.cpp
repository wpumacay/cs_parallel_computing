#include <iostream>
#include <omp.h>

#define MATRIX_SIZE 2000
#define WINDOW_SIZE 5
#define DELTA ( WINDOW_SIZE - 1 ) / 2

double matrix[MATRIX_SIZE][MATRIX_SIZE];
double result_matrix[MATRIX_SIZE][MATRIX_SIZE];

void convolution( double* matIn, double* matOut );

void printMatrix( double* mat );

int main()
{
    int x, y;
    for ( y = 0; y < MATRIX_SIZE; y++ )
    {
        for( x = 0; x < MATRIX_SIZE; x++ )
        {
            matrix[y][x] = 1.0;// x + y * MATRIX_SIZE;
        }
    }

    convolution( ( double * )matrix, ( double * )result_matrix );
    /// printMatrix( ( double * )matrix );
    /// printMatrix( ( double * )result_matrix );
}

void printMatrix( double *mat )
{
    int row, col;
    for( row = 0; row < MATRIX_SIZE; row++ )
    {
        for( col = 0; col < MATRIX_SIZE; col++ )
        {
            std::cout << *( mat + row * MATRIX_SIZE + col ) << "\t";
        }
        std::cout << std::endl;
    }
}


void convolution( double* matIn, double* matOut )
{
    /// int my_id = omp_get_thread_num();
    double t1 = omp_get_wtime();

    int row, col;
    for( row = 0; row < MATRIX_SIZE; row++ )
    {
        omp_set_num_threads(2);
        #pragma omp parallel for
        for( col = 0; col < MATRIX_SIZE; col++ )
        {
            int _left = ( col - DELTA >= 0 ? col - DELTA : 0 );
            int _right = ( col + DELTA <= MATRIX_SIZE - 1 ? col + DELTA : MATRIX_SIZE - 1 );
            int _top = ( row - DELTA >= 0 ? row - DELTA : 0 );
            int _bottom = ( row + DELTA <= MATRIX_SIZE - 1 ? row + DELTA : MATRIX_SIZE - 1 );

            int _row, _col;
            double _sum = 0.0;
            int count = 0;
            for ( _row = _top; _row <= _bottom; _row++ )
            {
                for( _col = _left; _col <= _right; _col++ )
                {
                    _sum += *( matIn + _row * MATRIX_SIZE + _col );
                    count++;
                }
            }
            *( matOut + row * MATRIX_SIZE + col ) = _sum / ( WINDOW_SIZE * WINDOW_SIZE );
        }
    }

    double t2 = omp_get_wtime();
    std::cout << "delta: " << t2 - t1 << std::endl;
}
