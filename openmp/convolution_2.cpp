#include <iostream>
#include <omp.h>
#include <cstdlib>

#define MATRIX_SIZE 8000
#define MATRIX_SIZE_2 8000
#define WINDOW_SIZE 5
#define DELTA ( WINDOW_SIZE - 1 ) / 2

float matrix[MATRIX_SIZE_2][MATRIX_SIZE_2];
float result_matrix[MATRIX_SIZE_2][MATRIX_SIZE_2];

float* _matrix;
float* _result_matrix;

float convolution( float* matIn, float* matOut );

void printMatrix( float* mat );

#define NUM_THREADS 4
#define SERIAL_TIME_O0 3.88098
#define SERIAL_TIME 1.24023

using namespace std;

int main()
{
    cout << "init" << endl;

    _matrix = new float[(long long) ( MATRIX_SIZE_2 * MATRIX_SIZE_2 )];
    _result_matrix = new float[(long long) ( MATRIX_SIZE_2 * MATRIX_SIZE_2 )];

    cout << "initialized new arrays" << endl;

    int i,j;
    for ( i = 0; i < MATRIX_SIZE_2; i++ )
    {
        for ( j = 0; j < MATRIX_SIZE_2; j++ )
        {
            *( _matrix + i * MATRIX_SIZE_2 + j ) = 1.0;
        }
    }

    cout << "finished initializing new arrays" << endl;

    int x, y;
    for ( y = 0; y < MATRIX_SIZE; y++ )
    {
        for( x = 0; x < MATRIX_SIZE; x++ )
        {
            matrix[y][x] = 1.0;// x + y * MATRIX_SIZE;
        }
    }

    cout << "calculating convolution" << endl;
    float delta = convolution( ( float * )matrix, ( float * )result_matrix );
    cout << "efficiency: " << ( SERIAL_TIME / delta ) / NUM_THREADS << endl;
    cout << "speedup: " << ( SERIAL_TIME / delta ) << endl;
    // convolution( _matrix, _result_matrix ); 
    cout << "printing matrix" << endl;
    // printMatrix( (float*)matrix );
    // printMatrix( (float*)result_matrix );
}

void printMatrix( float *mat )
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


float convolution( float* matIn, float* matOut )
{
    /// int my_id = omp_get_thread_num();
    float t1 = omp_get_wtime();

    int row, col;
    omp_set_num_threads( NUM_THREADS );
    #pragma omp parallel private( row, col )
    for( row = 0; row < MATRIX_SIZE; row++ )
    {
        // omp_set_num_threads( NUM_THREADS );
        // #pragma omp parallel for
        #pragma omp for
        for( col = 0; col < MATRIX_SIZE; col++ )
        {
            int _left = ( col - DELTA >= 0 ? col - DELTA : 0 );
            int _right = ( col + DELTA <= MATRIX_SIZE - 1 ? col + DELTA : MATRIX_SIZE - 1 );
            int _top = ( row - DELTA >= 0 ? row - DELTA : 0 );
            int _bottom = ( row + DELTA <= MATRIX_SIZE - 1 ? row + DELTA : MATRIX_SIZE - 1 );

            int _row, _col;
            float _sum = 0.0;
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

    float t2 = omp_get_wtime();
    std::cout << "delta: " << t2 - t1 << std::endl;
    return t2 - t1;
}
