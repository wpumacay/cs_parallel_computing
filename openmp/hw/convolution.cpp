#include <iostream>
#include <omp.h>
#include <cstdlib>

// Using a matrix of size 8000x8000
#define MATRIX_SIZE 8000
// The matrix used as kernel is :
// [ 1  1  1  1  1 ,
//   1  1  1  1  1 ,
//   1  1  1  1  1 ,
//   1  1  1  1  1 ,
//   1  1  1  1  1 ] which applies an average filter
#define WINDOW_SIZE 5
// This parameter is just half the window size in order
// to generate the appropiate region to applythe kernel
#define DELTA ( WINDOW_SIZE - 1 ) / 2

float matrix[MATRIX_SIZE][MATRIX_SIZE];
float result_matrix[MATRIX_SIZE][MATRIX_SIZE];

// Function used to calculate the convolution
float convolution( float* matIn, float* matOut );
// DEbug function used to see if the convolution is done correctly
void printMatrix( float* mat );

#define NUM_THREADS 4 // Using 4 threads. The best time was achieved using 4 threads
#define SERIAL_TIME_O0 3.88098 // Serial time got using no compiler optimizations
#define SERIAL_TIME 1.24023 // Serial time got using O2 compiler optimizations

using namespace std;

int main()
{
    // Initializing the matrix with 1s *******************************
    int x, y;
    for ( y = 0; y < MATRIX_SIZE; y++ )
    {
        for( x = 0; x < MATRIX_SIZE; x++ )
        {
            matrix[y][x] = 1.0;// x + y * MATRIX_SIZE;
        }
    }
    // ***************************************************************
    // Calculating convolution ***************************************
    cout << "calculating convolution ..." << endl;
    float delta = convolution( ( float * )matrix, ( float * )result_matrix );
    cout << "finished calculating convolution" << endl;
    // ***************************************************************
    // Printing the results ******************************************
    cout << "time spent: " << delta << endl;
    cout << "speedup: " << ( SERIAL_TIME / delta ) << endl;
    cout << "efficiency: " << ( SERIAL_TIME / delta ) / NUM_THREADS << endl;
    // cout << "printing matrix" << endl;
    // printMatrix( (float*)matrix );
    // printMatrix( (float*)result_matrix );
    // ***************************************************************
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


/*
* @brief: Convolution function
* @param float* matIn   The input matrix to calculate the convolution in
* @param float* matOut  The matrix used to store the results of the convolution
*/
float convolution( float* matIn, float* matOut )
{
    float t1 = omp_get_wtime();

    int row, col;
    omp_set_num_threads( NUM_THREADS );
    // Create the threads first to reuse them
    #pragma omp parallel private( row, col )
    for( row = 0; row < MATRIX_SIZE; row++ )
    {
        // variant used to see if there is some improvement *****
        // omp_set_num_threads( NUM_THREADS ); 
        // #pragma omp parallel for
        // ******************************************************
        // There is no improvement by applying this method. Just no more threads are created each ...
        // time this for is executed, but the time doesn't improve
        #pragma omp for
        for( col = 0; col < MATRIX_SIZE; col++ )
        {
            // This section calculates the limits to apply to kernel **************************
            int _left = ( col - DELTA >= 0 ? col - DELTA : 0 );
            int _right = ( col + DELTA <= MATRIX_SIZE - 1 ? col + DELTA : MATRIX_SIZE - 1 );
            int _top = ( row - DELTA >= 0 ? row - DELTA : 0 );
            int _bottom = ( row + DELTA <= MATRIX_SIZE - 1 ? row + DELTA : MATRIX_SIZE - 1 );
            // ********************************************************************************
            // This part applies the average filter by adding all elements in the valid region ...
            // and calculating the average over the mask
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
    // Return how much time it takes to calculate the convolution using NUM_THREADS threads
    return t2 - t1;
}
