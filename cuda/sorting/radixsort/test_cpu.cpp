
#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

#define VECT_SIZE 100
#define RAND_INT(x) rand() % x

int g_arr[VECT_SIZE];
int g_arrOut[VECT_SIZE];

void printArray( int* arr, int size )
{
    cout << "{ ";
    for ( int q = 0; q < size; q++ )
    {
        cout << arr[q] << " ";
    }
    cout << "}" << endl;
}

namespace radixsort
{

    void partition( int *arr, int *arrOut, int size, int bit );
    void sort( int *arr, int *arrOut, int size );

}

int main()
{
    
    for ( int q = 0; q < VECT_SIZE; q++ )
    {
        g_arr[q] = RAND_INT( VECT_SIZE );
    }
    

    /*
    g_arr[0] = 2; g_arr[1] = 1;
    g_arr[2] = 7; g_arr[3] = 3;
    g_arr[4] = 5; g_arr[5] = 3;
    g_arr[6] = 0; g_arr[7] = 6;
    */
    cout << "array to sort: " << endl;

    printArray( g_arr, VECT_SIZE );

    cout << "sorting ..." << endl;

    radixsort::sort( g_arr, g_arrOut, VECT_SIZE );

    cout << "finished sorting " << endl;

    printArray( g_arr, VECT_SIZE );

    return 0;
}





void radixsort::partition( int *arr, int *arrOut, int size, int bit )
{
    // Simulate parallel version by creating a buffer of each computation
    int *tBuff = new int[size];
    int *tScanBuff = new int[size];
    int *tIndxBuff = new int[size];
    int *tIndxOutBuff = new int[size];

    // Work like each thread
    for ( int tIndx = 0; tIndx < size; tIndx++ )
    {
        // Apply predicate
        tBuff[tIndx] = ( ( arr[tIndx] >> bit ) & 1 ) == 0;
        /* syncthreads here */
    }

    // printArray( tBuff, size );

    // Apply exclusive-scan up to the index it should
    tScanBuff[0] = 0;
    int s = 0;
    for ( int tIndx = 1; tIndx < size; tIndx++ )
    {
        s += tBuff[tIndx - 1];
        tScanBuff[tIndx] = s;
    }/* Here we should use a parallel scan implementation */

    // printArray( tScanBuff, size );

    int numZeros = tScanBuff[size - 1] + tBuff[size - 1];

    // geenrate t arraybuff
    for ( int tIndx = 0; tIndx < size; tIndx++ )
    {
        tIndxBuff[tIndx] = numZeros + tIndx - tScanBuff[tIndx];
    }

    // printArray( tIndxBuff, size );

    // geenrate out arraybuff
    for ( int tIndx = 0; tIndx < size; tIndx++ )
    {
        tIndxOutBuff[tIndx] = !tBuff[tIndx] ? 
                                tIndxBuff[tIndx] : tScanBuff[tIndx];
    }

    // printArray( tIndxOutBuff, size );

    for ( int tIndx = 0; tIndx < size; tIndx++ )
    {
        arrOut[tIndxOutBuff[tIndx]] = arr[tIndx];
    }

    // printArray( arrOut, size );

    for ( int tIndx = 0; tIndx < size; tIndx++ )
    {
        arr[tIndx] = arrOut[tIndx];
    }

    delete[] tBuff;
    delete[] tScanBuff;
    delete[] tIndxBuff;
    delete[] tIndxOutBuff;

    // printArray( arr, size );

    // cin.get();
}

void radixsort::sort( int *arr, int *arrOut, int size )
{
    // max num of bits
    int nbits = ceil( log2( VECT_SIZE ) );
    for ( int q = 0; q < nbits; q++ )
    {
        radixsort::partition( arr, arrOut, size, q );
    }
}