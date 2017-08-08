
#include <iostream>
#include <cmath>
#include "cuda_vectorsum.h"

using namespace std;

#define SIZE 20

int main()
{
    int n_arr[SIZE];
    for ( int q = 0; q < SIZE; q++ )
    {
        n_arr[q] = q;
    }

    calculateSum( n_arr, SIZE );

    cout << "nSum: " << n_arr[0] << endl;

    return 0;
}
