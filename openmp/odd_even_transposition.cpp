

#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

#define VEC_SIZE 20000

namespace odd_even_sort
{
    
    void sort( vector<int> &pVec )
    {
        int i,phase;
        for( phase = 0; phase < pVec.size(); phase++ )
        {
            if ( phase % 2 == 0 )
            {
                for( i = 1; i < pVec.size(); i += 2 )
                {
                    if ( pVec[i-1] > pVec[i] )
                    {
                        int _temp = pVec[i];
                        pVec[i] = pVec[i-1];
                        pVec[i-1] = _temp;
                        /// cout << "swapped: " << pVec[i] << " - " << pVec[i-1] << endl;
                    }
                }
            }
            else
            {
                for( i = 1; i < pVec.size() - 1; i += 2 )
                {
                    if ( pVec[i] > pVec[i+1] )
                    {
                        int _temp = pVec[i];
                        pVec[i] = pVec[i+1];
                        pVec[i+1] = _temp;
                        /// cout << "swapped: " << pVec[i] << " - " << pVec[i+1] << endl;
                    }
                }
            }
        } 
    }

    void parallelSort( vector<int> &pVec )
    {
        int i,phase;
        #pragma omp parallel num_threads(10) default(none) private( i, phase ) shared( pVec )
        for( phase = 0; phase < pVec.size(); phase++ )
        {
            if ( phase % 2 == 0 )
            {
                #pragma omp for 
                for( i = 1; i < pVec.size(); i += 2 )
                {
                    if ( pVec[i-1] > pVec[i] )
                    {
                        int _temp = pVec[i];
                        pVec[i] = pVec[i-1];
                        pVec[i-1] = _temp;
                        /// cout << "swapped: " << pVec[i] << " - " << pVec[i-1] << endl;
                    }
                }
            }
            else
            {
                #pragma omp for
                for( i = 1; i < pVec.size() - 1; i += 2 )
                {
                    if ( pVec[i] > pVec[i+1] )
                    {
                        int _temp = pVec[i];
                        pVec[i] = pVec[i+1];
                        pVec[i+1] = _temp;
                        /// cout << "swapped: " << pVec[i] << " - " << pVec[i+1] << endl;
                    }
                }
            }
        } 
    }


    void printVec( vector<int> &pVec )
    {
        cout << "[ ";
        int q;
        for( q = 0; q < pVec.size(); q++ )
        {
            cout << pVec[q] << " ";
        }
        cout << "]" << endl;
    }

}


int main()
{
    vector<int> testVect;
    int q;
    for( q = 0; q < VEC_SIZE; q++ )
    {
        testVect.push_back( q + 1 );
    }

    random_shuffle( testVect.begin(), testVect.end() );
    
    /// odd_even_sort::printVec( testVect );
    double t1 = omp_get_wtime();
    odd_even_sort::sort( testVect );
    double t2 = omp_get_wtime();
    /// odd_even_sort::printVec( testVect );

    double dt_serial = t2 - t1;
    
    random_shuffle( testVect.begin(), testVect.end() );

    /// odd_even_sort::printVec( testVect );
    t1 = omp_get_wtime();
    odd_even_sort::parallelSort( testVect );
    t2 = omp_get_wtime();
    /// odd_even_sort::printVec( testVect );

    double dt_parallel = t2 - t1;
    
    double speedup = dt_serial / dt_parallel;
    
    cout << "speedup: " << speedup << endl;

    return 0;
}
