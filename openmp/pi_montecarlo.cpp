

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#define RANDOM() ( ( double ) rand() ) / RAND_MAX
#define NUM_THREADS 4

using namespace std;

int main()
{


    int _nPoints = 1000000;
    int _inCirclePts = 0;
    
    #pragma omp parallel for num_threads( NUM_THREADS ) reduction(+:_inCirclePts)
    for ( int q = 0; q < _nPoints; q++ )
    {
        // Random points inside a square of side 1
        // Area is 1*1 = 1
        // Area of the circle is Pi * 0.25
        // Pi = 4 * Acircle / Asquare ~ 4 * inCircleCount / inSquareCount
        double _x = RANDOM();
        double _y = RANDOM();
        double _d = sqrt( ( _x - 0.5 ) * ( _x - 0.5 ) +
                          ( _y - 0.5 ) * ( _y - 0.5 ) );
        if ( _d < 0.5 )
        {
            _inCirclePts += 1;
        }
    }

    double _pi = 4.0 * ( ( double ) _inCirclePts ) / _nPoints;

    cout << "serial montecarlo pi calculation: " << _pi << endl;


}
