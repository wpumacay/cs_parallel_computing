


#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

#define VECT_SIZE 10000000
#define NUM_THREADS 4

void merge( vector<int> &vect, int _left, int _mid, int _right );

void merge_sort( vector<int> &vect )
{
    int _curr_size;
    int _left_start;
    
    for ( _curr_size = 1; _curr_size <= vect.size() - 1; _curr_size = 2 * _curr_size )
    {
        for ( _left_start = 0; _left_start < vect.size() - 1; _left_start += 2 * _curr_size )
        {
            int _mid = _left_start + _curr_size - 1;
            
            int _right_end = min( _left_start + 2 * _curr_size - 1, (int)( vect.size() - 1 ) );
            merge( vect, _left_start, _mid, _right_end );
        }
    }
}

void merge_sort_parallel( vector<int> &vect )
{
    int _curr_size;
    int len_vect = vect.size();
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel shared(vect) private( _curr_size )
    for ( _curr_size = 1; _curr_size <= len_vect - 1; _curr_size = 2 * _curr_size )
    {
        #pragma omp for 
        for ( int _left_start = 0; _left_start < len_vect - 1; _left_start += 2 * _curr_size )
        {
            int _mid = _left_start + _curr_size - 1;
            
            int _right_end = min( _left_start + 2 * _curr_size - 1, len_vect - 1 );

            //if ( omp_get_thread_num() == 0 )
            //{
            //	std::cout << "foo" << std::endl;
            //}

            merge( vect, _left_start, _mid, _right_end );
        }
    }
}



void merge( vector<int> &vect, int _left, int _mid, int _right )
{
    int q;

    int _n1 = _mid - _left + 1;// size of left array
    int _n2 = _right - _mid;// size of right array

    vector<int> _L, _R;

    for ( q = 0; q < _n1; q++ )
    {
        _L.push_back( vect[_left + q] );
    }
    for ( q = 0; q < _n2; q++ )
    {
        _R.push_back( vect[_mid + q + 1] );
    }

    int p1 = 0;
    int p2 = 0;
    q = _left;

    while( p1 < _n1 && p2 < _n2 )
    {
        if ( _L[p1] <= _R[p2] )
        {
            vect[q] = _L[p1];
            p1++;
        }
        else
        {
            vect[q] = _R[p2];
            p2++;
        }
        q++;
    }

    while ( p1 < _n1 )
    {
        vect[q] = _L[p1];
        p1++;
        q++;
    }

    while ( p2 < _n2 )
    {
        vect[q] = _R[p2];
        p2++;
        q++;
    }
}

void print_vector( vector<int> &vect )
{
    // return;
    cout << "[ ";
    for ( int q = 0; q < vect.size(); q++ )
    {
        cout << vect[q] << " ";
    }
    cout << "]" << endl;
}


int main()
{
    cout << "testing serial merge_sort ******" << endl;
    vector<int> _arr;
    for ( int q = 1; q <= VECT_SIZE; q++ )
    {
        _arr.push_back( q );
    }
    random_shuffle( _arr.begin(), _arr.end() );

    //print_vector( _arr );
    double _t1 = omp_get_wtime();
    merge_sort( _arr );
    double _t2 = omp_get_wtime();
    //print_vector( _arr );

    double serial_time = _t2 - _t1;

    cout << "time: " << _t2 - _t1 << endl;
    cout << "********************************" << endl;
    cout << "testing parallel merge_sort ****" << endl;

    random_shuffle( _arr.begin(), _arr.end() );

    //print_vector( _arr );
    _t1 = omp_get_wtime();
    merge_sort_parallel( _arr );
    _t2 = omp_get_wtime();
    //print_vector( _arr );
   
    double parallel_time = _t2 - _t1;

    cout << "time: " << _t2 - _t1 << endl;

    cout << "speedup: " << serial_time / parallel_time << endl;
    cout << "efficiency: " << ( serial_time / parallel_time ) / ( NUM_THREADS ) << endl;
    cout << "********************************" << endl;


    return 0;
}
