
#include <iostream>
#include <string>

using namespace std;

#define BUFF_SIZE 16

__global__ void helloWorldKernel( char* d_data )
{
    d_data[0] = 'H';
    d_data[1] = 'e';
    d_data[2] = 'l';
    d_data[3] = 'l';
    d_data[4] = 'o';
    d_data[5] = ' ';
    d_data[6] = 'W';
    d_data[7] = 'o';
    d_data[8] = 'r';
    d_data[9] = 'l';
    d_data[10] = 'd';
    d_data[11] = '!';
    d_data[12] = '\n';
    d_data[13] = '\0';
}


int main()
{
	char _buff[BUFF_SIZE];

	char* d_data;
	cudaMalloc( ( void ** ) &d_data, 
                sizeof( char ) * BUFF_SIZE );

    helloWorldKernel<<<1,1>>>( d_data );

    cudaMemcpy( _buff, d_data, 
                sizeof( char ) * BUFF_SIZE, 
                cudaMemcpyDeviceToHost );

    cout << "out: " << _buff << endl;

    cudaFree( d_data );

	return 0;
}