

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <CL/cl.hpp>

using namespace std;

#include "utils.h"





int main()
{
    // ****************************************************
    vector<cl::Platform> v_platforms;
    cl::Platform::get( &v_platforms );

    // Choose the platform, just the first one for now
    cl::Platform _platform = v_platforms.front();

    vector<cl::Device> v_devices;
    _platform.getDevices( CL_DEVICE_TYPE_ALL, &v_devices );
    
    // Choose the device
    cl::Device _device = v_devices.front();

    cl::Context _context( _device );
    // ****************************************************

    cl::Program _program = utils::createProgram( _context, string( "hello_multi_array_kernel.cl" ) );
    
    const int NUM_ROWS = 3;
    const int NUM_COLS = 2;
    const int COUNT_SIZE = NUM_ROWS * NUM_COLS;

    int* _arr = new int[COUNT_SIZE];

    _arr[0 * NUM_COLS + 0] = 1;
    _arr[0 * NUM_COLS + 1] = 1;
    _arr[1 * NUM_COLS + 0] = 2;
    _arr[1 * NUM_COLS + 1] = 2;
    _arr[2 * NUM_COLS + 0] = 3;
    _arr[2 * NUM_COLS + 1] = 3;

    cl::Buffer d_in_buff( _context,  
                          CL_MEM_READ_WRITE |
                          CL_MEM_HOST_READ_ONLY |
                          CL_MEM_COPY_HOST_PTR,
                          sizeof( int ) * COUNT_SIZE,
                          _arr );

    cl::Kernel _kernel( _program, "processMultiArray" );
    _kernel.setArg( 0, d_in_buff );

    cl::CommandQueue _queue( _context, _device );
    _queue.enqueueNDRangeKernel( _kernel,
                                 cl::NullRange,
                                 cl::NDRange( 2, 3 ) );
    _queue.enqueueReadBuffer( d_in_buff, CL_TRUE,
                              0, sizeof( int ) * COUNT_SIZE,
                              _arr );

    utils::print2dMat<int>( _arr, NUM_COLS, NUM_ROWS );
    

    delete[] _arr;

    return 0;
}
