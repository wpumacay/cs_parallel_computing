
#include <iostream>
#include <vector>
#include <string>
#include <CL/cl.hpp>
#include "utils.h"

using namespace std;

struct CLres
{
    cl::Context* context;
    cl::Platform* platform;
    cl::Device* device;
};

#define VECTOR_SIZE 1000

void printvector( const vector<int> &vec )
{
    for ( int q = 0; q < vec.size(); q++ )
    {
        cout << vec[q] << " ";
    }
    cout << endl;
}

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
    
    CLres _cl_res;
    _cl_res.context = &_context;
    _cl_res.platform = &_platform;
    _cl_res.device = &_device;

    // ****************************************************

    cl::Program _program = utils::createProgram( _context, 
                                                 string( "hello_array_kernel.cl" ) );

    
    vector<int> _vec( VECTOR_SIZE );

    for ( int q = 0; q < VECTOR_SIZE; q++ )
    {
        _vec[q] = ( q + 1 );
    } 
    
    cl::Buffer d_inBuff( _context, 
                         CL_MEM_READ_ONLY | 
                         CL_MEM_HOST_NO_ACCESS |
                         CL_MEM_COPY_HOST_PTR,
                         sizeof( int ) * _vec.size(),
                         _vec.data() );

    cl::Buffer d_outBuff( _context,
                          CL_MEM_WRITE_ONLY |
                          CL_MEM_HOST_READ_ONLY,
                          sizeof( int ) * _vec.size() );

    cl::Kernel _kernel( _program, "processArray" );
    auto err = _kernel.setArg( 0, d_inBuff );
    cout << "err: " << err << endl;
    err = _kernel.setArg( 1, d_outBuff );
    cout << "err: " << err << endl;

    cl::CommandQueue _queue( _context, _device );
    _queue.enqueueNDRangeKernel( _kernel, 
                                 cl::NullRange,
                                 cl::NDRange( _vec.size() ) ); 

    _queue.enqueueReadBuffer( d_outBuff, CL_FALSE,
                              0, sizeof( int ) * _vec.size(),
                              _vec.data() );

    _queue.finish();

    printvector( _vec );

    return 0;
}
