

#include <iostream>
#include <vector>

using namespace std;

#include <CL/cl.hpp>


#include "utils.h"

#define VEC_SIZE 2048

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

    cl::Program _program = utils::createProgram( _context, 
                                                 _device, 
                                                 string( "../cl/kernel_numerical_reduction.cl" ) );

    cl::Kernel _kernel( _program, "numericalReduction" );


    cl_int _err = 0;
    auto _workGroupSize = _kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>( _device, &_err );
    auto _numWorkGroups = VEC_SIZE / _workGroupSize;

    cout << "workgroupsize: " << _workGroupSize << endl;
    cout << "numworkgroups: " << _numWorkGroups << endl;

    int h_vec_in[VEC_SIZE];
    int* h_vec_out = new int[_numWorkGroups];

    for ( int q = 0; q < VEC_SIZE; q++ )
    {
        h_vec_in[q] = ( q + 1 );
    }   
    cl::Buffer d_in_buff( _context,
                          CL_MEM_READ_ONLY |
                          CL_MEM_HOST_NO_ACCESS |
                          CL_MEM_COPY_HOST_PTR,
                          sizeof( int ) * VEC_SIZE,
                          h_vec_in );
    cl::Buffer d_out_buff( _context,
                           CL_MEM_WRITE_ONLY |
                           CL_MEM_HOST_READ_ONLY,
                           sizeof( int ) * _numWorkGroups );
    

    _kernel.setArg( 0, d_in_buff );
    _kernel.setArg( 1, sizeof( int ) * _workGroupSize, nullptr );
    _kernel.setArg( 2, d_out_buff );


    cl::CommandQueue _queue( _context, _device );
    _queue.enqueueNDRangeKernel( _kernel,
                                 cl::NullRange,
                                 cl::NDRange( VEC_SIZE ),
                                 cl::NDRange( _workGroupSize ) );

    _queue.enqueueReadBuffer( d_out_buff, CL_TRUE, 
                              0, sizeof( int ) * _numWorkGroups,
                              h_vec_out );


    utils::printbuff( h_vec_out, _numWorkGroups );

    delete[] h_vec_out;

    return 0;
}
