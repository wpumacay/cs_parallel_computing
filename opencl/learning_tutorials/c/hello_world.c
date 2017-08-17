


#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define MEM_SIZE 100


int main()
{

	// Initialize the platform and device **********

	cl_device_id _device_id 	= NULL;
	cl_context _context     	= NULL;
	cl_command_queue _cmd_queue = NULL;
	cl_mem d_mem_buffer 		= NULL;
	cl_program _program 		= NULL;
	cl_kernel _kernel 			= NULL;
	cl_platform_id _platform_id = NULL;
	cl_uint _num_devices		= 0;
	cl_uint _num_platforms		= 0;
	cl_int _ret					= 0;

	_ret = clGetPlatformIDs( 1, &_platform_id, &_num_platforms );
	printf( "NumPlatforms: %d\n", _num_platforms );

	_ret = clGetDeviceIDs( _platform_id, CL_DEVICE_TYPE_GPU, 
						   1, &_device_id, &_num_devices );
	printf( "NumDevices: %d\n", _num_devices );

	// Create a CL context
	_context   = clCreateContext( NULL, 1, &_device_id, NULL, NULL, &_ret );
	printf( "CreateContext-errorCode: %d\n", _ret );
	_cmd_queue = clCreateCommandQueue( _context, _device_id, 0, &_ret );
	printf( "CreateCommandQueue-errorCode: %d\n", _ret );

	// *********************************************

	// Create the memory buffer in the host
	char h_str[MEM_SIZE];

	// Create the memory buffer to store the memory ( like cudaMalloc )
	d_mem_buffer = clCreateBuffer( _context, 
								   CL_MEM_READ_WRITE, 
								   sizeof( char ) * MEM_SIZE, 
								   NULL, 
								   &_ret );
	printf( "CreateBuffInDevice-errorCode: %d\n", _ret );
	// Create a kernel program from the related cl file
	USource _res_src = utils_readFile( "../cl/hello_world_kernel.cl" );

	char* _src_str  = _res_src.src_str;
	size_t _src_size = _res_src.src_size;

	printf( "Kernel to be used: \n" );
	printf( "%s\n", _src_str );

	printf( "Building program from src ... \n" );

	_program = clCreateProgramWithSource( _context, 1, 
								     	  ( const char ** ) &_src_str, 
								     	  ( const size_t * ) &_src_size, 
								     	  &_ret );
	printf( "CreatingProgram-errorCode: %d\n", _ret );
	_ret = clBuildProgram( _program, 1, &_device_id, 
						   NULL, NULL, NULL );

	printf( "Building-errorCode: %d\n", _ret );
	printf( "Finished building \n" );

	// Create the kernel
	_kernel = clCreateKernel( _program, "HelloWorld", &_ret );
	printf( "KernelCreate-errorCode: %d\n", _ret );

	// Set kernel args
	_ret = clSetKernelArg( _kernel, 0, 
						   sizeof( cl_mem ), ( void * )&d_mem_buffer );

	clEnqueueTask( _cmd_queue, _kernel, 0, NULL, NULL );

	clEnqueueReadBuffer( _cmd_queue, d_mem_buffer, 
						 CL_TRUE, 0, sizeof( char ) * MEM_SIZE, 
						 h_str, 0, NULL, NULL );

	printf( "result: %s\n", h_str );

	// Free the resources
	_ret = clFlush( _cmd_queue );
	_ret = clFinish( _cmd_queue );
	_ret = clReleaseKernel( _kernel );
	_ret = clReleaseProgram( _program );
	_ret = clReleaseMemObject( d_mem_buffer );
	_ret = clReleaseCommandQueue( _cmd_queue );
	_ret = clReleaseContext( _context );

	free( _src_str );

	return 0;
}