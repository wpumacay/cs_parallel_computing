


#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

#define VEC_SIZE 2048

int main()
{

	// Initialize the platform and device **********

	cl_context _context     	= NULL;
	cl_command_queue _cmd_queue = NULL;

	cl_platform_id _platform_id = NULL;
	cl_device_id _device_id 	= NULL;
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

	// Create the program 
	cl_program _program = NULL;

	USource _res_src = utils_readFile( "../cl/kernel_numerical_reduction.cl" );
	char* _src_str   = _res_src.src_str;
	size_t _src_size = _res_src.src_size;

	printf( "Building program from src ... \n" );
	_program = clCreateProgramWithSource( _context, 1, 
										  ( const char ** )&_src_str, 
										  ( const size_t* )&_src_size, 
										  &_ret );
	printf( "CreatingProgram-errorCode: %d\n", _ret );
	_ret = clBuildProgram( _program, 1, &_device_id, 
						   NULL, NULL, NULL );
	printf( "Building-errorCode: %d\n", _ret );
	printf( "Finished building \n" );

	// Create the kernel
	cl_kernel _kernel = NULL;
	_kernel = clCreateKernel( _program, 
							  "numericalReduction", 
							  &_ret );
	printf( "KernelCreate-errorCode: %d\n", _ret );

	// Divide the work into workgroups
	size_t _workGroupSize;
	_ret = clGetKernelWorkGroupInfo( _kernel, _device_id, 
							  		 CL_KERNEL_WORK_GROUP_SIZE,
							  		 sizeof( size_t ),
							  		 &_workGroupSize, NULL );

	size_t _numWorkGroups = VEC_SIZE / _workGroupSize;

	printf( "workGroupSize: %d \n", _workGroupSize );
	printf( "nWorkdGroups: %d \n", _numWorkGroups );

	// Create buffers

	int h_in_buff[VEC_SIZE];
	int* h_out_buff = ( int* ) malloc( sizeof( int ) * _numWorkGroups );

	int _res = 0;

	for ( int q = 0; q < VEC_SIZE; q++ )
	{
		h_in_buff[q] = ( q + 1 );
		_res += ( q + 1 );
	}


	cl_mem d_in_buff = clCreateBuffer( _context, 
									   CL_MEM_READ_ONLY |
									   CL_MEM_HOST_NO_ACCESS |
									   CL_MEM_COPY_HOST_PTR, 
									   sizeof( int ) * VEC_SIZE, 
									   h_in_buff, &_ret );

	cl_mem d_out_buff = clCreateBuffer( _context,
		 						    	CL_MEM_WRITE_ONLY |
		 						    	CL_MEM_HOST_READ_ONLY,
		 						    	sizeof( int ) * _numWorkGroups,
		 						    	NULL, &_ret );

	// Setup kernelparams for launch
	_ret = clSetKernelArg( _kernel, 0, 
						   sizeof( cl_mem ), 
						   ( void * ) &d_in_buff );
	_ret = clSetKernelArg( _kernel, 1, 
	     				   sizeof( int ) * _workGroupSize, 
	     				   NULL );
	_ret = clSetKernelArg( _kernel, 2, 
						   sizeof( cl_mem ), 
						   ( void * ) &d_out_buff );

	size_t _vec_size = VEC_SIZE;
	_ret = clEnqueueNDRangeKernel( _cmd_queue, _kernel,
								   1, NULL,
								   &_vec_size,
								   &_workGroupSize, 0, NULL, NULL );


	clEnqueueReadBuffer( _cmd_queue, 
						 d_out_buff, 
						 CL_TRUE, 
						 0, 
						 sizeof( int ) * _numWorkGroups, 
						 h_out_buff, 
						 0, NULL, NULL );

	int _res2 = 0;
	for ( int q = 0; q < _numWorkGroups; q++ )
	{
		printf( "h_out_buff[%d] = %d \n", q, h_out_buff[q] );
		_res2 += h_out_buff[q];
	}

	printf( "res  : %d\n", _res );
	printf( "res2 : %d\n", _res2 );


	// Free the resources
	_ret = clFlush( _cmd_queue );
	_ret = clFinish( _cmd_queue );
	_ret = clReleaseKernel( _kernel );
	_ret = clReleaseProgram( _program );

	_ret = clReleaseMemObject( d_in_buff );
	_ret = clReleaseMemObject( d_out_buff );

	_ret = clReleaseCommandQueue( _cmd_queue );
	_ret = clReleaseContext( _context );

	free( h_out_buff );

	return 0;
}