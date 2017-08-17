#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

#VEC_SIZE = 2048
VEC_SIZE = 1024

_available_platforms = cl.get_platforms()
_platform = _available_platforms[0]
print 'using platform: ', _platform

_devices_for_platform = _platform.get_devices( cl.device_type.GPU )
_device = _devices_for_platform[0]
print 'using device: ', _device

_ctx = cl.Context( [_device] )
_cmd_queue = cl.CommandQueue( _ctx )


_src_str = ""

with open( '../cl/kernel_numerical_reduction.cl' ) as _file :
	_src_str = _file.read()

_program = cl.Program( _ctx, _src_str )
_program.build()

_kernel = cl.Kernel( _program, "numericalReduction" )

_workGroupSize = _kernel.get_work_group_info( cl.kernel_work_group_info. WORK_GROUP_SIZE,
										 	  _device )
_numWorkGroups = VEC_SIZE / _workGroupSize
print 'workGroupSize: ', _workGroupSize
print 'numWorkGroups: ', _numWorkGroups

h_in_buff = np.zeros( VEC_SIZE ).astype( np.int32 )
h_out_buff = np.zeros( _numWorkGroups ).astype( np.int32 )
for q in range( len( h_in_buff ) ) :
	h_in_buff[q] = ( q + 1 )

d_in_buff = cl.Buffer( _ctx,
					   cl.mem_flags.READ_ONLY |
					   cl.mem_flags.HOST_NO_ACCESS |
					   cl.mem_flags.COPY_HOST_PTR,
					   hostbuf = h_in_buff )

d_out_buff = cl.Buffer( _ctx,
						cl.mem_flags.WRITE_ONLY |
						cl.mem_flags.HOST_READ_ONLY,
						size = h_out_buff.nbytes )

_h_local_buff = np.zeros( _workGroupSize ).astype( np.int32 )
d_local_buff = cl.LocalMemory( _h_local_buff.nbytes )

_kernel.set_arg( 0, d_in_buff )
_kernel.set_arg( 1, d_local_buff )
_kernel.set_arg( 2, d_out_buff )

cl.enqueue_nd_range_kernel( _cmd_queue, 
							_kernel,
							( VEC_SIZE, 1, 1 ),
							( _workGroupSize, 1, 1 ) )

cl.enqueue_copy( _cmd_queue, h_out_buff, d_out_buff )


print 'res: ', h_out_buff