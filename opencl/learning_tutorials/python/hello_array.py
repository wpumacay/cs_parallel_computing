#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

_ctx = cl.create_some_context()
_cmd_queue = cl.CommandQueue( _ctx )

VEC_SIZE = 1000

h_in_buff  = np.zeros( VEC_SIZE ).astype( np.int32 )
for q in range( len( h_in_buff ) ) :
	h_in_buff[q] = q + 1

h_out_buff = np.zeros( VEC_SIZE ).astype( np.int32 )

d_in_buff = cl.Buffer( _ctx, 
					   cl.mem_flags.READ_ONLY |
					   cl.mem_flags.HOST_NO_ACCESS |
					   cl.mem_flags.COPY_HOST_PTR,
					   hostbuf = h_in_buff )

d_out_buff = cl.Buffer( _ctx,
						cl.mem_flags.WRITE_ONLY |
						cl.mem_flags.HOST_READ_ONLY,
						hostbuf = h_out_buff )

_src_str = ""

with open( '../cl/hello_array_kernel.cl' ) as _file :
	_src_str = _file.read()

_program = cl.Program( _ctx, _src_str )
_program.build()

_program.processArray( _cmd_queue, h_in_buff.shape, None, d_in_buff, d_out_buff )

cl.enqueue_copy( _cmd_queue, h_out_buff, d_out_buff )

print 'input: '
print h_in_buff
print 'output: '
print h_out_buff