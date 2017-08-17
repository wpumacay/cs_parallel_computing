



__kernel void numericalReduction( __global int* d_g_in_data, 
                                  __local int* d_l_data,
                                  __global int* d_g_out_data )
{

    size_t _globalId  = get_global_id( 0 );
    size_t _localSize = get_local_size( 0 );
    size_t _localId   = get_local_id( 0 );
    
    d_l_data[_localId] = d_g_in_data[_globalId];
    
    barrier( CLK_LOCAL_MEM_FENCE );

    for ( int i = _localSize >> 1; i > 0; i >>= 1 )
    {
        if ( _localId < i )
        {
            d_l_data[_localId] += d_l_data[_localId + i];
        }

        barrier( CLK_LOCAL_MEM_FENCE );
    }

    if ( _localId == 0 )
    {
        d_g_out_data[get_group_id(0)] = d_l_data[0];
    }

}
