

__kernel void processMultiArray( __global int* d_data )
{
    size_t id = get_global_id( 1 ) * 
                get_global_size( 0 ) + 
                get_global_id( 0 );

    d_data[id] = d_data[id] * 3;
}
