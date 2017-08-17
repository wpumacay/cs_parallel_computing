
__kernel void processArray( __global int* d_in_data,
                            __global int* d_out_data )
{
    int _indx = get_global_id( 0 );

    d_out_data[_indx] = d_in_data[_indx] * 2;
}
