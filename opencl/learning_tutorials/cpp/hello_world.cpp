

#include <CL/cl.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <utility>

using namespace std;


int main()
{
    vector<cl::Platform> v_platforms;
    cl::Platform::get( &v_platforms );

    cl::Platform _platform = v_platforms.front();
    //cl::Platform _platform = v_platforms.back();

    cout << "#platforms available: " << v_platforms.size() << endl;

    vector<cl::Device> v_devices;
    _platform.getDevices( CL_DEVICE_TYPE_ALL, &v_devices );

    cout << "#devices available: " << v_devices.size() << endl;

    cl::Device _device = v_devices.front();
    
    string _devInfo;

    _device.getInfo( CL_DEVICE_NAME, &_devInfo );
    
    cout << "devInfo-name: " << _devInfo << endl;

    ifstream _hwFile( "../cl/hello_world_kernel.cl" );

    string _srcKernel;
    string _tmp;

    while ( std::getline( _hwFile, _tmp ) )
    {
        _srcKernel += _tmp + "\n";
    }

    cl::Program::Sources _sources( 1, 
                                   std::make_pair( _srcKernel.c_str(),
                                                   _srcKernel.length() + 1 ) );

    
    cl::Context _context( _device );
    cl::Program _program( _context, _sources );

    auto _err = _program.build( "-cl-std=CL1.2" );

    char _buff[16];

    cl::Buffer d_memBuf( _context,
                         CL_MEM_WRITE_ONLY |
                         CL_MEM_HOST_READ_ONLY,
                         sizeof( _buff ) );

    cl::Kernel _kernel( _program, "HelloWorld", &_err );
    _kernel.setArg( 0, d_memBuf );


    cl::CommandQueue _queue( _context, _device );
    _queue.enqueueTask( _kernel );
    _queue.enqueueReadBuffer( d_memBuf, CL_TRUE, 0, sizeof( _buff ), _buff );

    cout << _buff;

    return 0;
}









