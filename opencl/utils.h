
#pragma once


#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <CL/cl.h>


using namespace std;


namespace utils
{
    template<class T>
    void printvector( const vector<T> &vec )
    {
        for ( int q = 0; q < vec.size(); q++ )
        {
            cout << vec[q] << " ";
        }
        cout << endl;
    }

    template<class T>
    void printbuff( T* buff, int size )
    {
        for ( int q = 0; q < size; q++ )
        {
            cout << buff[q] << " ";
        }
        cout << endl;
    }

    template<class T>
    void print2dMat( const T* mat2d, int sx, int sy )
    {
        for ( int x = 0; x < sx; x++ )
        {
            for ( int y = 0; y < sy; y++ )
            {
                cout << *( mat2d + x + y * sx ) << "\t";
            }
            cout << endl;
        }
    }


    string loadKernelSrc( const string &path )
    {
        string _src;
        
        ifstream _file( path.c_str() );
        
        string _line;

        while( getline( _file, _line ) )
        {
            _src += _line;
        }

        return _src;
    }

    
    cl::Program createProgram( const cl::Context &context,
                               const cl::Device &device,
                               const string& pathToKernelSrc )
    {
        string _strKernelSrc = loadKernelSrc( pathToKernelSrc );

        cl::Program::Sources _kernelSrc( 1, 
                                         make_pair( _strKernelSrc.c_str(),
                                                    _strKernelSrc.size() + 1 ) );
        
        cl::Program _program( context, _kernelSrc );
        cl_int _err = _program.build( "-cl-std=CL1.2" );

        if ( _err != CL_SUCCESS )
        {
            cout << "build info: ******" << endl;

            cout << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>( device ) << endl;

            cout << "******************" << endl;
        }

        return _program;
    }
                               





}




