
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_SOURCE_SIZE 0x100000 // max 128 kB

typedef struct _USource
{
	char* src_str;
	size_t src_size;
} USource;


USource utils_readFile( const char* path )
{
	FILE* _fp;

	USource _res;

	_fp = fopen( path, "r" );
	if ( !_fp )
	{
		fprintf( stderr, "utils::readFile> Failed to load the kernel\n" );
		_res.src_str  = NULL;
		_res.src_size = 0;

		return _res;
	}

	char* _source_str   = ( char* ) malloc( sizeof( char ) * MAX_SOURCE_SIZE );
	size_t _source_size = fread( _source_str, 1, MAX_SOURCE_SIZE, _fp );

	fclose( _fp );

	_res.src_str  = _source_str;
	_res.src_size = _source_size;

	return _res;
}





#endif

