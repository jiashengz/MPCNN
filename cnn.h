#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#define DEFAULT_CACHE_SIZE 16384

using namespace std;

static int num_images, 				// B
		   num_input_channels, 		// C
		   num_output_channels, 	// K
		   image_width, 			// W
		   image_height, 			// H
		   filter_width,			// R
		   filter_height,			// S
		   horizontal_stride,		// sigma_w
		   vertical_stride,			// sigma_h
		   cache_size,				// M

void find_optimal_parameters()
{

}

//
//  timer
//
double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

//
//  command line option processing
//
int find_option( int argc, char **argv, const char *option )
{
    for( int i = 1; i < argc; i++ )
        if( strcmp( argv[i], option ) == 0 )
            return i;
    return -1;
}

int read_int( int argc, char **argv, const char *option, int default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return atoi( argv[iplace+1] );
    return default_value;
}

char *read_string( int argc, char **argv, const char *option, char *default_value )
{
    int iplace = find_option( argc, argv, option );
    if( iplace >= 0 && iplace < argc-1 )
        return argv[iplace+1];
    return default_value;
}