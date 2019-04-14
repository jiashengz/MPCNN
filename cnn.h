#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#define DEFAULT_CACHE_SIZE 16384

using namespace std;

//
// global data structure
//
typedef struct 
{
    int B;
    int W;
    int H;
    int K;
    int C;
    int R;
    int S;
    int sigH;
    int sigW;
} global_config_t;

//
// block data structure
//
typedef struct 
{
    int block_B;
    int block_W;
    int block_H;
    int block_K;
    int block_C;
    int block_Rp;
    int block_Sp;
    int block_Rpp;
    int block_Spp;
} block_config_t;

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

