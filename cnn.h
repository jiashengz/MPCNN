#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
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

void set_random_array(int *array, int length)
{
    for (int i = 0; i < length; i++)
    {
        *(array + length) = rand() % 256;
    }
}

void naive_cnn(int *images, int *filters, int *output, global_config_t t)
{
    for (int b = 0; b < t.B; b++)
    {
        for (int c = 0; c < t.C; c++)
        {
            for (int k = 0; k < t.K; k++)
            { 
                for(int w = 0; w < t.W; w++)
                {
                    for (int h = 0; h < t.H; h++)
                    {
                        for (int r = 0; r < t.R; r++)
                        {
                            for (int s = 0; s < t.S; s++)
                            {
                                int offset_output = k * t.H * t.W * t.B + h * t.W * t.B + w * t.B + b;
                                int offset_input = (r + t.sigW * w) * (t.sigH * (t.H - 1) + t.S) * t.C * t.B + (s + t.sigH * h) * t.C * t.B + c * t.B + b;
                                int offset_filter = k * t.R * t.S * t.C + r * t.S * t.C + s * t.C + c;
                                output[offset_output] += images[offset_input] * filters[offset_filter];
                                //cout << offset_output << " ";
                            }
                        }
                    }
                }
            }
        }
    }
}
