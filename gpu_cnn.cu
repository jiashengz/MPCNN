#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include "cnn.h"

#define NUM_THREADS 256
#define RUN_COMPARE false

using namespace std;

__device__ void block_conv(global_config_t * global_config, block_config_t* block_config,
	int * images, int * filters, int * result, 
	int B, int C, int K, int W, int H, int RP, int RPP, int SP, int SPP){

	int b = 0, c = 0, k = 0, w = 0, h = 0, rp = 0, rpp = 0, sp = 0, spp = 0;

	int block_B = min (global_config->B - B, block_config->block_B);
	int block_C = min (global_config->C - C, block_config->block_C);
	int block_K = min (global_config->K - K, block_config->block_K);
	int block_W = min (global_config->W - W, block_config->block_W);
	int block_H = min (global_config->H - H, block_config->block_H);
	int block_Rp = min (global_config->R/global_config->sigW-RP, block_config->block_Rp);
	int block_Rpp = min (global_config->sigW - RPP, block_config->block_Rpp);
	int block_Sp = min (global_config->S/global_config->sigH - SP, block_config->block_Sp);
	int block_Spp = min (global_config->sigH - SPP, block_config->block_Spp);

	for (b = 0;b < block_B; ++b)
	{
		for (c = 0;c < block_C; ++c)
		{
			for (k = 0; k < block_K; ++k)
			{
				for (w = 0; w < block_W; ++w)
				{
					for (h = 0; h < block_H; ++h)
					{
						for (rp = 0 ; rp < block_Rp; ++rp)
						{
							for (rpp = 0 ; rpp < block_Rpp; ++rpp)
							{
								for (sp = 0; sp < block_Sp; ++sp)
								{
									for (spp = 0; spp < block_Spp; ++spp)
									{
                                        //cerr << (k+K)*global_config->H*global_config->W*global_config->B+(h+H)*global_config->W*global_config->B+(w+W)*global_config->B+b+B << " ";
										atomicAdd(result+(k+K)*global_config->H*global_config->W*global_config->B+(h+H)*global_config->W*global_config->B+(w+W)*global_config->B+b+B,
										images[(rpp+RPP+global_config->sigW*(rp+RP+w+W))*((global_config->H-1)*global_config->sigH+global_config->S)*global_config->C*global_config->B
											+ (SPP+spp+global_config->sigH*(sp+SP+h+H))*global_config->C*global_config->B+(c+C)*global_config->B+b+B]
										* filters[(k+K)*global_config->S*global_config->R*global_config->C+(global_config->sigW*(rp+RP)+rpp+RPP)*global_config->S*global_config->C+(global_config->sigH*(sp+SP)+spp+SPP)*global_config->C+c+C]);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}


__global__ void gpu_conv(global_config_t * gpu_config_global, block_config_t* gpu_config_block,
	int * gpu_input, int * gpu_filter, int * gpu_output,
	int input_size, int filter_size, int output_size,
	int nb, int nk, int nw, int nh){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= 256) return;
	int b_steps = gpu_config_global->B / gpu_config_block->block_B / nb;
	int b_index = tid / (256 / nb);
	int b_start = gpu_config_block->block_B * (b_index) * b_steps;

	int k_index = tid / (256 / (nb * nk)) % (nk);
	int k_steps = gpu_config_global->K / gpu_config_block->block_K / nk;
	int k_start = gpu_config_block->block_K * k_index * k_steps;

	int w_index = tid / (256 / (nb * nk * nw)) % nw;
	int w_steps = gpu_config_global->W / gpu_config_block->block_W / nw;
	int w_start = gpu_config_block->block_W * w_index * w_steps;

	int h_index = tid / (256 / (nb * nk * nw * nh)) % nh;
	int h_steps = gpu_config_global->H / gpu_config_block->block_H / nh;
	int h_start = gpu_config_block->block_H * h_index * h_steps;

	int b_step, k_step, w_step, h_step;
	int b = 0, c = 0, k = 0, w = 0, h = 0, rp = 0, rpp = 0, sp = 0, spp = 0;
	for (b = b_start, b_step = b_steps;b < gpu_config_global->B && b_step > 0; b += gpu_config_block->block_B, b_step--)
	{
		for (c = 0;c < gpu_config_global->C; c += gpu_config_block->block_C)
		{
			for (k = k_start, k_step = k_steps; k < gpu_config_global->K && k_step > 0; k += gpu_config_block->block_K, k_step--)
			{
				for (w = w_start, w_step = w_steps; w < gpu_config_global->W && w_step > 0; w += gpu_config_block->block_W, w_step--)
				{
					for (h = h_start, h_step = h_steps; h < gpu_config_global->H && h_step > 0; h += gpu_config_block->block_H, h_step--)
					{
						for (rp = 0; rp < gpu_config_global->R / gpu_config_global->sigW; rp += gpu_config_block->block_Rp)
						{
							for (rpp = 0 ; rpp < gpu_config_global->sigW; rpp += gpu_config_block->block_Rpp)
							{
								for (sp =0 ; sp < gpu_config_global->S / gpu_config_global->sigH; sp += gpu_config_block->block_Sp)
								{
									for (spp = 0; spp < gpu_config_global->sigH; spp += gpu_config_block->block_Spp)
									{
										block_conv(gpu_config_global, gpu_config_block, gpu_input, gpu_filter, gpu_output, b, c, k, w, h, rp, rpp, sp, spp);
									}
								}
							}
						}
					}
				}
			}
		}
	}


}

void compute_conv(global_config_t * global_config, block_config_t* block_config,
	int * images, int * filters, int * result,
	int input_size, int filter_size, int output_size,
	int nb, int nk, int nw, int nh){

	cudaDeviceSynchronize();
    int * gpu_input, * gpu_filter, *gpu_output;
    global_config_t * gpu_config_global;
    block_config_t * gpu_config_block;
    if(cudaSuccess != cudaMalloc((void **) &gpu_config_global, sizeof(global_config_t)))
    	cerr << "gpu global config allocation failed" << endl;
    if(cudaSuccess != cudaMalloc((void **) &gpu_config_block, sizeof(block_config_t)))
    	cerr << "gpu block config allocation failed" << endl;
    if(cudaSuccess != cudaMalloc((void **) &gpu_input, input_size * sizeof(int)))
    	cerr << "gpu input allocation failed" << endl;
    if(cudaSuccess != cudaMalloc((void **) &gpu_filter, filter_size * sizeof(int)))
    	cerr << "gpu filter allocation failed" << endl;
    if(cudaSuccess != cudaMalloc((void **) &gpu_output, output_size * sizeof(int)))
    	cerr << "gpu output allocation failed" << endl;

    cudaMemcpy(gpu_input, images, input_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_filter, filters, filter_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(gpu_output, 0, output_size * sizeof(int));
    cudaMemcpy(gpu_config_global, global_config, sizeof(global_config_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_config_block, block_config, sizeof(block_config_t), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	double simulation_time = read_timer();

	gpu_conv<<<1,NUM_THREADS>>>(gpu_config_global, gpu_config_block, gpu_input, gpu_filter, gpu_output, input_size, filter_size, output_size, nb, nk, nw, nh);
	
	cudaDeviceSynchronize();
	cout << "GPU CNN: " << read_timer() - simulation_time << endl;

    cudaMemcpy(result, gpu_output, output_size * sizeof(int), cudaMemcpyDeviceToHost);

	
}

int main(int argc, char **argv)
{
	cudaDeviceSynchronize();
// Initialize input image
    srand(time(NULL));

    global_config_t test_global = {256, 512, 512, 6, 3, 6, 6, 2, 2};
    block_config_t test_block = {2, 64, 64, 3, 3, 3, 3, 1, 1};
    // block_config_t test_block = {100, 6, 6, 6, 3, 3, 3, 1, 1};
    // global_config_t test_global = {1, 4, 4, 1, 1, 1, 1, 1, 1};
    // block_config_t test_block = {1, 2, 2, 1, 1, 1, 1, 1, 1};  
    int input_w = test_global.sigW * (test_global.W - 1) + test_global.R;
    int input_h = test_global.sigH * (test_global.H - 1) + test_global.S;

    int input_size = input_h * input_w * test_global.C * test_global.B;
    int output_size = test_global.K * test_global.H * test_global.W * test_global.B;
    int filter_size = test_global.K * test_global.R * test_global.S * test_global.C;

    int * filter = new int[filter_size];

    for (int i = 0; i < filter_size; i++)
    {
        filter[i] = 1;
    }

    int * test_input = new int[input_size];

    for (int i = 0; i < input_size; i++)
    {
        test_input[i] = rand() % 256;
    }

    // int * output_naive = new int[output_size]();
    int * output_gpu = new int[output_size]();

    int nb = read_int( argc, argv, "-nb", 8 );
    int nk = read_int( argc, argv, "-nk", 2 );
    int nw = read_int( argc, argv, "-nw", 4 );
    int nh = read_int( argc, argv, "-nh", 4 );

	compute_conv(&test_global, &test_block, test_input, filter, output_gpu, input_size, filter_size, output_size, nb, nk, nw, nh);
	

	if (RUN_COMPARE){
		int * output_naive = new int[output_size]();

		double simulation_time = read_timer();
		naive_cnn(test_input, filter, output_naive, test_global);
		cout << "Naive CNN: " << read_timer() - simulation_time << endl;
		bool correct = true;
		for (int i = 0; i < output_size; i++)
	    {
	    	if(output_naive[i] != output_gpu[i]){
	    		correct = false;
	        	cout << output_naive[i] << "  VS.  " << output_gpu[i] << endl;
	        	break;
	    	}
	    }

	    if(correct) cout << "All results matches" << endl;

	    delete[] output_naive;
	}
		

    delete[] test_input; 
    delete[] output_gpu;
	return 0;
}
