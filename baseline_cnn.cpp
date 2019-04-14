#include "cnn.h"

using namespace std;

class baseline_cnn
{
	global_config_t global_config;
	block_config_t block_config;
public:
	baseline_cnn(global_config_t g, block_config_t b){
		global_config = g;
		block_config = b;
	};
	~baseline_cnn();
	
    void block_conv(const float * images, const float * filters, float * result, 
        int B, int C, int K, int W, int H, int RP, int RPP, int SP, int SPP){
        int b = 0, c = 0, k = 0, w = 0, h = 0, rp = 0, rpp = 0, sp = 0, spp = 0;
        for (;b < block_config.block_B; ++b)
        {
            for (;c < block_config.block_C; ++c)
            {
                for (; k < block_config.block_K; ++k)
                {
                    for (; w < block_config.block_W; ++w)
                    {
                        for (; h < block_config.block_H; ++h)
                        {
                            for (; rp < block_config.block_Rp; ++rp)
                            {
                                for (; rpp < block_config.block_Rpp; ++rpp)
                                {
                                    for (; sp < block_config.block_Sp; ++sp)
                                    {
                                        for (; spp < block_config.block_Spp; ++spp)
                                        {
                                            result[(k+K)*global_config.H*global_config.W*global_config.B+(h+H)*global_config.W*global_config.B+(w+W)*global_config.B+b+B] 
                                                += images[(b+B)*global_config.H*global_config.W*global_config.C+(c+C)*global_config.H*global_config.W+((h+H+sp+SP)*global_config.sigH+SPP+spp)*global_config.W+((w+W+rp+RP)*global_config.sigW)+RPP+rpp]
                                                * filters[(k+K)*global_config.R*global_config.S*global_config.C+(c+C)*global_config.R*global_config.S+((sp+SP)*global_config.sigH+spp+SPP)*global_config.R+((rp+RP)*global_config.sigW+rpp+RPP)];
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

	void conv(const float * images, const float * filters, float * result){
		int b = 0, c = 0, k = 0, w = 0, h = 0, rp = 0, rpp = 0, sp = 0, spp = 0;
        for (;b < global_config.B - block_config.block_B; b += block_config.block_B)
        {
            for (;c < global_config.C - block_config.block_C; c += block_config.block_C)
            {
                for (; k < global_config.K - block_config.block_K; k += block_config.block_K)
                {
                    for (; w < global_config.W - block_config.block_W; w += block_config.block_W)
                    {
                        for (; h < global_config.H - block_config.block_H; h += block_config.block_H)
                        {
                            for (; rp < global_config.R / global_config.sigW - block_config.block_Rp; rp += block_config.block_Rp)
                            {
                                for (; rpp < global_config.sigW - block_config.block_Rpp; rpp += block_config.block_Rpp)
                                {
                                    for (; sp < global_config.S / global_config.sigH - block_config.block_Sp; sp += block_config.block_Sp)
                                    {
                                        for (; spp < global_config.sigH - block_config.block_Spp; spp += block_config.block_Spp)
                                        {
                                            block_conv(images, filters, result, b, c, k, w, h, rp, rpp, sp, spp);
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
};