#include "cnn.h"

using namespace std;

class baseline_cnn
{
	global_config_t global_config;
	block_config_t block_config;
public:
	baseline_cnn(global_config_t g, block_config_t b){};
	~baseline_cnn();
	
	void conv(float * images, float * filters){

	}
};