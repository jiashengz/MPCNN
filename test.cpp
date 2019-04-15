#include "baseline_cnn.h"

using namespace std;

int main(int argc, char **argv)
{
    // Initialize input image
    srand(time(NULL));

    global_config_t test_global = {100, 36, 36, 6, 3, 6, 6, 2, 2};
    block_config_t test_block = {100, 6, 6, 6, 3, 3, 3, 1, 1};
    // global_config_t test_global = {1, 4, 4, 1, 1, 1, 1, 1, 1};
    // block_config_t test_block = {1, 2, 2, 1, 1, 1, 1, 1, 1};  
    int input_w = test_global.sigW * (test_global.W - 1) + test_global.R;
    int input_h = test_global.sigH * (test_global.H - 1) + test_global.S;

    const int input_size = input_h * input_w * test_global.C * test_global.B;
    const int output_size = test_global.K * test_global.H * test_global.W * test_global.B;
    const int filter_size = test_global.K * test_global.R * test_global.S * test_global.C;

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

    int * output_naive = new int[output_size]();
    int * output_baseline = new int[output_size]();

    baseline_cnn base(test_global, test_block);

    base.conv(test_input, filter, output_baseline);
    cerr << endl;
    naive_cnn(test_input, filter, output_naive, test_global);
    cerr << endl;

    for (int i = 0; i < output_size; i++)
    {

        cout << output_naive[i] << "  VS.  " << output_baseline[i] << endl;
        assert(output_naive[i] == output_baseline[i]);
    }
    delete[] test_input; 
    delete[] output_naive;
    delete[] output_baseline;
    return 0;
}