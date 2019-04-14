#include "cnn.h"

using namespace std;

int main(int argc, char **argv)
{
    // Initialize input image
    srand(time(NULL));

    global_config_t test_global = {100, 64, 64, 5, 3, 5, 5, 2, 2};
    
    int input_w = test_global.sigW * test_global.W + test_global.R;
    int input_h = test_global.sigH * test_global.H + test_global.S;

    const int input_size = input_h * input_w * test_global.C * test_global.B;
    const int output_size = test_global.K * test_global.H * test_global.W * test_global.B;


    int test_input[input_size];

    for (int i = 0; i < input_size; i++)
    {
        cout << i << endl;
        test_input[i] = rand() % 256;
    }

    int output_naive[output_size], output_baseline[output_size];

    // set_random_array(test_input, input_size);
    return 0;
}