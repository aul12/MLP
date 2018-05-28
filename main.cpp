#include <iostream>
#include <limits>
#include <cmath>
#include "Layer.hpp"
#include "Mlp.hpp"


int main() {

    std::function<double(double)> sigmoid = [](auto x){
        return 1.0/(1.0 + std::exp(-x));
    };
    std::function<double(double)> sigmoid_diff = [](auto x) {
        return std::exp(-x)/std::pow(1+std::exp(-x), 2);
    };

    std::vector<std::array<double,2>> inputValues = {{{0,0}, {0,1}, {1,0}, {1,1}}};
    std::vector<std::array<double,2>> outputValues = {{{0,0}, {0,1}, {0,1}, {1,0}}};

    Mlp<2,100,100,2> mlp(sigmoid, sigmoid_diff, 0.01);


    std::cout << mlp.train(inputValues, outputValues, 0.001) << std::endl;

    for(auto s=0; s<4; s++) {
        const auto &inputVec = inputValues[s];
        const auto &trainerOutput = outputValues[s];
        const auto out = mlp.forward(inputVec);
        std::cout << "[" << inputVec[0] << ", " << inputVec[1] << "] -> [" << out[0] << "," << out[1] << "]" << std::endl;
    }


    return 0;
}