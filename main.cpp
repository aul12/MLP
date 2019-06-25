#include <iostream>
#include "Layer.hpp"
#include "Mlp.hpp"
#include "Functions.hpp"

int main() {
    std::vector<std::array<double,2>> inputValues{{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<std::array<double,2>> outputValues{{0,0}, {0,1}, {0,1}, {1,0}};

    ml::Mlp<2,5,5,2> mlp{ml::functions::relu,
            ml::functions::reluDiff,
            ml::functions::meanSquareError<2>, 0.01};

    auto errorCallback = [](double error) {
        std::cout << "Error: " << error << std::endl;
    };

    std::cout << mlp.train(inputValues, outputValues, 0.1, errorCallback) << std::endl;

    for(auto s=0; s<4; s++) {
        const auto &inputVec = inputValues[s];
        const auto out = mlp.forward(inputVec);
        std::cout << "[" << inputVec[0] << ", " << inputVec[1] << "] -> [" << out[0] << "," << out[1] << "]" << std::endl;
    }


    return 0;
}
