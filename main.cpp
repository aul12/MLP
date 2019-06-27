#include <iostream>
#include "Layer.hpp"
#include "Mlp.hpp"
#include "Functions.hpp"
#include "Util.h"

int main() {
    std::vector<std::array<double,3>> inputValues{{0,0,0}, {0,0,1}, {0,1,0}, {0,1,1},{1,0,0}, {1,0,1}, {1,1,0}, {1,1,1}};
    std::vector<std::array<double,2>> outputValues{{0,0}, {0,1}, {0,1},{1,0},{0,1},{1,0},{1,0},{1,1}};

    ml::Mlp<3,20,20,2> mlp{ml::functions::sigmoid};

    auto errorCallback = [](double error) {
        std::cout << "Error: " << error << std::endl;
    };

    std::cout << mlp.train(inputValues, outputValues, 0.1,
            ml::functions::meanSquareError<2>, 0.01, errorCallback) << std::endl;

    for(const auto & inputVec : inputValues) {
        const auto out = mlp.forward(inputVec);
        std::cout << "[" << inputVec[0] << ", " << inputVec[1] << ", " << inputVec[2]
                << "] -> [" << out[0] << ", " << out[1] << "]" << std::endl;
    }

    ml::util::saveToFile("test.json", mlp);

    auto newMlp = ml::util::loadFromFile<3,20,20,2>("test.json");

    return 0;
}
