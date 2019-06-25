//
// Created by paul on 19.04.18.
//

#ifndef MLPTEST_LAYER_HPP
#define MLPTEST_LAYER_HPP

#include <array>
#include <functional>
#include <tuple>
#include <random>

template <int InputSize, int OutputSize>
class Layer {
public:
    Layer() {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        for(auto o=0; o<OutputSize; o++) {
            for(auto i=0; i<InputSize; i++) {
                weights[o][i] = dis(gen);
            }
            biases[o] = dis(gen);
        }
    }

    auto propagate(const std::array<double, InputSize> &inputVec,
                   const std::function<double(double)> &activationFunction) {
        for(auto o = 0; o < OutputSize; o++) {
            lastDendriticPotential[o] = 0;
            for(auto i = 0; i < InputSize; i++) {
                lastDendriticPotential[o] += inputVec[i] * weights[o][i];
            }
            lastDendriticPotential[o] += biases[o];
            lastOutput[o] = activationFunction(lastDendriticPotential[o]);
        }
        return lastOutput;
    }

    auto backPropagate(const std::array<double, OutputSize> &errorVec,
                       const std::function<double(double)> &transDiff) {
        std::array<double, InputSize> errorInPrevLayer;
        for(auto i = 0; i < InputSize; i++) {
            errorInPrevLayer[i] = 0;
            for(auto o = 0; o < OutputSize; o++) {
                errorInPrevLayer[i] += errorVec[o] * weights[o][i] * transDiff(lastDendriticPotential[o]);
            }
        }
        return errorInPrevLayer;
    }

    auto adaptWeights(const std::array<double, OutputSize> &errorVec,
                      const std::array<double,InputSize> &input, double learnRate) {
        for(auto o = 0; o < OutputSize; o++) {
            for(auto i = 0; i < InputSize; i++) {
                weights[o][i] += errorVec[o] * learnRate * input[i];
            }
            biases[o] += errorVec[o] * learnRate;
        }
    }

private:
    std::array<std::array<double,InputSize>,OutputSize> weights;
    std::array<double, OutputSize> biases;
    std::array<double, OutputSize> lastDendriticPotential;
    std::array<double, OutputSize> lastOutput;
};


#endif //MLPTEST_LAYER_HPP
