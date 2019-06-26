//
// Created by paul on 19.04.18.
//

#ifndef MLPTEST_LAYER_HPP
#define MLPTEST_LAYER_HPP

#include <array>
#include <functional>
#include <tuple>
#include <random>

#include "json.hpp"

namespace ml {
    template<int InputSize, int OutputSize>
    class Layer {
    public:
        Layer() {
            std::random_device rd;  //Will be used to obtain a seed for the random number engine
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
            std::normal_distribution<> dis;
            for (auto &weight : weights) {
                weight = dis(gen);
            }

            for (auto &bias : biases) {
                bias = dis(gen);
            }
        }

        auto forward(const std::array<double, InputSize> &inputVec,
                       const std::function<double(double)> &activationFunction) {
            for (auto o = 0; o < OutputSize; o++) {
                lastDendriticPotential[o] = 0;
                for (auto i = 0; i < InputSize; i++) {
                    lastDendriticPotential[o] += inputVec[i] * weights[o * InputSize + i];
                }
                lastDendriticPotential[o] += biases[o];
                lastOutput[o] = activationFunction(lastDendriticPotential[o]);
            }
            return lastOutput;
        }

        auto backPropagate(const std::array<double, OutputSize> &errorVec,
                           const std::function<double(double)> &transDiff) {
            std::array<double, InputSize> errorInPrevLayer;
            for (auto i = 0; i < InputSize; i++) {
                errorInPrevLayer[i] = 0;
                for (auto o = 0; o < OutputSize; o++) {
                    errorInPrevLayer[i] +=
                            errorVec[o] * weights[o * InputSize + i] * transDiff(lastDendriticPotential[o]);
                }
            }
            return errorInPrevLayer;
        }

        auto adaptWeights(const std::array<double, OutputSize> &errorVec,
                          const std::array<double, InputSize> &input, double learnRate) {
            for (auto o = 0; o < OutputSize; o++) {
                for (auto i = 0; i < InputSize; i++) {
                    weights[o * InputSize + i] += errorVec[o] * learnRate * input[i];
                }
                biases[o] += errorVec[o] * learnRate;
            }
        }

    private:
        std::array<double, InputSize * OutputSize> weights;
        std::array<double, OutputSize> biases;
        std::array<double, OutputSize> lastDendriticPotential;
        std::array<double, OutputSize> lastOutput;

    public:
        friend void to_json(nlohmann::json& j, const Layer<InputSize, OutputSize> &layer) {
            j["inputSize"] = InputSize;
            j["outputSize"] = OutputSize;
            j["weights"] = layer.weights;
            j["biases"] = layer.biases;
        }

        friend void from_json(const nlohmann::json &j, Layer<InputSize, OutputSize> &layer) {
            assert(InputSize == j.at("inputSize").get<int>());
            assert(OutputSize == j.at("outputSize").get<int>());
            layer.weights = j.at("weights").get<std::array<double, InputSize*OutputSize>>();
            layer.biases = j.at("biases").get<std::array<double, OutputSize>>();
        }
    };


}

#endif //MLPTEST_LAYER_HPP
