//
// Created by paul on 25.05.18.
//

#ifndef MLPTEST_MLP_HPP
#define MLPTEST_MLP_HPP

#include <cassert>
#include "Layer.hpp"

// Helper
template<unsigned int index, unsigned int... remPack> struct getVal;
template<unsigned int index, unsigned int In, unsigned int... remPack> struct getVal<index, In,remPack...>
{
    static const unsigned int val = getVal<index-1, remPack...>::val;
};
template<unsigned int In, unsigned int...remPack> struct getVal<1,In,remPack...>
{
    static const unsigned int val = In;
};
template<unsigned int...remPack> struct getLast
{
    static const unsigned int val = getVal<sizeof...(remPack), remPack...>::val;
};


template<int INPUT, int OUTPUT, int ... FOLLOWING_NEURONS>
class Mlp {
public:
    Mlp(const std::function<double(double)> &transfer,
            const std::function<double(double)> &transferDiff, const double learnRate)
            : learnRate(learnRate), followingMlp(transfer, transferDiff, learnRate),
                transfer(transfer), transferDiff(transferDiff), layer(){};

    auto forward(std::array<double, INPUT> x) -> std::array<double, getLast<FOLLOWING_NEURONS...>::val> {
        return followingMlp.forward(layer.propagate(x,transfer));
    }

    auto train(std::vector<std::array<double, INPUT>> inputs,
               std::vector<std::array<double,getLast<FOLLOWING_NEURONS...>::val>> outputs, double maxError) {
        assert(inputs.size() == outputs.size());
        auto error = std::numeric_limits<double>::max();
        while (error > maxError) {
            for (auto c = 0; c < inputs.size(); c++) {
                adapt(inputs[c], outputs[c]);
            }

            error = 0.0;
            for (auto c = 0; c < inputs.size(); c++) {
                auto mlpOutput = forward(inputs[c]);
                for(auto i = 0; i<getLast<FOLLOWING_NEURONS...>::val; i++) {
                    error += std::pow(mlpOutput[i] - outputs[c][i],2);
                }
            }
        }
        return error;
    }

    auto adapt(std::array<double,INPUT> input, std::array<double, getLast<FOLLOWING_NEURONS...>::val> trainerOutput) -> std::array<double, INPUT> {
        auto output = layer.propagate(input,transfer);
        auto outputError = followingMlp.adapt(output, trainerOutput);
        auto inputError = layer.backPropagate(outputError,transferDiff);
        layer.adaptWeights(outputError, input, learnRate);
        return inputError;
    }

private:
    Mlp<OUTPUT, FOLLOWING_NEURONS...> followingMlp;
    Layer<INPUT, OUTPUT> layer;
    const std::function<double(double)> &transfer;
    const std::function<double(double)> &transferDiff;
    const double learnRate;
};

template<int INPUT, int OUTPUT>
class Mlp<INPUT,OUTPUT> {
public:
    Mlp(const std::function<double(double)> &transfer,
                 const std::function<double(double)> &transferDiff, const double learnRate)
            : learnRate(learnRate),
              transfer(transfer), transferDiff(transferDiff){};

    std::array<double, OUTPUT> forward(std::array<double, INPUT> x) {
        return layer.propagate(x, transfer);
    }

    auto adapt(std::array<double,INPUT> input, std::array<double, OUTPUT> trainerOutput) -> std::array<double, INPUT> {
        auto mlpOutput = layer.propagate(input, transfer);
        std::array<double, OUTPUT> outputErrror;
        for(auto c=0; c<OUTPUT; c++) {
            outputErrror[c] = trainerOutput[c] - mlpOutput[c];
        }
        auto inputError = layer.backPropagate(outputErrror, transferDiff);
        layer.adaptWeights(outputErrror, input, learnRate);
        return inputError;
    }

private:
    Layer<INPUT, OUTPUT> layer;
    const std::function<double(double)> &transfer;
    const std::function<double(double)> &transferDiff;
    const double learnRate;
};


#endif //MLPTEST_MLP_HPP
