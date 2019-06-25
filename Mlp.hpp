//
// Created by paul on 25.05.18.
//

#ifndef MLPTEST_MLP_HPP
#define MLPTEST_MLP_HPP

#include <cassert>
#include <utility>
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
    static constexpr auto LAST_OUTPUT = getLast<FOLLOWING_NEURONS...>::val;
    using TransferF = std::function<double(double)>;
    using CostF = std::function<double(std::array<double, LAST_OUTPUT>, std::array<double, LAST_OUTPUT>)>;

    Mlp(const TransferF &transfer, const TransferF &transferDiff, const CostF &costF,
            double learnRate)
            :  followingMlp(transfer, transferDiff, costF, learnRate), layer{},
                transfer(transfer), transferDiff(transferDiff), costF{costF}, learnRate(learnRate) {};

    auto forward(const std::array<double, INPUT> &x) -> std::array<double, LAST_OUTPUT> {
        return followingMlp.forward(layer.propagate(x,transfer));
    }

    auto train(const std::vector<std::array<double, INPUT>> &inputs,
               const std::vector<std::array<double, LAST_OUTPUT>> &outputs, double maxError) {
        assert(inputs.size() == outputs.size());
        auto error = std::numeric_limits<double>::max();
        while (error > maxError) {
            for (std::size_t c = 0; c < inputs.size(); c++) {
                adapt(inputs[c], outputs[c]);
            }

            error = 0.0;
            for (std::size_t c = 0; c < inputs.size(); c++) {
                auto mlpOutput = forward(inputs[c]);

                error += costF(mlpOutput, outputs[c]);
            }
        }
        return error;
    }

    auto adapt(const std::array<double,INPUT> &input, const std::array<double, LAST_OUTPUT> &trainerOutput) -> std::array<double, INPUT> {
        auto output = layer.propagate(input,transfer);
        auto outputError = followingMlp.adapt(output, trainerOutput);
        auto inputError = layer.backPropagate(outputError,transferDiff);
        layer.adaptWeights(outputError, input, learnRate);
        return inputError;
    }

private:
    Mlp<OUTPUT, FOLLOWING_NEURONS...> followingMlp;
    Layer<INPUT, OUTPUT> layer;
    const TransferF transfer;
    const TransferF transferDiff;
    const CostF costF;
    const double learnRate;
};

template<int INPUT, int OUTPUT>
class Mlp<INPUT,OUTPUT> {
public:
    static constexpr auto LAST_OUTPUT = OUTPUT;
    using TransferF = std::function<double(double)>;
    using CostF = std::function<double(std::array<double, LAST_OUTPUT>, std::array<double, LAST_OUTPUT>)>;

    Mlp(TransferF transfer, TransferF transferDiff, const CostF &costF,
            double learnRate)
            :  layer{}, transfer(std::move(transfer)), transferDiff(std::move(transferDiff)), costF{costF}, learnRate(learnRate) {};

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
    const TransferF transfer;
    const TransferF transferDiff;
    const CostF costF;
    const double learnRate;
};


#endif //MLPTEST_MLP_HPP
