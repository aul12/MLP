#include <utility>

//
// Created by paul on 25.05.18.
//

#ifndef MLPTEST_MLP_HPP
#define MLPTEST_MLP_HPP

#include <cassert>
#include <utility>
#include "Layer.hpp"
#include "Functions.hpp"

namespace ml {
    namespace helper {
        template<unsigned int index, unsigned int... remPack>
        struct getVal;
        template<unsigned int index, unsigned int In, unsigned int... remPack>
        struct getVal<index, In, remPack...> {
            static const unsigned int val = getVal<index - 1, remPack...>::val;
        };
        template<unsigned int In, unsigned int...remPack>
        struct getVal<1, In, remPack...> {
            static const unsigned int val = In;
        };
        template<unsigned int...remPack>
        struct getLast {
            static const unsigned int val = getVal<sizeof...(remPack), remPack...>::val;
        };
    }

    template<int INPUT, int OUTPUT, int ... FOLLOWING_LAYERS>
    class Mlp {
    public:
        static constexpr auto LAST_OUTPUT = helper::getLast<FOLLOWING_LAYERS...>::val;
        using TransferF = std::function<double(double)>;
        using CostF = std::function<double(std::array<double, LAST_OUTPUT>, std::array<double, LAST_OUTPUT>)>;

        Mlp() = default;
        explicit Mlp(const functions::TransferFunction &transferFunction)
                : followingMlp(transferFunction), layer{},
                  transferFunction{transferFunction} {};

        auto forward(const std::array<double, INPUT> &x) const -> std::array<double, LAST_OUTPUT> {
            auto thisLayerResult = layer.forward(x, transferFunction);
            return followingMlp.forward(thisLayerResult);
        }

        auto train(const std::vector<std::array<double, INPUT>> &inputs,
                   const std::vector<std::array<double, LAST_OUTPUT>> &outputs, double maxError,
                   const CostF &costF, double learnRate,
                   const std::optional<std::function<void(double)>> &errorCallback = std::nullopt) {
            assert(inputs.size() == outputs.size());
            auto error = std::numeric_limits<double>::max();
            while (error > maxError) {
                for (std::size_t c = 0; c < inputs.size(); c++) {
                    adapt(inputs[c], outputs[c], learnRate);
                }

                error = 0.0;
                for (std::size_t c = 0; c < inputs.size(); c++) {
                    auto mlpOutput = forward(inputs[c]);

                    error += costF(mlpOutput, outputs[c]);
                }

                if (errorCallback.has_value()) {
                    errorCallback.value()(error);
                }
            }
            return error;
        }

        auto adapt(const std::array<double, INPUT> &input,
                   const std::array<double, LAST_OUTPUT> &trainerOutput,
                   double learnRate) -> std::array<double, INPUT> {
            auto output = layer.forward(input, transferFunction);
            auto outputError = followingMlp.adapt(output, trainerOutput, learnRate);
            auto inputError = layer.backPropagate(outputError, transferFunction.getDerivative());
            layer.adaptWeights(outputError, input, learnRate);
            return inputError;
        }

    private:
        Mlp<OUTPUT, FOLLOWING_LAYERS...> followingMlp;
        Layer <INPUT, OUTPUT> layer;
        functions::TransferFunction transferFunction;

    public:
        friend void to_json(nlohmann::json& j, const Mlp<INPUT, OUTPUT, FOLLOWING_LAYERS...> &mlp) {
            nlohmann::json layerJson;
            layerJson["layer"] = mlp.layer;
            layerJson["transferFunction"] = mlp.transferFunction.getId();

            j["layers"].emplace_back(layerJson);
            to_json(j, mlp.followingMlp); // Yes CLion is unhappy here, it doesn't understand the recursive template
        }

        friend void from_json(const nlohmann::json& j, Mlp<INPUT, OUTPUT, FOLLOWING_LAYERS...> &mlp) {
            assert(!j["layers"].empty());
            auto it = j.at("layers").begin();
            mlp.layer = it->at("layer").get<Layer<INPUT,OUTPUT>>();
            nlohmann::json newJson;
            newJson["layers"] = nlohmann::json::array();
            for (++it; it != j.at("layers").end(); ++it) {
                 newJson["layers"].emplace_back(*it);
            }
            mlp.transferFunction = functions::TransferFunction::functions.at(
                    j.at("layers").begin()->at("transferFunction").get<std::string>());
            mlp.followingMlp = newJson.get<Mlp<OUTPUT, FOLLOWING_LAYERS...>>();
        }
    };

    template<int INPUT, int OUTPUT>
    class Mlp<INPUT, OUTPUT> {
    public:
        static constexpr auto LAST_OUTPUT = OUTPUT;
        using TransferF = std::function<double(double)>;
        using CostF = std::function<double(std::array<double, LAST_OUTPUT>, std::array<double, LAST_OUTPUT>)>;

        Mlp() = default;
        explicit Mlp(functions::TransferFunction transferFunction)
                : layer{}, transferFunction{std::move(transferFunction)} {};

        auto forward(std::array<double, INPUT> x) const -> std::array<double, OUTPUT> {
            return layer.forward(x, transferFunction);
        }

        auto
        adapt(std::array<double, INPUT> input, std::array<double, OUTPUT> trainerOutput,
                double learnRate) -> std::array<double, INPUT> {
            auto mlpOutput = layer.forward(input, transferFunction);
            std::array<double, OUTPUT> outputErrror;
            for (auto c = 0; c < OUTPUT; c++) {
                outputErrror[c] = trainerOutput[c] - mlpOutput[c];
            }
            auto inputError = layer.backPropagate(outputErrror, transferFunction.getDerivative());
            layer.adaptWeights(outputErrror, input, learnRate);
            return inputError;
        }

    private:
        Layer <INPUT, OUTPUT> layer;
        functions::TransferFunction transferFunction;

    public:
        friend void to_json(nlohmann::json& j, const Mlp<INPUT, OUTPUT> &mlp) {
            nlohmann::json layerJson;
            layerJson["layer"] = mlp.layer;
            layerJson["transferFunction"] = mlp.transferFunction.getId();

            j["layers"].emplace_back(layerJson);
        }

        friend void from_json(const nlohmann::json& j, Mlp<INPUT, OUTPUT> &mlp) {
            assert(j["layers"].size() == 1);
            auto it = j.at("layers").begin();
            mlp.layer = it->at("layer").get<Layer<INPUT,OUTPUT>>();
            mlp.transferFunction = functions::TransferFunction::functions.at(
                    it->at("transferFunction").get<std::string>());
        }
    };
}

#endif //MLPTEST_MLP_HPP
