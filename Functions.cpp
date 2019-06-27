//
// Created by paulnykiel on 26.06.19.
//
#include "Functions.hpp"

#include <cmath>

namespace ml::functions {
    namespace impl {
        auto sigmoid(double x) -> double {
            return 1.0 / (1.0 + std::exp(-x));
        }

        auto sigmoidDiff(double x) -> double {
            auto exp = std::exp(-x);
            auto expInc = exp + 1;
            return exp / (expInc * expInc);
        }

        auto relu(double x) -> double {
            return x > 0 ? x : 0;
        }

        auto reluDiff(double x) -> double {
            return x > 0 ? 1 : 0;
        }

        auto softplus(double x) -> double {
            return std::log(1 + std::exp(x));
        }

        auto softplusDiff(double x) -> double {
            auto exp = std::exp(x);
            return exp / (exp + 1);
        }

        auto identity(double x) -> double {
            return x;
        }

        auto identityDiff(double) -> double {
            return 1;
        }
    }

    auto TransferFunction::operator()(double x) const -> double {
        return f(x);
    }

    auto TransferFunction::derivative(double x) const -> double {
        return diff(x);
    }

    auto TransferFunction::getFunction() const -> std::function<double(double)> {
        return f;
    }

    auto TransferFunction::getDerivative() const -> std::function<double(double)> {
        return diff;
    }

    auto TransferFunction::getId() const -> std::string {
        return id;
    }

    std::map<std::string, TransferFunction> TransferFunction::functions;

    const TransferFunction sigmoid{impl::sigmoid, impl::sigmoidDiff, "sigmoid"};
    const TransferFunction relu{impl::relu, impl::reluDiff, "relu"};
    const TransferFunction softplus{impl::softplus, impl::softplusDiff, "softplus"};
    const TransferFunction identity{impl::identity, impl::identityDiff, "identity"};
}
