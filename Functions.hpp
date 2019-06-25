#ifndef MLPTEST_FUNCTIONS_HPP
#define MLPTEST_FUNCTIONS_HPP

#include <cmath>

namespace ml::functions {

    template <int S>
    auto meanSquareError(const std::array<double, S> a, const std::array<double, S> b) -> double {
        double error = 0;

        for(int i = 0; i<S; i++) {
            auto diff = a[i] - b[i];
            error += diff*diff;
        }

        return error;
    }

    auto sigmoid(double x) -> double {
        return 1.0/(1.0 + std::exp(-x));
    }

    auto sigmoidDiff(double x) -> double {
        auto exp = std::exp(-x);
        auto expInc = exp + 1;
        return exp/(expInc * expInc);
    }

    auto relu(double x) -> double  {
        return x > 0 ? x : 0;
    }

    auto reluDiff(double x) -> double {
        return x > 0 ? 1 : 0;
    }

    auto softplus(double x) -> double {
        return std::log(1 + std::exp(x));
    }

    auto softplusDiff(double x) -> double {
        return std::exp(x) / (std::exp(x) + 1);
    }

    auto identity(double x) -> double {
        return x;
    }

    auto identityDiff(double) -> double {
        return 1;
    }
}

#endif
