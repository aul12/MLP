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
        return std::exp(-x)/std::pow(1+std::exp(-x), 2);
    }
}

#endif
