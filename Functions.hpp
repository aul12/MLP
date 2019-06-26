#ifndef MLPTEST_FUNCTIONS_HPP
#define MLPTEST_FUNCTIONS_HPP

#include <cmath>

namespace ml::functions {

    template<int S>
    auto meanSquareError(const std::array<double, S> a, const std::array<double, S> b) -> double {
        double error = 0;

        for (int i = 0; i < S; i++) {
            auto diff = a[i] - b[i];
            error += diff * diff;
        }

        return error;
    }

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
            return std::exp(x) / (std::exp(x) + 1);
        }

        auto identity(double x) -> double {
            return x;
        }

        auto identityDiff(double) -> double {
            return 1;
        }
    }

    class TransferFunction {
    public:
        TransferFunction() = default;

        template<typename F, typename G>
        TransferFunction(F f, G diff, std::string id) noexcept
                : f{std::move(f)}, diff{std::move(diff)}, id{std::move(id)} {

            TransferFunction::functions.emplace(this->id, *this);
        }

        auto operator()(double x) const -> double {
            return f(x);
        }

        auto derivative(double x) const -> double {
            return diff(x);
        }

        auto getFunction() const -> std::function<double(double)> {
            return f;
        }

        auto getDerivative() const -> std::function<double(double)> {
            return diff;
        }

        auto getId() const -> std::string {
            return id;
        }

    private:
        std::function<double(double)> f, diff;
        std::string id;

    public:
        static std::map<std::string, TransferFunction> functions;
    };

    std::map<std::string, TransferFunction> TransferFunction::functions;

    const TransferFunction sigmoid{impl::sigmoid, impl::sigmoidDiff, "sigmoid"};
    const TransferFunction relu{impl::relu, impl::reluDiff, "relu"};
    const TransferFunction softplus{impl::softplus, impl::softplusDiff, "softplus"};
    const TransferFunction identity{impl::identity, impl::identityDiff, "identity"};
}

#endif
