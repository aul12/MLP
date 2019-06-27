#ifndef MLPTEST_FUNCTIONS_HPP
#define MLPTEST_FUNCTIONS_HPP

#include <map>
#include <array>
#include <functional>

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

    class TransferFunction {
    public:
        TransferFunction() = default;

        template<typename F, typename G>
        TransferFunction(F f, G diff, std::string id) noexcept
                : f{std::move(f)}, diff{std::move(diff)}, id{std::move(id)} {

            TransferFunction::functions.emplace(this->id, *this);
        }

        auto operator()(double x) const -> double;

        auto derivative(double x) const -> double;

        auto getFunction() const -> std::function<double(double)>;

        auto getDerivative() const -> std::function<double(double)>;

        auto getId() const -> std::string;

    private:
        std::function<double(double)> f, diff;
        std::string id;

    public:
        static std::map<std::string, TransferFunction> functions;
    };


    extern const TransferFunction sigmoid, relu, softplus, identity;
}

#endif
