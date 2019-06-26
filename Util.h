//
// Created by paulnykiel on 26.06.19.
//

#ifndef MLP_UTIL_H
#define MLP_UTIL_H

#include <fstream>

namespace ml::util {
    template <int ... LAYERS>
    auto loadFromFile(std::istream &istream) -> Mlp<LAYERS...> {
        nlohmann::json j;
        istream >> j;
        return j.get<Mlp<LAYERS...>>();
    }

    template <int ... LAYERS>
    auto loadFromFile(const std::string &fname) -> Mlp<LAYERS...> {
        std::ifstream ifstream{fname};
        if (!ifstream) {
            throw std::runtime_error{"Could not open file for reading!"};
        }

        return loadFromFile<LAYERS...>(ifstream);
    }

    template <int ... LAYERS>
    void saveToFile(std::ostream &ostream, const Mlp<LAYERS...> &mlp) {
        nlohmann::json j = mlp;
        ostream << j.dump(4);
    }

    template <int ... LAYERS>
    void saveToFile(const std::string &fname, const Mlp<LAYERS...> &mlp) {
        std::ofstream ofstream{fname};
        if (!ofstream) {
            throw std::runtime_error{"Could not open file for writing!"};
        }

        saveToFile(ofstream, mlp);
    }
}

#endif //MLP_UTIL_H
