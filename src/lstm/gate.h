//
// Created by Malcolm Roalson on 8/23/24.
//

#ifndef MANTIS_GATE_H
#define MANTIS_GATE_H

#include <Eigen/Core>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <iostream>
#include <fstream>

namespace mantis::lstm {
    class Gate {
        std::function<double(double)> transformer;

    public:
        MatrixXd weight;
        VectorXd bias;

        Gate(unsigned int inputSize, unsigned int outputSize, std::function<double(double)> transformer)
                : transformer(std::move(transformer)) {
            weight = MatrixXd(outputSize, inputSize);
            bias = VectorXd(outputSize);

            for (int i = 0; i < weight.rows(); i++) {
                for (int j = 0; j < weight.cols(); j++) weight(i, j) = 2 * ((double) random() / RAND_MAX) - 1;
            }

            for (int i = 0; i < bias.size(); i++) bias(i) = 2 * ((double) random() / RAND_MAX) - 1;
        }

        std::tuple<VectorXd, VectorXd> apply(const VectorXd &input) const;

        void write(std::ofstream &file) const;

        void load(std::ifstream &file);
    };
}

#endif //MANTIS_GATE_H
