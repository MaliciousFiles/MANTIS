//
// Created by Malcolm Roalson on 8/23/24.
//

#include "gate.h"
using namespace mantis::lstm;

std::tuple<VectorXd, VectorXd> Gate::apply(const VectorXd& gateInput) const {
    VectorXd input = weight * gateInput + bias;

    VectorXd output = input.unaryExpr(std::ref(transformer));

    return {input, output};
}

void Gate::write(std::ofstream &file) const {
    for (int r = 0; r < weight.rows(); r++) {
        for (int c = 0; c < weight.cols(); c++) {
            file << weight(r,c) << ",";
        }
    }

    for (int i = 0; i < bias.size(); i++) {
        file << bias(i) << ",";
    }

    file << std::endl;
}

void Gate::load(std::ifstream &file) {
    std::string line;
    std::getline(file, line);

    std::stringstream ss(line);

    std::string num;
    for (int r = 0; r < weight.rows(); r++) {
        for (int c = 0; c < weight.cols(); c++) {
            std::getline(ss, num, ',');
            weight(r, c) = std::stod(num);
        }
    }

    for (int i = 0; i < bias.size(); i++) {
        std::getline(ss, num, ',');
        bias(i) = std::stod(num);
    }
}