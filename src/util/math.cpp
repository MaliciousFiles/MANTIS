//
// Created by Malcolm Roalson on 10/16/24.
//
#import <cmath>

#include "./math.h"

namespace mantis::util {
    double sigmoid(double in) {
        return in / (1 + abs(in));
    }

    double sigmoid_der(double in) {
        return 1 / pow(1 + abs(in), 2);
    }

    double tanh(double in) {
        return std::tanh(in);
    }

    double tanh_der(double in) {
        return 1-pow(std::tanh(in), 2);
    }
}