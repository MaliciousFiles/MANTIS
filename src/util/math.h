//
// Created by Malcolm Roalson on 10/26/24.
//

#ifndef MANTIS_MATH_H
#define MANTIS_MATH_H

namespace mantis::util {
    double sigmoid(double in);
    double sigmoid_der(double in);
    double tanh(double in);
    double tanh_der(double in);
}

#endif //MANTIS_MATH_H
