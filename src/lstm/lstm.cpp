//
// Created by Malcolm Roalson on 10/16/24.
//

#include "lstm.h"

using namespace mantis::lstm;
using namespace Eigen;

Cache* LSTM::forwardPass(const VectorXd& dataIn) {
    VectorXd gateInput(GATE_INPUT_SIZE);
    gateInput << dataIn, hiddenState;

    auto [forgetIn, forgetOut] = gates.forget.apply(gateInput);
    auto [sInputIn, sInputOut] = gates.sInput.apply(gateInput);
    auto [tInputIn, tInputOut] = gates.tInput.apply(gateInput);
    auto [outputIn, outputOut] = gates.output.apply(gateInput);

    VectorXd cellStateOut = cellState.cwiseProduct(forgetOut) +
                            sInputOut.cwiseProduct(tInputOut);
    VectorXd hiddenStateOut = cellState.unaryExpr(std::ref(util::tanh))
            .cwiseProduct(outputOut);

    auto [dataGateIn, dataOut] = gates.data.apply(hiddenStateOut);

    return new Cache{cellState, hiddenState,
                     gateInput, dataIn,
                     forgetIn, forgetOut,
                     sInputIn, sInputOut,
                     tInputIn, tInputOut,
                     outputIn, outputOut,
                     cellStateOut, hiddenStateOut,
                     dataGateIn, dataOut};
}

Gradient* LSTM::backwardPass(const VectorXd& dH,
                       const VectorXd& dC,
                       const Cache& cache) {
    VectorXd dOutput = dH.cwiseProduct(
            cache.cellStateOut.unaryExpr(std::ref(util::tanh))).cwiseProduct(
            cache.outputIn.unaryExpr(std::ref(sigmoid_der)));

    VectorXd dCFromH = dH.cwiseProduct(cache.outputOut).cwiseProduct(cache.cellStateOut.unaryExpr(std::ref(tanh_der)));
    VectorXd dtInput = (dC.cwiseProduct(cache.sInputOut) + dCFromH.cwiseProduct(cache.sInputOut)).cwiseProduct(
            cache.tInputIn.unaryExpr(std::ref(tanh_der)));

    VectorXd dsInput = (dC.cwiseProduct(cache.tInputOut) + dCFromH.cwiseProduct(cache.tInputOut)).cwiseProduct(
            cache.sInputIn.unaryExpr(std::ref(sigmoid_der)));

    VectorXd dForget = (dC.cwiseProduct(cache.cellStateIn) + dCFromH.cwiseProduct(cache.cellStateIn)).cwiseProduct(
            cache.outputIn.unaryExpr(std::ref(sigmoid_der)));


    VectorXd dHNew = (gates.forget.weight.transpose() * dForget +
                      gates.sInput.weight.transpose() * dsInput +
                      gates.tInput.weight.transpose() * dtInput +
                      gates.output.weight.transpose() * dOutput)(Eigen::seqN(INPUT_SIZE, STATE_SIZE));
    VectorXd dCNew = dC.cwiseProduct(dForget) + dCFromH.cwiseProduct(dForget);

    MatrixXd dWf = dForget * cache.gateInput.transpose();
    MatrixXd dWsi = dsInput * cache.gateInput.transpose();
    MatrixXd dWti = dtInput * cache.gateInput.transpose();
    MatrixXd dWo = dOutput * cache.gateInput.transpose();
    return new Gradient{
            dHNew, dCNew,
            dForget, dsInput,
            dtInput, dOutput,
            dWf, dWsi, dWti, dWo
    };
}

long backprops = 0;
void LSTM::backpropagation(const VectorXd& expected) {
    Cache* lastCache = caches[BACKPROP_INTERVAL-1];
    VectorXd prediction = adjustOutput(lastCache->dataOut);

    double loss = 0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        loss += pow(prediction(i) - expected(i), 2);
    }
    if (backprops++ % DEBUG_INTERVAL == 0) {
        std::cout << "loss: " << round(loss*100000)/100000 << " | " << interpretOutput(prediction) << " vs " << interpretOutput(expected) << std::endl;
    }

    VectorXd dData = (2 * (expected - prediction)).cwiseProduct(lastCache->dataGateIn.unaryExpr(std::ref(tanh_der)));

    gates.data.weight += LEARNING_MOD * (dData * lastCache->hiddenStateOut.transpose());
    gates.data.bias += LEARNING_MOD * dData;

    VectorXd dH = gates.data.weight.transpose() * dData;
    VectorXd dC = dH.cwiseProduct(lastCache->outputOut).cwiseProduct(lastCache->cellStateOut.unaryExpr(std::ref(tanh_der)));

    Gradient* totalGradient = nullptr;

    for (int i = BACKPROP_INTERVAL-1; i >= 0; i--) {
        Gradient* gradient = backwardPass(dH, dC, *caches[i]);

        dH = gradient->dH;
        dC = gradient->dC;

        if (totalGradient == nullptr) {
            totalGradient = gradient;
        } else {
            totalGradient->dWf += gradient->dWf;
            totalGradient->dBf += gradient->dBf;
            totalGradient->dWsi += gradient->dWsi;
            totalGradient->dBsi += gradient->dBsi;
            totalGradient->dWti += gradient->dWti;
            totalGradient->dBti += gradient->dBti;
            totalGradient->dWo += gradient->dWo;
            totalGradient->dBo += gradient->dBo;

            delete gradient;
        }
    }

    gates.forget.weight += LEARNING_MOD * totalGradient->dWf / BACKPROP_INTERVAL;
    gates.forget.bias += LEARNING_MOD * totalGradient->dBf / BACKPROP_INTERVAL;
    gates.sInput.weight += LEARNING_MOD * totalGradient->dWsi / BACKPROP_INTERVAL;
    gates.sInput.bias += LEARNING_MOD * totalGradient->dBsi / BACKPROP_INTERVAL;
    gates.tInput.weight += LEARNING_MOD * totalGradient->dWti / BACKPROP_INTERVAL;
    gates.tInput.bias += LEARNING_MOD * totalGradient->dBti / BACKPROP_INTERVAL;
    gates.output.weight += LEARNING_MOD * totalGradient->dWo / BACKPROP_INTERVAL;
    gates.output.bias += LEARNING_MOD * totalGradient->dBo / BACKPROP_INTERVAL;

    delete totalGradient;
}

VectorXd LSTM::predict(Eigen::VectorXd input) {
    if (t != 0 && t % BACKPROP_INTERVAL == 0) {
        backpropagation(input);
    }

    Cache* cache = forwardPass(input);
    cellState = cache->cellStateOut;
    hiddenState = cache->hiddenStateOut;

    caches.push(cache);

    t++;
    if (t == SAVE_INTERVAL) {
        std::ofstream save("gradients"+NAME+".csv");

        gates.forget.write(save);
        gates.sInput.write(save);
        gates.tInput.write(save);
        gates.output.write(save);
        gates.data.write(save);

        for (int i = 0; i < cellState.size(); i++) {
            save << cellState(i) << ",";
        }
        save << std::endl;
        for (int i = 0; i < hiddenState.size(); i++) {
            save << hiddenState(i) << ",";
        }
        save << std::endl;

        save.close();

        t = 0;
    }

    return cache->dataOut;
}