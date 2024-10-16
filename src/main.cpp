#include <iostream>
#include <fstream>
#include <vector>
#include "util/circbuf.h"
#include "util/gate.h"
#include "inputs.cpp"

#define DEBUG_INTERVAL 501
#define RND 10000000
#define CELL_STATE_SIZE 26
#define BACKPROP_INTERVAL 10
#define SAVE_INTERVAL 1000000000
#define LEARNING_MOD 0.01
#define GATE_INPUT_SIZE (INPUT_SIZE + CELL_STATE_SIZE)

#define INPUT inputs::math/*twoUserOverlap/*singleUserDayJob*/
#define INPUT_SIZE inputs::mathSize/*twoUserOverlapSize/*singleUserDayJobSize*/
#define ADJUST_OUTPUT outputAdjusters::none/*userAuth*/
#define INTERPRET_OUTPUT outputInterpreters::basic/*userAuth*/

namespace mantis {
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

    struct GateArray {
        Gate forget = Gate(GATE_INPUT_SIZE, CELL_STATE_SIZE, sigmoid);
        Gate sInput = Gate(GATE_INPUT_SIZE, CELL_STATE_SIZE, sigmoid);
        Gate tInput = Gate(GATE_INPUT_SIZE, CELL_STATE_SIZE, tanh);
        Gate output = Gate(GATE_INPUT_SIZE, CELL_STATE_SIZE, sigmoid);
        Gate data = Gate(CELL_STATE_SIZE, INPUT_SIZE, tanh);
    };

    struct Cache {
        const VectorXd& cellStateIn, hiddenStateIn,
                        gateInput, dataIn,
                        forgetIn, forgetOut,
                        sInputIn, sInputOut,
                        tInputIn, tInputOut,
                        outputIn, outputOut,
                        cellStateOut, hiddenStateOut,
                        dataGateIn, dataOut;
    };

    Cache* forwardPass(const GateArray& gates,
                      const VectorXd& cellStateIn,
                      const VectorXd& hiddenStateIn,
                      const VectorXd& dataIn) {
        VectorXd gateInput(GATE_INPUT_SIZE);
        gateInput << dataIn, hiddenStateIn;

        auto [forgetIn, forgetOut] = gates.forget.apply(gateInput);
        auto [sInputIn, sInputOut] = gates.sInput.apply(gateInput);
        auto [tInputIn, tInputOut] = gates.tInput.apply(gateInput);
        auto [outputIn, outputOut] = gates.output.apply(gateInput);

        VectorXd cellStateOut = cellStateIn.cwiseProduct(forgetOut) +
                sInputOut.cwiseProduct(tInputOut);
        VectorXd hiddenStateOut = cellStateIn.unaryExpr(std::ref(tanh))
                .cwiseProduct(outputOut);

        auto [dataGateIn, dataOut] = gates.data.apply(hiddenStateOut);

        return new Cache{cellStateIn, hiddenStateIn,
                     gateInput, dataIn,
                     forgetIn, forgetOut,
                     sInputIn, sInputOut,
                     tInputIn, tInputOut,
                     outputIn, outputOut,
                     cellStateOut, hiddenStateOut,
                     dataGateIn, dataOut};
    }

    struct Gradient {
        VectorXd dH,
                  dC,
                  dBf,
                  dBsi,
                  dBti,
                  dBo;
        MatrixXd dWf,
                  dWsi,
                  dWti,
                  dWo;
    };
    Gradient* backwardPass(const VectorXd& dH,
                      const VectorXd& dC,
                      const GateArray& gates,
                      const Cache& cache) {
        VectorXd dOutput = dH.cwiseProduct(
                cache.cellStateOut.unaryExpr(std::ref(tanh))).cwiseProduct(
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
                gates.output.weight.transpose() * dOutput)(Eigen::seqN(INPUT_SIZE, CELL_STATE_SIZE));
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
    void backpropagation(long t,
                         GateArray& gates,
                         const circbuf<Cache*>& caches,
                         const VectorXd& expected) {
        Cache* lastCache = caches[BACKPROP_INTERVAL-1];
        VectorXd prediction = ADJUST_OUTPUT(lastCache->dataOut);

        double loss = 0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            loss += pow(prediction(i) - expected(i), 2);
        }
        if (backprops++ % DEBUG_INTERVAL == 0) {
            std::cout << "loss: " << round(loss*RND)/RND << " | " << INTERPRET_OUTPUT(prediction) << " vs " << INTERPRET_OUTPUT(expected) << std::endl;
        }

        VectorXd dData = (2 * (expected - prediction)).cwiseProduct(lastCache->dataGateIn.unaryExpr(std::ref(tanh_der)));

        gates.data.weight += LEARNING_MOD * (dData * lastCache->hiddenStateOut.transpose());
        gates.data.bias += LEARNING_MOD * dData;

        VectorXd dH = gates.data.weight.transpose() * dData;
        VectorXd dC = dH.cwiseProduct(lastCache->outputOut).cwiseProduct(lastCache->cellStateOut.unaryExpr(std::ref(tanh_der)));

        Gradient* totalGradient = nullptr;

        for (int i = BACKPROP_INTERVAL-1; i >= 0; i--) {
            Gradient* gradient = backwardPass(dH, dC, gates, *caches[i]);

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
}

using namespace mantis;
using namespace std;
int main() {
    srandom(time(nullptr));

    GateArray gates;

    VectorXd cellState(CELL_STATE_SIZE);
    VectorXd hiddenState(CELL_STATE_SIZE);

    ifstream load("gradients.csv");
    if (load.good()) {
        gates.forget.load(load);
        gates.sInput.load(load);
        gates.tInput.load(load);
        gates.output.load(load);
        gates.data.load(load);

        std::string line;
        std::getline(load, line);
        std::stringstream ss1(line);
        std::string num;
        for (int i = 0; i < cellState.size(); i++) {
            getline(ss1, num, ',');
            cellState(i) = std::stod(num);
        }

        std::getline(load, line);
        std::stringstream ss2(line);
        for (int i = 0; i < hiddenState.size(); i++) {
            getline(ss2, num, ',');
            hiddenState(i) = std::stod(num);
        }
    }
    load.close();

    circbuf<Cache*> caches(BACKPROP_INTERVAL, [] (Cache* cache) {
        delete cache;
    });

    long t = 0;
    while (true) {
        if (t != 0 && t % BACKPROP_INTERVAL == 0) {
            backpropagation(t, gates, caches, INPUT(t));
        }

        Cache* cache = forwardPass(gates, cellState, hiddenState, INPUT(t));
        cellState = cache->cellStateOut;
        hiddenState = cache->hiddenStateOut;

        caches.push(cache);

        t++;
        if (t == SAVE_INTERVAL) {
            ofstream save("gradients.csv");

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
    }
}
