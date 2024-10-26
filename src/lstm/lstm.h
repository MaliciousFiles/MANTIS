//
// Created by Malcolm Roalson on 10/16/24.
//

#ifndef MANTIS_LSTM_H
#define MANTIS_LSTM_H

#import <string>
#import <Eigen/Core>
#import <iostream>
#import "gate.h"
#import "../util/circbuf.h"
#import "../util/math.h"
using namespace mantis::util;
using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace mantis::lstm {
    struct GateArray {
        Gate forget, sInput, tInput, output, data;

        GateArray(unsigned int inputSize, unsigned int stateSize) :
            forget(inputSize + stateSize, stateSize, sigmoid),
            sInput(inputSize + stateSize, stateSize, sigmoid),
            tInput(inputSize + stateSize, stateSize, util::tanh),
            output(inputSize + stateSize, stateSize, sigmoid),
            data(stateSize, inputSize, util::tanh) {}
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

    class LSTM {
    private:
        const unsigned int DEBUG_INTERVAL;

        const std::string NAME;
        const unsigned int STATE_SIZE, INPUT_SIZE, GATE_INPUT_SIZE, BACKPROP_INTERVAL, SAVE_INTERVAL;
        const double LEARNING_MOD;

        GateArray gates = GateArray(INPUT_SIZE, STATE_SIZE);

        const std::function<VectorXd(VectorXd)> adjustOutput;
        const std::function<std::string(VectorXd)> interpretOutput;

        circbuf<Cache*> caches;

        VectorXd cellState, hiddenState;

        long t = 0;

    public:
        LSTM(std::string name, unsigned int inputSize, unsigned int stateSize, unsigned int backpropInterval,
             unsigned int learnHandicap, unsigned int debugInterval, unsigned int saveInterval,
             std::function<VectorXd(VectorXd)> adjustOutput, std::function<std::string(VectorXd)> interpretOutput) :
            NAME(std::move(name)),
            INPUT_SIZE(inputSize), STATE_SIZE(stateSize), GATE_INPUT_SIZE(inputSize + stateSize),
            BACKPROP_INTERVAL(backpropInterval), LEARNING_MOD(pow(10.0, -(int) learnHandicap)),
            DEBUG_INTERVAL(debugInterval), SAVE_INTERVAL(saveInterval),
            caches(BACKPROP_INTERVAL, [] (Cache* cache) { delete cache; }),
            cellState(STATE_SIZE), hiddenState(STATE_SIZE),
            adjustOutput(std::move(adjustOutput)), interpretOutput(std::move(interpretOutput)) {
            std::ifstream load("gradients_"+name+".csv");
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
        };

        Cache* forwardPass(const VectorXd& dataIn);
        Gradient* backwardPass(const VectorXd& dH, const VectorXd& dC, const Cache& cache);
        void backpropagation(const VectorXd& expected);

        VectorXd predict(VectorXd input);
    };
}

#endif //MANTIS_LSTM_H
