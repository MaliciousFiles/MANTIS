#include "lstm/lstm.h"
#include "inputs.cpp"
#include <set>
#include <thread>
#include <matplot/matplot.h>

#define GRAPH false

using namespace mantis;
using namespace matplot;
using mantis::lstm::LSTM;

int main() {
    srandom(time(nullptr));

    LSTM math("math", 1, 7, 10, 2, 2001,
                 1000000000, outputAdjusters::none, outputInterpreters::basic);
    LSTM userAuth("user_auth", inputs::userAuthSize, 25, 10, 3, 501,
              1000000000, outputAdjusters::userAuth, outputInterpreters::userAuth);

#if GRAPH
    vector<vector<double>> inputs;
    vector<vector<double>> predictions;
    vector<figure_handle> figures;
    vector<vector<line_handle>> lines;

    const int graphs = 4;
    for (int i = 0; i < graphs; i++) {
        vector<double> input;
        vector<double> prediction;

        auto f = figure(true);
        auto plots = plot(input, "-b", prediction, "-r");
        ylim({-2, 2});

        figures.push_back(f);
        lines.push_back(plots);
        inputs.push_back(input);
        predictions.push_back(input);

        f->draw();
    }

    cout << "Setup windows";
    cin.ignore();

    bool updating = false;
    auto toggleUpdating = [&] () mutable -> void {
        while (1) {
            cin.ignore();
            updating = !updating;
        }
    };
    thread updateThread(toggleUpdating);
#endif

    long t = 0;
    while (true) {
        VectorXd input = inputs::singleUserDayJob(t);
        VectorXd prediction = userAuth.predict(input);

#if GRAPH
        if (updating) {
            for (int i = 0; i < graphs; i++) {
                inputs[i].push_back(input(i));
                predictions[i].push_back(prediction(i));

                lines[i][0]->y_data(predictions[i]);
                lines[i][1]->y_data(inputs[i]);

                figures[i]->draw();
            }
        }
#endif

        t++;
    }
}
