#include "lstm/lstm.h"
#include "inputs.cpp"
#include <set>
#include <matplot/matplot.h>

using namespace mantis;
using namespace matplot;
using mantis::lstm::LSTM;

int main() {
    srandom(time(nullptr));

    LSTM math("math", 1, 7, 10, 2, 2001,
                 1000000000, outputAdjusters::none, outputInterpreters::basic);


    vector<double> realY1, predY1, realY2, predY2, realY3, predY3;

    auto f1 = figure();
    auto lines1 = plot(predY1, "-b", realY1, "-r");
    ylim({-1.5, 1.5});

    auto f2 = figure();
    auto lines2 = plot(predY2, "-b", realY2, "-r");
    ylim({-1.5, 1.5});

    auto f3 = figure();
    auto lines3 = plot(predY3, "-b", realY3, "-r");
    ylim({-1.5, 1.5});

    long t;
    while (true) {
        VectorXd input(1);
        input << sin(t);
        VectorXd prediction = math.predict(input);

        if (t < 120) {
            realY1.push_back(input(0));
            predY1.push_back(prediction(0));

            lines1[0]->y_data(predY1);
            lines1[1]->y_data(realY1);
        }
        if (t > 50000 && t < 50120) {
            realY2.push_back(input(0));
            predY2.push_back(prediction(0));

            lines2[0]->y_data(predY2);
            lines2[1]->y_data(realY2);
        }
        if (t > 500000 && t < 500120) {
            realY3.push_back(input(0));
            predY3.push_back(prediction(0));

            lines3[0]->y_data(predY3);
            lines3[1]->y_data(realY3);
        }


        t++;
    }
}
