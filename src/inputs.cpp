#import <Eigen/Core>
#import <string>

using namespace Eigen;
using namespace std;

#define UIV_LEN 4

namespace mantis::inputs {
    // simulates a single user (successfully) logging in between 8-9 AM and logging out between 5-6 PM
    VectorXd singleUserDayJob(long timeStep) {
        VectorXd input(3 + UIV_LEN);

        int hour;
        int minute;
        int action;
        int uid;

        if (timeStep % 2 == 0) { // logging in
            hour = 8;
            minute = 60 * ((double) random() / RAND_MAX);
            action = 0;
            uid = 1000;
        } else { // logging out
            hour = 17;
            minute = 60 * ((double) random() / RAND_MAX);
            action = 2;
            uid = 1000;
        }

        input(0) = hour/24.0;
        input(1) = minute/60.0;
        input(2) = action;
        for (int i = 1; i <= UIV_LEN; i++) {
            input(2+i) = sin(i * uid);
        }

        return input;
    }

    // simulates two users with offset, overlapping shifts
    VectorXd twoUserOverlap(long timeStep) {
        VectorXd input(3 + UIV_LEN);

        int hour;
        int minute;
        int action;
        int uid;

        if (timeStep % 4 == 0) { // 1000 logging in
            hour = 8;
            minute = 60 * ((double) random() / RAND_MAX);
            action = 0;
            uid = 1000;
        } else if (timeStep % 4 == 1) { // 1001 logging in
            hour = 13;
            minute = 60 * ((double) random() / RAND_MAX);
            action = 0;
            uid = 1001;
        } else if (timeStep % 4 == 2) { // 1000 logging out
            hour = 17;
            minute = 60 * ((double) random() / RAND_MAX);
            action = 2;
            uid = 1000;
        } else { // 1001 logging out
            hour = 22;
            minute = 60 * ((double) random() / RAND_MAX);
            action = 2;
            uid = 1001;
        }

        input(0) = hour/24.0;
        input(1) = minute/60.0;
        input(2) = action;
        for (int i = 1; i <= UIV_LEN; i++) {
            input(2+i) = sin(i * uid);
        }

        return input;
    }
}

namespace mantis::outputAdjusters {
    VectorXd userAuth(const VectorXd& output) {
        VectorXd out(output.size());

        out(0) = (output(0) + 1) / 2; // hours are 0 -> 1
        out(1) = (output(1) + 1) / 2; // minutes are 0 -> 1
        out(2) = round(output(2) + 1); // can be 0 -> 2, round to nearest integer
        for (int i = 0; i < UIV_LEN; i++) {
            out(3 + i) = output(3 + i); // UIV components are -1 -> 1
        }

        return out;
    }
}

namespace mantis::outputInterpreters {
    string userAuth(const VectorXd& output) {
        stringstream out;
        out << "("
            << round(output(0)*24) << ":" << (round(output(1)*60) < 10 ? "0" : "") << round(output(1)*60) << ", "
            << ((((int) output(2)) == 0) ? "LOG_IN_SUCCESS" : ((((int) output(2)) == 1) ? "LOG_IN_FAIL" : "LOG_OUT")) << ", ";

        for (int i = 0; i < UIV_LEN; i++) {
            out << round(output(3 + i) * 1000)/1000;
            if (i < UIV_LEN-1) out << ", ";
            else out << ")";
        }

        return out.str();
    }
}