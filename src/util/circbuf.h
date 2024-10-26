//
// Created by Malcolm Roalson on 9/17/24.
//

#ifndef MANTIS_CIRCBUF_H
#define MANTIS_CIRCBUF_H
namespace mantis::util {
    template<typename T>
    class circbuf {
        T *data;
        int idx = 0;
        unsigned int size = 0;

        bool overwriting = false;
        std::function<void(T)> free;

    public:
        circbuf(unsigned int size, std::function<void(T)> free) : free(free),
                                                         size(size), data(new T[size]) {}

        void push(T obj) {
            if (overwriting) free(data[idx]);
            else if (idx + 1 == size) overwriting = true;

            data[idx++] = obj;

            idx %= size;
        };

        T &operator[](int i) {
            return data[i];
        };

        T &operator[](int i) const {
            return data[i];
        };
    };
}

#endif //MANTIS_CIRCBUF_H
