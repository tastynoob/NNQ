#include "qList.hpp"

namespace nnq {


    qList::qList() {
        data = nullptr;
        size = 0;
    }
    qList::qList(const std::initializer_list<qtype>& list) {
        size = list.size();
        data = qalloc(qtype, size);
        for (qint i = 0; i < size; i++) {
            data[i] = list.begin()[i];
        }
    }
    qList::qList(const qList& list) {
        size = list.size;
        data = qalloc(qtype, size);
        for (qint i = 0; i < size; i++) {
            data[i] = list.data[i];
        }
    }

    qList::qList(qint size) {
        data = qalloc(qtype, size);
        this->size = size;
    }
    qList::qList(qList& list) {
        data = qalloc(qtype, list.size);
        this->size = list.size;
        for (qint i = 0; i < list.size; i++) {
            data[i] = list.data[i];
        }
    }
    qList::~qList() {
        qfree(data);
        size = 0;
    }

    qList& qList::operator=(qList& list) {
        if (this != &list) {
            qfree(data);
            data = qalloc(qtype, list.size);
            this->size = list.size;
            for (qint i = 0; i < list.size; i++) {
                data[i] = list.data[i];
            }
        }
        return *this;
    }
    qtype& qList::operator[](qint index) const {
        return data[index];
    }

    qList& qList::operator+=(qtype& value) {
        qtype* temp = qalloc(qtype, size + 1);
        for (qint i = 0; i < size; i++) {
            temp[i] = data[i];
        }
        temp[size] = value;
        qfree(data);
        data = temp;
        size++;
        return *this;
    }







}


