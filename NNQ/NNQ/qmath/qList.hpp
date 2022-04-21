#pragma once


#include "../config.hpp"



namespace nnq {

class qList {
public:
    qtype* data=nullptr;
    qint size=0;
    qList();
    qList(const std::initializer_list<qtype>& list);
    qList(const qList& list);
    qList(qint size);
    qList(qList& list);
    ~qList();
    qList& operator=(qList& list);
    qtype& operator[](qint index) const;
    qList& operator+=(qtype& value);
};

}


