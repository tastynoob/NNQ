#pragma once
#include "config.hpp"



namespace nnq {


    class qComplex {
    public:
        qtype real, imag;
        qComplex();
        qComplex(qtype re);
        qComplex(const qComplex& c);
        qComplex(qtype re, qtype im);
        qComplex& operator=(qComplex& c);
        qComplex operator+(qComplex& c);
        qComplex operator-(qComplex& c);
        qComplex operator*(qComplex& c);
        qComplex operator/(qComplex& c);
        qComplex& operator+=(qComplex& c);
        qComplex& operator-=(qComplex& c);
        qComplex& operator*=(qComplex& c);
        qComplex& operator/=(qComplex& c);
        qComplex operator+(qtype& c);
        qComplex operator-(qtype& c);
        qComplex operator*(const qtype& c);
        qComplex operator/(qtype& c);
    };















} // namespace nnq

