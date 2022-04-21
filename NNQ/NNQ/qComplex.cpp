#include "qComplex.hpp"



namespace nnq {



qComplex::qComplex(): real(0), imag(0) {}
qComplex::qComplex(qtype re) : real(re), imag(0) {}
qComplex::qComplex(const qComplex& c) : real(c.real), imag(c.imag) {}
qComplex::qComplex(qtype re, qtype im) : real(re), imag(im) {}
qComplex& qComplex::operator=(qComplex& c) {
    real = c.real;
    imag = c.imag;
    return *this;
}
 qComplex qComplex::operator+(qComplex& c) {
    return qComplex(real + c.real, imag + c.imag);
}
 qComplex qComplex::operator-(qComplex& c) {
    return qComplex(real - c.real, imag - c.imag);
}
 qComplex qComplex::operator*(qComplex& c) {
    return qComplex(real * c.real - imag * c.imag, real * c.imag + imag * c.real);
}
 qComplex qComplex::operator/(qComplex& c) {
    qtype denom = c.real * c.real + c.imag * c.imag;
    return qComplex((real * c.real + imag * c.imag) / denom, (imag * c.real - real * c.imag) / denom);
}

 qComplex& qComplex::operator+=(qComplex& c) {
    real += c.real;
    imag += c.imag;
    return *this;
}
 qComplex& qComplex::operator-=(qComplex& c) {
    real -= c.real;
    imag -= c.imag;
    return *this;
}
 qComplex& qComplex::operator*=(qComplex& c) {
    qtype re_ = real * c.real - imag * c.imag;
    qtype im_ = real * c.imag + imag * c.real;
    real = re_;
    imag = im_;
    return *this;
}
 qComplex& qComplex::operator/=(qComplex& c) {
    qtype denom = c.real * c.real + c.imag * c.imag;
    qtype re_ = (real * c.real + imag * c.imag) / denom;
    qtype im_ = (imag * c.real - real * c.imag) / denom;
    real = re_;
    imag = im_;
    return *this;
}
 qComplex qComplex::operator+(qtype& c) {
    return qComplex(real + c, imag);
}
 qComplex qComplex::operator-(qtype& c) {
    return qComplex(real - c, imag);
}
 qComplex qComplex::operator*(const qtype& c) {
    return qComplex(real * c, imag * c);
}
 qComplex qComplex::operator/(qtype& c) {
    return qComplex(real / c, imag / c);
}


















} // namespace nnq

