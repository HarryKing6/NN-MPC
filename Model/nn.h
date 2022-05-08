#ifndef MPCC_NN_H
#define MPCC_NN_H


#include <torch/script.h>   // One-stop header.

#include <iostream>
#include <memory>

#include "config.h"
#include "types.h"
#include "Params/params.h"
#include "Model/model.h"

namespace mpcc{

struct NNDerivatives{
    const double dF1_vx;
    const double dF1_vy;
    const double dF1_r;
    const double dF1_D;
    const double dF1_delta;

    const double dF2_vx;
    const double dF2_vy;
    const double dF2_r;
    const double dF2_D;
    const double dF2_delta;

    const double dF3_vx;
    const double dF3_vy;
    const double dF3_r;
    const double dF3_D;
    const double dF3_delta;

};

class NN {
public:

    StateVector getF(const State &x,const Input &u);
    LinModelMatrix getLinModel(const State &x, const Input &u);
    std::vector<double> nnOutput(double vx, double vy, double r, double D, double delta);
    
    NN();
    NN(double Ts, const PathToJson &path);

private: 

    LinModelMatrix getModelJacobian(const State &x, const Input &u);
    LinModelMatrix discretizeModel(const LinModelMatrix &lin_model_c) const;
    std::vector<double> normalize(double vx, double vy, double r, double D, double delta) const;
    std::vector<double> denormalize(double dvx, double dvy, double dr) const;

    Param param_;
    const double Ts_;
    torch::Tensor input_tensor = torch::rand({1, 4, 5});
    torch::jit::Module module;

};




}


#endif //MPCC_NN_H

