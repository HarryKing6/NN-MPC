#include <torch/script.h>   // One-stop header.

#include <iostream>
#include <memory>

#include "config.h"
#include "types.h"
#include "Params/params.h"

namespace mpcc{

struct LinModelMatrix {
    A_MPC A;
    B_MPC B;
    g_MPC g;
};

struct ModelDerivatives{
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

    LinModelMatrix getLinModel(const State &x, const Input &u) const;
    
    nnModel();
    nnModel(double Ts const PathToJson &path);

private: 

    LinModelMatrix getModelJacobian(const State &x, const Input &u) const;
    LinModelMatrix discretizeModel(const LinModelMatrix &lin_model_c) const;

    Param param_;
    const double Ts_;
    torch::jit::script::Module module;
  
};




}

