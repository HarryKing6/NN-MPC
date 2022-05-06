#include "nn.h"

namespace mpcc{
NN::NN()
:Ts_(1.0)
{
  std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

NN::NN(double Ts,const PathToJson &path)
:Ts_(Ts),param_(Param(path.param_path))
{
  module = torch::jit::load("traced_model.pt");  
}


// LinModelMatrix NN::getLinModel(const State &x, const Input &u) const{
    
//   // compute linearized and discretized model
//   const LinModelMatrix lin_model_c = getModelJacobian(x,u);
//   // discretize the system
//   return lin_model_c;
// }

// LinModelMatrix NN::getModelJacobian(const State &x, const Input &u) const
// {

// }

// LinModelMatrix NN::nnOutput(const State &x, const Input &u) const
// {
//   double vx = x.vx;
//   double vy = x.vy;
//   double r = x.r;
//   double D = x.D+u.dD;
//   double delta = x.delta + u.dDelta;
//   std::vector<double> new = normalize(vx, vy, r, D, delta);

//   for(int i=0; i<4; i++){
//     for(int j=0; j<5; j++){
//       input[i][j] = input[i+1][j];
//     }
//   }

// }

std::vector<double> NN::normalize(double vx, double vy, double r, double D, double delta){
  std::vector<double> norm(5);
  vx = (vx - param_.vx_min) / (param_.vx_max - param_.vx_min);
  norm.push_back(vx);
  vy = (vy - param_.vx_min) / (param_.vy_max - param_.vy_min);
  norm.push_back(vy);
  r = (r - param_.r_min) / (param_.r_max - param_.r_min);
  norm.push_back(r);
  D = (D - param_.D_min) / (param_.D_max - param_.D_min);
  norm.push_back(D);
  delta = (delta - param_.delta_min) / (param_.delta_max - param_.delta_min);
  norm.push_back(delta);

  return norm;
}



}

