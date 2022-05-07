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

  std::vector<double> input_vector = normalize(0, 0, 0, 0, 0);
  std::cout << input_vector << "\n";

  for(int i=0; i<4; i++){
    for(int j=0; j<5; j++){
      input_tensor[0][i][j] = input_vector[j];
    }
  }

}
StateVector NN::getF(const State &x,const Input &u){
  const double phi = x.phi;
  const double vx = x.vx;
  const double vy = x.vy;
  const double r  = x.r;
  const double D = x.D;
  const double delta = x.delta;
  const double vs = x.vs;

  const double dD = u.dD;
  const double dDelta = u.dDelta;
  const double dVs = u.dVs;

  std::vector<double> F = nnOutput(vx, vy, r, D, delta);

  StateVector f;
  f(0) = vx*std::cos(phi) - vy*std::sin(phi);
  f(1) = vy*std::cos(phi) + vx*std::sin(phi);
  f(2) = r;
  f(3) = F[0];
  f(4) = F[1];
  f(5) = F[2];
  f(6) = vs;
  f(7) = dD;
  f(8) = dDelta;
  f(9) = dVs;

  return f;
}

LinModelMatrix NN::getLinModel(const State &x, const Input &u){
    
  // compute linearized and discretized model
  const LinModelMatrix lin_model_c = getModelJacobian(x,u);
  // discretize the system
  return lin_model_c;
}

LinModelMatrix NN::getModelJacobian(const State &x, const Input &u)
{
  const double phi = x.phi;
  const double vx = x.vx;
  const double vy = x.vy;
  const double r  = x.r;
  const double D = x.D;
  const double delta = x.delta;

  // LinModelMatrix lin_model_c;
  A_MPC A_c = A_MPC::Zero();
  B_MPC B_c = B_MPC::Zero();
  g_MPC g_c = g_MPC::Zero();

  const double d1 = 0.001;
  const double d2 = 0.005;
  // const double d3 = 0.01;

  std::vector<double> F0 = nnOutput(vx, vy, r, D, delta);
  std::vector<double> F1_1 = nnOutput(vx+d1, vy, r, D, delta);
  std::vector<double> F1_2 = nnOutput(vx+d2, vy, r, D, delta);
  std::vector<double> F2_1 = nnOutput(vx, vy+d1, r, D, delta);
  std::vector<double> F2_2 = nnOutput(vx, vy+d2, r, D, delta);
  std::vector<double> F3_1 = nnOutput(vx, vy, r+d1, D, delta);
  std::vector<double> F3_2 = nnOutput(vx, vy, r+d2, D, delta);
  std::vector<double> F4_1 = nnOutput(vx, vy, r, D+d1, delta);
  std::vector<double> F4_2 = nnOutput(vx, vy, r, D+d2, delta);
  std::vector<double> F5_1 = nnOutput(vx, vy, r, D, delta+d1);
  std::vector<double> F5_2 = nnOutput(vx, vy, r, D, delta+d2);


  // vx
  const double dvx_vx = 
  0.5/d1*(F1_1[0] - F0[0]) + 0.5/d2*(F1_2[0]-F0[0]); 
  const double dvy_vx = 
  0.5/d1*(F1_1[1] - F0[1]) + 0.5/d2*(F1_2[1]-F0[1]);
  const double dr_vx = 
  0.5/d1*(F1_1[2] - F0[2]) + 0.5/d2*(F1_2[2]-F0[2]);  
  // vy
  const double dvx_vy = 
  0.5/d1*(F2_1[0] - F0[0]) + 0.5/d2*(F2_2[0]-F0[0]); 
  const double dvy_vy = 
  0.5/d1*(F2_1[1] - F0[1]) + 0.5/d2*(F2_2[1]-F0[1]);
  const double dr_vy = 
  0.5/d1*(F2_1[2] - F0[2]) + 0.5/d2*(F2_2[2]-F0[2]);  
  // r
  const double dvx_r = 
  0.5/d1*(F3_1[0] - F0[0]) + 0.5/d2*(F3_2[0]-F0[0]); 
  const double dvy_r = 
  0.5/d1*(F3_1[1] - F0[1]) + 0.5/d2*(F3_2[1]-F0[1]);
  const double dr_r = 
  0.5/d1*(F3_1[2] - F0[2]) + 0.5/d2*(F3_2[2]-F0[2]);  
  // D
  const double dvx_D = 
  0.5/d1*(F4_1[0] - F0[0]) + 0.5/d2*(F4_2[0]-F0[0]); 
  const double dvy_D = 
  0.5/d1*(F4_1[1] - F0[1]) + 0.5/d2*(F4_2[1]-F0[1]);
  const double dr_D = 
  0.5/d1*(F4_1[2] - F0[2]) + 0.5/d2*(F4_2[2]-F0[2]);
  // delta
  const double dvx_delta = 
  0.5/d1*(F5_1[0] - F0[0]) + 0.5/d2*(F5_2[0]-F0[0]); 
  const double dvy_delta = 
  0.5/d1*(F5_1[1] - F0[1]) + 0.5/d2*(F5_2[1]-F0[1]);
  const double dr_delta = 
  0.5/d1*(F5_1[2] - F0[2]) + 0.5/d2*(F5_2[2]-F0[2]);    
  
  // Jacobians: A Matrix  
  // row1: dX
  A_c(0,2) = -vx*std::sin(phi) - vy*std::cos(phi);
  A_c(0,3) = std::cos(phi);
  A_c(0,4) = -std::sin(phi);
  // row2: dY 
  A_c(1,2) = -vy*std::sin(phi) + vx*std::cos(phi);
  A_c(1,3) = std::sin(phi);
  A_c(1,3) = std::cos(phi);
  // row3: dphi 
  A_c(2,5) = 1.0;
  // row4: dvx
  A_c(3,3) = dvx_vx;
  A_c(3,4) = dvx_vy;
  A_c(3,5) = dvx_r;
  A_c(3,7) = dvx_D;
  A_c(3,8) = dvx_delta;
  // row5: dvy
  A_c(4,3) = dvy_vx;
  A_c(4,4) = dvy_vy;
  A_c(4,5) = dvy_r;
  A_c(4,7) = dvy_D;
  A_c(4,8) = dvy_delta;
  // row6: dr
  A_c(5,3) = dr_vx;
  A_c(5,4) = dr_vy;
  A_c(5,5) = dr_r;
  A_c(5,7) = dr_D;
  A_c(5,8) = dr_delta;
  // row7: ds 
  // all zero
  // row8: dD
  // all zero 
  // row 9: ddelta 
  // all zero 
  // row 10: dvs
  // all zero

  // Jacobians: B Matrix 
    B_c(7,0) = 1.0;
    B_c(8,1) = 1.0;
    B_c(9,2) = 1.0;

  // Jacobians: C Matrix
  const StateVector f = getF(x,u);
  g_c = f - A_c*stateToVector(x) - B_c*inputToVector(u);

  return {A_c,B_c,g_c};
}

std::vector<double> NN::nnOutput(double vx, double vy, double r, double D, double delta)
{

  std::vector<double> input_vector = normalize(vx, vy, r, D, delta);

  for(int i=0; i<3; i++){
    for(int j=0; j<5; j++){
      input_tensor[0][i][j] = input_tensor[0][i+1][j];
    }
  }
  for(int j=0; j<5; j++){
    input_tensor[0][4][j] = input_vector[j]; 
  }

  
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor);

  at::Tensor output = module.forward(inputs).toTensor(); 
  std::vector<double> output_v(output.data_ptr<double>(), output.data_ptr<double>() + output.numel());
  
  return denormalize(output_v[0],output_v[1],output_v[2],output_v[3],output_v[4]);
}

std::vector<double> NN::normalize(double vx, double vy, double r, double D, double delta) const{
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

std::vector<double> NN::denormalize(double vx, double vy, double r, double D, double delta) const{
  std::vector<double> denorm(5);
  vx = vx * (param_.vx_max-param_.vx_min) + param_.vx_min;
  denorm.push_back(vx);
  vy = vy * (param_.vy_max-param_.vy_min) + param_.vy_min;
  denorm.push_back(vy);
  r = r * (param_.r_max-param_.r_min) + param_.r_min;
  denorm.push_back(r);
  D = D * (param_.D_max-param_.D_min) + param_.D_min;
  denorm.push_back(D);
  delta = delta * (param_.delta_max-param_.delta_min) + param_.delta_min;
  denorm.push_back(delta);

  return denorm;
}

}

