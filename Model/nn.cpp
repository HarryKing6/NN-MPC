#include "nn.h"

namespace mpcc{
NN::nnModel()
:Ts_(1.0)
{
  std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

NN::nnModel(double Ts,const PathToJson &path)
:Ts_(Ts),param_(Param(path.param_path))
{
  module = torch::jit::load("traced_model.pt");
}


LinModelMatrix NN::getLinModel(const State &x, const Input &u) const{
    
    // compute linearized and discretized model
    const LinModelMatrix lin_model_c = getModelJacobian(x,u);
    // discretize the system
    return discretizeModel(lin_model_c);
}

LinModelMatrix NN::getModelJacobian(const State &x, const Input &u) const
{
    

}



}






int main(int argc, const char* argv[]) {
  
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("/home/alexzheng/python/traced_model.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "ok\n";


  std::vector<torch::jit::IValue> inputs; 
  inputs.push_back(torch::zeros({4,5}));

  at::Tensor output = module.forward(inputs).toTensor(); 
  std::cout << output.slice() << std::endl;

}

