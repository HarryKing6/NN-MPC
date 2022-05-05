#include <torch/script.h>   // One-stop header.

#include <iostream>
#include <memory>

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

