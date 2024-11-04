// my_extension.cpp
#include <torch/extension.h>

// Function declarations
void slip3d_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, float dt, int rollouts, int timesteps, int NX, int NC,
                      float D, float B, float C, float lf, float lr, float Iz, float throttle_to_wheelspeed, float steering_max,
                      int BEVmap_size_px, float BEVmap_res, float BEVmap_size, float car_l2, float car_w2, float cg_height, 
                      float LPF_tau_rpy, float LPF_tau_st, float LPF_tau_th, float res_coeff, float drag_coeff, int block_dim, int grid_dim);

void noslip3d_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, float dt, int rollouts, int timesteps, int NX, int NC,
                      float D, float B, float C, float lf, float lr, float Iz, float throttle_to_wheelspeed, float steering_max,
                      int BEVmap_size_px, float BEVmap_res, float BEVmap_size, float car_l2, float car_w2, float cg_height, 
                      float LPF_tau_rpy, float LPF_tau_st, float LPF_tau_th, float res_coeff, float drag_coeff, int block_dim, int grid_dim);

void preprocess_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor sa, torch::Tensor bev_context, int step,
                         torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, 
                         int BEVmap_size_px, float BEVmap_res, float BEVmap_size,
                         int rollouts, int timesteps, int NX, int NC, float car_l2, 
                         float car_w2, torch::Tensor std_state, torch::Tensor mean_state, float patch_size,
                         int block_dim, int grid_dim);

void euler_step_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor Ddot_q, int step,
                         torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, 
                         int BEVmap_size_px, float BEVmap_res, float BEVmap_size,
                         float dt, int rollouts, int timesteps, int NX, int NC, float car_l2, 
                         float car_w2, torch::Tensor std_state, torch::Tensor mean_state,
                         int block_dim, int grid_dim);

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rollout_slip3d", &slip3d_launcher, "slip3d rollout");
    m.def("rollout_noslip3d", &noslip3d_launcher, "noslip3d rollout");
    m.def("rollout_preprocess", &preprocess_launcher, "preprocess");
    m.def("rollout_euler_step", &euler_step_launcher, "euler_step");
}