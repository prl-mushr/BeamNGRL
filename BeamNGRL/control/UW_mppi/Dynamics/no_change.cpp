#include <stdio.h>
#define x_index 0
#define y_index 1
#define z_index 2
#define roll_index 3
#define pitch_index 4
#define yaw_index 5
#define vx_index 6
#define vy_index 7
#define vz_index 8
#define ax_index 9
#define ay_index 10
#define az_index 11
#define wx_index 12
#define wy_index 13
#define wz_index 14
#define st_index 0
#define th_index 1
#define GRAVITY 9.8f

__global__ void rollout(float* state, const float* controls, const float* BEVmap_height, const float* BEVmap_normal, const float dt, const int rollouts, const int timesteps, const int NX, const int NC,
                        const float D, const float B, const float C, const float lf, const float lr, const float Iz, const float throttle_to_wheelspeed, const float steering_max,
                        const int BEVmap_size_px, const float BEVmap_res, const float BEVmap_size, float car_l2, const float car_w2, const float cg_height, 
                        const float LPF_tau_rpy, const float LPF_tau_st, const float LPF_tau_th, const float res_coeff, const float drag_coeff)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int state_index = k*timesteps*NX;
    int next;

    // copy initial state to all timesteps:
    for(int t = 0; t < timesteps-1; t++)
    {
        next = (t + 1)*NX + state_index;

        state[next + x_index] = state[state_index + x_index];
        state[next + y_index] = state[state_index + y_index];
        state[next + z_index] = state[state_index + z_index];
        state[next + roll_index] = state[state_index + roll_index];
        state[next + pitch_index] = state[state_index + pitch_index];
        state[next + yaw_index] = state[state_index + yaw_index];
        state[next + vx_index] = state[state_index + vx_index];
        state[next + vy_index] = state[state_index + vy_index];
        state[next + vz_index] = state[state_index + vz_index];
        state[next + ax_index] = state[state_index + ax_index];
        state[next + ay_index] = state[state_index + ay_index];
        state[next + az_index] = state[state_index + az_index];
        state[next + wx_index] = state[state_index + wx_index];
        state[next + wy_index] = state[state_index + wy_index];
        state[next + wz_index] = state[state_index + wz_index];
    }
}