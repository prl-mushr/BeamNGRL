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
// very generous limits on acceleration and velocity:

__device__ float map_to_elev(const float x, const float y, const float* elev, const int map_size_px, const float res_inv)
{
    int img_X = fminf(fmaxf((int)((x*res_inv) + map_size_px/2), 0), map_size_px - 1);
    int img_Y = fminf(fmaxf((int)((y*res_inv) + map_size_px/2), 0), map_size_px - 1);

    return elev[img_Y * map_size_px + img_X];
}

__device__ void get_footprint_z(float* fl, float* fr, float* bl, float* br, float& z, 
                                const float x, const float y, const float cy, const float sy, 
                                const float* elev, const float map_size_px, const float res_inv, 
                                const float car_l2, const float car_w2)
{
    fl[0] = car_l2*cy - car_w2*sy + x;
    fl[1] = car_l2*sy + car_w2*cy + y;

    fr[0] = car_l2*cy - (-1)*car_w2*sy + x;
    fr[1] = car_l2*sy + (-1)*car_w2*cy + y;
    
    bl[0] = (-1)*car_l2*cy - car_w2*sy + x;
    bl[1] = (-1)*car_l2*sy + car_w2*cy + y;
    
    br[0] = (-1)*car_l2*cy - (-1)*car_w2*sy + x;
    br[1] = (-1)*car_l2*sy + (-1)*car_w2*cy + y;

    z = map_to_elev(x, y, elev, map_size_px, res_inv);
    fl[2] = map_to_elev(fl[0], fl[1], elev, map_size_px, res_inv);
    fr[2] = map_to_elev(fr[0], fr[1], elev, map_size_px, res_inv);
    bl[2] = map_to_elev(bl[0], bl[1], elev, map_size_px, res_inv);
    br[2] = map_to_elev(br[0], br[1], elev, map_size_px, res_inv);
}

__global__ void forward_rollout(float* state, const float* BEVmap_height, const float dt, const int rollouts, const int timesteps, const int NX,
                                const int BEVmap_size_px, const float BEVmap_res, const float BEVmap_size, float car_l2, const float car_w2) {
    
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k > rollouts)
    {
        return;
    }
    int state_index = k*timesteps*NX;

    int curr, next;

    float x=0, y=0, z=0, roll=0, pitch=0, yaw=0, vx=0, vy=0, vz=0, last_vx=0, last_vy=0, last_vz=0, ax=0, ay=0, az=0, wx=0, wy=0, wz=0;
    float fl[3], fr[3], bl[3], br[3];
    float cp, sp, cr, sr, cy, sy;
    float dt_inv = 1/dt;
    float res_inv = 1.0f/BEVmap_res, dummy;

    __syncthreads();

    x       = state[state_index + x_index];
    y       = state[state_index + y_index];
    z       = state[state_index + z_index];
    last_vx = state[state_index + vx_index];
    last_vy = state[state_index + vy_index];
    last_vz = state[state_index + vz_index];
    roll    = state[state_index + roll_index];
    pitch   = state[state_index + pitch_index];
    yaw     = state[state_index + yaw_index];
    cy = cosf(yaw);
    sy = sinf(yaw);

    for (int t = 0; t < timesteps-1; t++) 
    {
        curr = t*NX + state_index;
        next = (t + 1)*NX + state_index;

        // I'm doing this mostly for readability.
        vx = state[curr + vx_index];
        vy = state[curr + vy_index];
        vz = state[curr + vz_index];
        wx = state[curr + wx_index];
        wy = state[curr + wy_index];
        wz = state[curr + wz_index];

        get_footprint_z(fl, fr, bl, br, dummy, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);

        roll  = atan2f( (fl[2] + bl[2]) - (fr[2] + br[2]), 4*car_w2);
        pitch = atan2f( (bl[2] + br[2]) - (fl[2] + fr[2]), 4*car_l2);
        yaw  += dt * (wy * (sr / cp) + wz * (cr / cp));

        cr = cosf(roll);
        sr = sinf(roll);
        cp = cosf(pitch);
        sp = sinf(pitch);
        cy = cosf(yaw);
        sy = sinf(yaw);

        ax = (vx - last_vx)*dt_inv  - (vy*wz - vz*wy + sp*GRAVITY);
        ay = (vy - last_vy)*dt_inv  - (-vx*wz + vz*wx - sr*cp*GRAVITY);
        az = (vz - last_vz)*dt_inv  - (vx*wy - vy*wx - cp*cr*GRAVITY);

        last_vx = vx;
        last_vy = vy;
        last_vz = vz;

        x += dt * (vx * cp * cy + vy * (sr * sp * cy - cr * sy) + vz * (cr * sp * cy + sr * sy));
        y += dt * (vx * cp * sy + vy * (sr * sp * sy + cr * cy) + vz * (cr * sp * sy - sr * cy));
        z += dt * (vx * (-sp)   + vy * (sr * cp)                + vz * (cr * cp));
    
        state[next + x_index]     = x;
        state[next + y_index]     = y;
        state[next + z_index]     = z;
        state[next + roll_index]  = roll;
        state[next + pitch_index] = pitch;
        state[next + yaw_index]   = yaw;
        state[next + ax_index]    = ax;
        state[next + ay_index]    = ay;
        state[next + az_index]    = az;
    }
}
