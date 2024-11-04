// my_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
#define GRAVITY 9.81f
// very generous limits on acceleration and velocity:
#define max_vel 40.0f
#define max_acc 50.0f

__device__ float nan_to_num(float x, float replace)
{
    if (std::isnan(x) or std::isinf(x)) 
    {
        return replace;
    }
    return x;
}

__device__ float clamp(float x, float lower, float upper)
{
    return fminf(fmaxf(x, lower), upper);
}

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

    float z_cent = map_to_elev(0, 0, elev, map_size_px, res_inv);
    z = map_to_elev(x, y, elev, map_size_px, res_inv) - z_cent;
    fl[2] = map_to_elev(fl[0], fl[1], elev, map_size_px, res_inv) - z_cent;
    fr[2] = map_to_elev(fr[0], fr[1], elev, map_size_px, res_inv) - z_cent;
    bl[2] = map_to_elev(bl[0], bl[1], elev, map_size_px, res_inv) - z_cent;
    br[2] = map_to_elev(br[0], br[1], elev, map_size_px, res_inv) - z_cent;
}

__global__ void slip3d(float* state, const float* controls, const float* BEVmap_height, const float* BEVmap_normal, const float dt, const int rollouts, const int timesteps, const int NX, const int NC,
                        const float D, const float B, const float C, const float lf, const float lr, const float Iz, const float throttle_to_wheelspeed, const float steering_max,
                        const int BEVmap_size_px, const float BEVmap_res, const float BEVmap_size, const float car_l2, const float car_w2, const float cg_height, 
                        const float LPF_tau_rpy, const float LPF_tau_st, const float LPF_tau_th, const float res_coeff, const float drag_coeff) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k > rollouts)
    {
        return;
    }
    int state_index = k*timesteps*NX;
    int control_index = k*timesteps*NC;

    int curr, next, ctrl_base;

    float x=0, y=0, z=0, roll=0, pitch=0, last_roll=0, last_pitch=0, last_roll_rate=0, last_pitch_rate = 0, yaw=0, vx=0, vy=0, vz=0, ax=0, ay=0, az=0, wx=0, wy=0, wz=0;
    float st, w;

    float vf, vfy, vr, Kr, Kf, alphaf, alphar, alpha_z, sigmaf, sigmar, sigmaf_x, sigmaf_y, sigmar_x, sigmar_y, Fr, Ff, Frx, Fry, Ffx, Ffy;
    float roll_rate=0, pitch_rate=0, yaw_rate=0;
    float cp, sp, cr, sr, cy, sy;
    float fl[3], fr[3], bl[3], br[3];
    float res_inv = 1.0f/BEVmap_res;
    float Nf, Nr;

    __syncthreads();
    // 0th control is copied as is
    st = controls[st_index] * steering_max;
    w = controls[th_index] * throttle_to_wheelspeed;


    last_roll_rate = state[state_index + wx_index];
    last_pitch_rate = state[state_index + wy_index];

    for(int t = 0; t < timesteps-1; t++)
    {
        curr = t*NX + state_index;
        next = (t + 1)*NX + state_index;

        ctrl_base = t*NC + control_index;
        
        st = LPF_tau_st*controls[ctrl_base + st_index] * steering_max + (1 - LPF_tau_st) * st;
        w = LPF_tau_th*controls[ctrl_base + th_index] * throttle_to_wheelspeed + (1 - LPF_tau_th) * w;

        x = state[curr + x_index];
        y = state[curr + y_index];

        vx = state[curr + vx_index];
        vy = state[curr + vy_index];
        vz = 0;

        wx = state[curr + wx_index];
        wy = state[curr + wy_index];
        wz = state[curr + wz_index];

        last_roll = state[curr + roll_index];
        last_pitch = state[curr + pitch_index];

        yaw = state[curr + yaw_index];
        
        cy = cosf(yaw);
        sy = sinf(yaw);

        get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);

        roll = (atan2f( (fl[2] + bl[2]) - (fr[2] + br[2]),  4*car_w2))*LPF_tau_rpy + last_roll*(1 - LPF_tau_rpy);
        pitch = (atan2f( (bl[2] + br[2]) - (fl[2] + fr[2]), 4*car_l2))*LPF_tau_rpy + last_pitch*(1 - LPF_tau_rpy);

        roll_rate = (LPF_tau_rpy*(roll - last_roll)/dt) + (1 - LPF_tau_rpy)*last_roll_rate;
        pitch_rate = (LPF_tau_rpy*(pitch - last_pitch)/dt) + (1 - LPF_tau_rpy)*last_pitch_rate;

        last_roll_rate = roll_rate;
        last_pitch_rate = pitch_rate;

        cp = cosf(pitch);
        sp = sinf(pitch);
        cr = cosf(roll);
        sr = sinf(roll);

        wx = roll_rate - sp*yaw_rate;
        wy = cp*sr*yaw_rate + cr*pitch_rate;

        vf = (vx * cosf(st) + vy * sinf(st));
        vfy = (vy * cosf(st) - vx * sinf(st));
        vr = vx;

        Kr = (w - vr) / vr;
        Kf = (w - vf) / vf;

        alphaf = st - atan2f(wz * lf + vy, vx);
        alphar = atan2f(wz * lr - vy, vx);

        sigmaf_x = nan_to_num( Kf / (1 + Kf), 0.01);
        sigmaf_y = nan_to_num( tanf(alphaf) / (1 + Kf), 0.01);
        sigmaf = fmaxf(sqrtf(sigmaf_x * sigmaf_x + sigmaf_y * sigmaf_y), 0.0001);

        sigmar_x = nan_to_num( Kr / (1 + Kr), 0.01);
        sigmar_y = nan_to_num( tanf(alphar) / (1 + Kr), 0.01);
        sigmar = fmaxf(sqrtf(sigmar_x * sigmar_x + sigmar_y * sigmar_y), 0.0001);

        Nf = (az*lf - ax*cg_height)/(lf + lr);
        Nr = (az*lr + ax*cg_height)/(lf + lr);

        Fr = Nr * D * sinf(C * atanf(B * sigmar));
        Ff = Nf * D * sinf(C * atanf(B * sigmaf));

        Frx = (Fr * sigmar_x / sigmar) - res_coeff*vr - drag_coeff*vr*fabsf(vr);
        Fry = (Fr * sigmar_y / sigmar) - drag_coeff*vy*fabsf(vy);
        Ffx = (Ff * sigmaf_x / sigmaf) - res_coeff*vf - drag_coeff*vf*fabsf(vf) ;
        Ffy = (Ff * sigmaf_y / sigmaf) - drag_coeff*vfy*fabsf(vfy);

        ax = Frx + Ffx * cosf(st) - Ffy * sinf(st) + sp*GRAVITY;
        // ax = clamp(ax, -max_acc, max_acc);
        ay = Fry + Ffy * cosf(st) + Ffx * sinf(st) + sr*cp*GRAVITY;
        // ay = clamp(ay, -max_acc, max_acc);
        az = GRAVITY*cr*cp - vx*wy + vy*wx; // don't integrate this acceleration
        // az = clamp(az, -max_acc, max_acc);
        alpha_z = (Ffx * sinf(st) * lf + Ffy * lf * cosf(st) - Fry * lr) / Iz;

        vx += (ax + vy*wz - sp*GRAVITY) * dt;
        // vx = clamp(vx, -max_vel, max_vel);
        vy += (ay - vx*wz - sr*cp*GRAVITY) * dt;
        // vy = clamp(vy, -max_vel, max_vel);
        wz += alpha_z * dt;

        yaw_rate = wy*(sr/cp) + wz*(cr/cp);

        yaw += yaw_rate*dt;
        // updated cy sy
        cy = cosf(yaw);
        sy = sinf(yaw);

        x += dt * ( vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) + vz * (cr * sp * cy + sr * sy) );
        y += dt * ( vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) + vz * (cr * sp * sy - sr * cy) );

        state[next + x_index] = x;
        state[next + y_index] = y;
        state[next + z_index] = z; // not really updated
        state[next + roll_index] = roll;
        state[next + pitch_index] = pitch;
        state[next + yaw_index] = yaw;
        state[next + vx_index] = vx;
        state[next + vy_index] = vy;
        state[next + vz_index] = vz;
        state[next + ax_index] = ax;
        state[next + ay_index] = ay;
        state[next + az_index] = az;
        state[next + wx_index] = wx;
        state[next + wy_index] = wy;
        state[next + wz_index] = wz;
    }
}

__global__ void noslip3d(float* state, const float* controls, const float* BEVmap_height, const float* BEVmap_normal, const float dt, const int rollouts, const int timesteps, const int NX, const int NC,
                        const float D, const float B, const float C, const float lf, const float lr, const float Iz, const float throttle_to_wheelspeed, const float steering_max,
                        const int BEVmap_size_px, const float BEVmap_res, const float BEVmap_size, const float car_l2, const float car_w2, const float cg_height, 
                        const float LPF_tau_rpy, const float LPF_tau_st, const float LPF_tau_th, const float res_coeff, const float drag_coeff) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k > rollouts)
    {
        return;
    }
    int state_index = k*timesteps*NX;
    int control_index = k*timesteps*NC;

    int curr, next, ctrl_base;

    float x, y, z, roll, pitch, last_roll, last_pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz;
    float st, w;
    float cp, sp, cr, sr, cy, sy, ct;
    float fl[3], fr[3], bl[3], br[3];
    float res_inv = 1.0f/BEVmap_res;
    float last_vx = 0.0f;
    __syncthreads();

    for(int t = 0; t < timesteps-1; t++)
    {
        curr = t*NX + state_index;
        next = (t + 1)*NX + state_index;

        ctrl_base = t*NC + control_index;
        
        st = controls[ctrl_base + st_index] * steering_max;
        w = controls[ctrl_base + th_index] * throttle_to_wheelspeed;

        float K = tanf(st)/(lf+lr);

        x = state[curr + x_index];
        y = state[curr + y_index];

        last_vx = state[curr + vx_index];
        vy = 0;
        vz = 0;

        wx = 0;
        wy = 0;
        wz = state[curr + wz_index];

        last_roll = state[curr + roll_index];
        last_pitch = state[curr + pitch_index];

        yaw = state[curr + yaw_index];
        
        cy = cosf(yaw);
        sy = sinf(yaw);

        get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);

        roll = atan2f( (fl[2] + bl[2]) - (fr[2] + br[2]),  4*car_w2);
        pitch = atan2f( (bl[2] + br[2]) - (fl[2] + fr[2]), 4*car_l2);

        wx = (roll - last_roll)/dt;
        wy = (pitch - last_pitch)/dt;

        cp = cosf(pitch);
        sp = sinf(pitch);
        cr = cosf(roll);
        sr = sinf(roll);
        ct = nan_to_num(sqrtf(1 - (sp*sp) - (sr*sr)), 0.0); // if roll and pitch are super large at the same time this can go nan.

        ax = nan_to_num((vx - last_vx) + sp*GRAVITY, 0.0);
        ay = nan_to_num((vx*wz) + sr*GRAVITY, 0.0);
        az = nan_to_num(GRAVITY*ct - vx*wy + vy*wx, GRAVITY); // don't integrate this acceleration

        vx = w;
        wz = K*vx;
        yaw += wz*dt;
        // updated cy sy
        cy = cosf(yaw);
        sy = sinf(yaw);

        x += dt * ( vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) + vz * (cr * sp * cy + sr * sy) );
        y += dt * ( vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) + vz * (cr * sp * sy - sr * cy) );

        state[next + x_index] = x;
        state[next + y_index] = y;
        state[next + z_index] = z; // not really updated
        state[next + roll_index] = roll;
        state[next + pitch_index] = pitch;
        state[next + yaw_index] = yaw;
        state[next + vx_index] = vx;
        state[next + vy_index] = vy;
        state[next + vz_index] = vz;
        state[next + ax_index] = ax;
        state[next + ay_index] = ay;
        state[next + az_index] = az;
        state[next + wx_index] = wx;
        state[next + wy_index] = wy;
        state[next + wz_index] = wz;
    }
}


__device__ void get_crop(float* bev_context, const float x, const float y, const float cy, const float sy, 
                        const float* elev, const int map_size_px, const float res, const float res_inv,
                        const float car_l2, const float car_w2, const float patch_size, int rollouts, int k)
{
    int patch_size_px = int(patch_size * res_inv), img_X, img_Y;
    float px, py, offset_x, offset_y;
    float center_height = map_to_elev(x, y, elev, map_size_px, res_inv);
    float std_inv = 2/patch_size;

    for(int i=0; i < patch_size_px; i++)
    {
        offset_x = (i*res) - (patch_size*0.5);
        for(int j=0; j < patch_size_px; j++)
        {
            offset_y = (j*res) - (patch_size*0.5);
            px = offset_x*cy - offset_y*sy + x;
            py = offset_x*sy + offset_y*cy + y;
            bev_context[k * (patch_size_px * patch_size_px) + j * patch_size_px + i] = (map_to_elev(px, py, elev, map_size_px, res_inv) - center_height)*std_inv;
        }
    }

}

__global__ void preprocess( float* state, const float* controls, float* sa, float* bev_context, int step,
                            const float* BEVmap_height, const float* BEVmap_normal, 
                            const int BEVmap_size_px, const float BEVmap_res, const float BEVmap_size,
                            const int rollouts, const int timesteps,const int NX, const int NC, const float car_l2, 
                            const float car_w2, const float* std_state, const float* mean_state, const float patch_size)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int state_index = k*timesteps*NX;
    int SA_index = k*12;
    int control_index = k*timesteps*NC;
    int curr, next, ctrl_base;
    float fl[3], fr[3], bl[3], br[3];

    float x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz;
    float cp, sp, cr, sr, cy, sy;
    float res_inv = 1.0f/BEVmap_res;
    float res = BEVmap_res;

    curr = step*NX + state_index;
    next = (step + 1)*NX + state_index;
    ctrl_base = step*NC + control_index;
    
    x = state[curr + x_index];
    y = state[curr + y_index];
    z = state[curr + z_index];
    roll = state[curr + roll_index];
    pitch = state[curr + pitch_index];
    yaw = state[curr + yaw_index];
    vx = state[curr + vx_index];
    vy = state[curr + vy_index];
    vz = state[curr + vz_index];
    ax = state[curr + ax_index];
    ay = state[curr + ay_index];
    az = state[curr + az_index];
    wx = state[curr + wx_index];
    wy = state[curr + wy_index];
    wz = state[curr + wz_index];

    cy = cos(yaw);
    sy = sin(yaw);

    get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);

    roll = atanf(  ((fl[2] + bl[2]) - (fr[2] + br[2]) ) / (4*car_w2));
    pitch = atanf( ((bl[2] + br[2]) - (fl[2] + fr[2]) ) / (4*car_l2));

    cp = cos(pitch);
    sp = sin(pitch);
    cr = cos(roll);
    sr = sin(roll);

    get_crop(bev_context, x,y,cy,sy, BEVmap_height, BEVmap_size_px, res, res_inv, car_l2, car_w2, patch_size, rollouts, k);
    sa[SA_index + 0] = (vx - mean_state[vx_index])/std_state[vx_index];
    sa[SA_index + 1] = (vy - mean_state[vy_index])/std_state[vy_index];
    sa[SA_index + 2] = (vz - mean_state[vz_index])/std_state[vz_index];
    sa[SA_index + 3] = (wx - mean_state[wx_index])/std_state[wx_index];
    sa[SA_index + 4] = (wy - mean_state[wy_index])/std_state[wy_index];
    sa[SA_index + 5] = (wz - mean_state[wz_index])/std_state[wz_index];
    sa[SA_index + 6] = (cr - cos(mean_state[roll_index]))/cos(std_state[roll_index]);
    sa[SA_index + 7] = (sr - sin(mean_state[roll_index]))/sin(std_state[roll_index]);
    sa[SA_index + 8] = (cp - cos(mean_state[pitch_index]))/cos(std_state[pitch_index]);
    sa[SA_index + 9] = (sp - sin(mean_state[pitch_index]))/sin(std_state[pitch_index]);
    sa[SA_index + 10] = (controls[ctrl_base + st_index] - mean_state[NX - NC + st_index])/std_state[NX - NC + st_index];
    sa[SA_index + 11] = (controls[ctrl_base + th_index] - mean_state[NX - NC + th_index])/std_state[NX - NC + th_index];
}

__global__ void euler_step( float* state, const float* controls, const float* Ddot_q, int step,
                            const float* BEVmap_height, const float* BEVmap_normal, 
                            const int BEVmap_size_px, const float BEVmap_res, const float BEVmap_size,
                            const float dt, const int rollouts, const int timesteps,const int NX, const int NC, const float car_l2, 
                            const float car_w2, const float* std_state, const float* mean_state)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int state_index = k*timesteps*NX;
    int SA_index = k*12;
    int control_index = k*timesteps*NC;
    int curr, next, ctrl_base;

    float x, y, z, roll, pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz;
    float cp, sp, cr, sr, cy, sy;
    float fl[3], fr[3], bl[3], br[3];
    float res_inv = 1.0f/BEVmap_res;
    float res = BEVmap_res;
    float dummy;
    float dt_inv = 1/dt;

    curr = step*NX + state_index;
    next = (step + 1)*NX + state_index;
    ctrl_base = step*NC + control_index;
    
    x = state[curr + x_index];
    y = state[curr + y_index];
    z = state[curr + z_index];
    roll = state[curr + roll_index];
    pitch = state[curr + pitch_index];
    yaw = state[curr + yaw_index];
    vx = state[curr + vx_index];
    vy = state[curr + vy_index];
    vz = state[curr + vz_index];
    ax = state[curr + ax_index];
    ay = state[curr + ay_index];
    az = state[curr + az_index];
    wx = state[curr + wx_index];
    wy = state[curr + wy_index];
    wz = state[curr + wz_index];

    cy = cos(yaw);
    sy = sin(yaw);
    cp = cos(pitch);
    sp = sin(pitch);
    cr = cos(roll);
    sr = sin(roll);

    vx += Ddot_q[0] * std_state[ax_index] * dt;
    vy += Ddot_q[1] * std_state[ay_index] * dt;
    vz += Ddot_q[2] * std_state[az_index] * dt;

    wx += Ddot_q[3] * std_state[wx_index] * dt;
    wy += Ddot_q[4] * std_state[wy_index] * dt;
    wz += Ddot_q[5] * std_state[wz_index] * dt;

    ax = Ddot_q[0] * std_state[ax_index] - (vy*wz - vz*wy + sp*GRAVITY);
    ay = Ddot_q[1] * std_state[ay_index] - (-vx*wz + vz*wx - sr*cp*GRAVITY);
    az = Ddot_q[2] * std_state[az_index] - (vx*wy - vy*wx - cp*cr*GRAVITY);

    yaw += dt * (wy * (sr / cp) + wz * (cr / cp));

    x += dt * (vx * cp * cy + vy * (sr * sp * cy - cr * sy) + vz * (cr * sp * cy + sr * sy));
    y += dt * (vx * cp * sy + vy * (sr * sp * sy + cr * cy) + vz * (cr * sp * sy - sr * cy));
    z += dt * (vx * (-sp)   + vy * (sr * cp)                + vz * (cr * cp));

    get_footprint_z(fl, fr, bl, br, dummy, x, y, cy, sy, BEVmap_height, BEVmap_size_px, res_inv, car_l2, car_w2);

    roll = atan2( (fl[2] + bl[2]) - (fr[2] + br[2]),  4*car_w2);
    pitch = atan2( (bl[2] + br[2]) - (fl[2] + fr[2]), 4*car_l2);

    state[next + x_index] = x;
    state[next + y_index] = y;
    state[next + z_index] = z;
    state[next + roll_index] = roll;
    state[next + pitch_index] = pitch;
    state[next + yaw_index] = yaw;
    state[next + vx_index] = vx;
    state[next + vy_index] = vy;
    state[next + vz_index] = vz;
    state[next + ax_index] = ax;
    state[next + ay_index] = ay;
    state[next + az_index] = az;
    state[next + wx_index] = wx;
    state[next + wy_index] = wy;
    state[next + wz_index] = wz;
}

// Wrapper for slip3d kernel
void slip3d_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, float dt, int rollouts, int timesteps, int NX, int NC,
                      float D, float B, float C, float lf, float lr, float Iz, float throttle_to_wheelspeed, float steering_max,
                      int BEVmap_size_px, float BEVmap_res, float BEVmap_size, float car_l2, float car_w2, float cg_height, 
                      float LPF_tau_rpy, float LPF_tau_st, float LPF_tau_th, float res_coeff, float drag_coeff, int block_dim, int grid_dim) {

    int threads = grid_dim;
    int blocks = block_dim;

    slip3d<<<blocks, threads>>>(
        state.data_ptr<float>(), controls.data_ptr<float>(), BEVmap_height.data_ptr<float>(), BEVmap_normal.data_ptr<float>(), dt, rollouts, timesteps, NX, NC,
        D, B, C, lf, lr, Iz, throttle_to_wheelspeed, steering_max,
        BEVmap_size_px, BEVmap_res, BEVmap_size, car_l2, car_w2, cg_height,
        LPF_tau_rpy, LPF_tau_st, LPF_tau_th, res_coeff, drag_coeff
    );
    cudaDeviceSynchronize();
}

// Wrapper for noslip3d kernel
void noslip3d_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, float dt, int rollouts, int timesteps, int NX, int NC,
                      float D, float B, float C, float lf, float lr, float Iz, float throttle_to_wheelspeed, float steering_max,
                      int BEVmap_size_px, float BEVmap_res, float BEVmap_size, float car_l2, float car_w2, float cg_height, 
                      float LPF_tau_rpy, float LPF_tau_st, float LPF_tau_th, float res_coeff, float drag_coeff, int block_dim, int grid_dim) {

    int threads = grid_dim;
    int blocks = block_dim;

    noslip3d<<<blocks, threads>>>(
        state.data_ptr<float>(), controls.data_ptr<float>(), BEVmap_height.data_ptr<float>(), BEVmap_normal.data_ptr<float>(), dt, rollouts, timesteps, NX, NC,
        D, B, C, lf, lr, Iz, throttle_to_wheelspeed, steering_max,
        BEVmap_size_px, BEVmap_res, BEVmap_size, car_l2, car_w2, cg_height,
        LPF_tau_rpy, LPF_tau_st, LPF_tau_th, res_coeff, drag_coeff
    );
    cudaDeviceSynchronize();
}

void preprocess_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor sa, torch::Tensor bev_context, int step,
                         torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, 
                         int BEVmap_size_px, float BEVmap_res, float BEVmap_size,
                         int rollouts, int timesteps, int NX, int NC, float car_l2, 
                         float car_w2, torch::Tensor std_state, torch::Tensor mean_state, float patch_size,
                         int block_dim, int grid_dim) {

    int threads = grid_dim;
    int blocks = block_dim;

    preprocess<<<blocks, threads>>>(
        state.data_ptr<float>(), controls.data_ptr<float>(), sa.data_ptr<float>(), bev_context.data_ptr<float>(), step,
        BEVmap_height.data_ptr<float>(), BEVmap_normal.data_ptr<float>(), 
        BEVmap_size_px, BEVmap_res, BEVmap_size,
        rollouts, timesteps, NX, NC, car_l2, 
        car_w2, std_state.data_ptr<float>(), mean_state.data_ptr<float>(), patch_size
    );
    cudaDeviceSynchronize();
}

void euler_step_launcher(torch::Tensor state, torch::Tensor controls, torch::Tensor Ddot_q, int step,
                         torch::Tensor BEVmap_height, torch::Tensor BEVmap_normal, 
                         int BEVmap_size_px, float BEVmap_res, float BEVmap_size,
                         float dt, int rollouts, int timesteps, int NX, int NC, float car_l2, 
                         float car_w2, torch::Tensor std_state, torch::Tensor mean_state,
                         int block_dim, int grid_dim) {

    int threads = grid_dim;
    int blocks = block_dim;

    euler_step<<<blocks, threads>>>(
        state.data_ptr<float>(), controls.data_ptr<float>(), Ddot_q.data_ptr<float>(), step,
        BEVmap_height.data_ptr<float>(), BEVmap_normal.data_ptr<float>(), 
        BEVmap_size_px, BEVmap_res, BEVmap_size,
        dt, rollouts, timesteps, NX, NC, car_l2, 
        car_w2, std_state.data_ptr<float>(), mean_state.data_ptr<float>()
    );
   cudaDeviceSynchronize();
}
