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

__device__ float nan_to_num(float x, float replace)
{
    if (std::isnan(x) or std::isinf(x)) 
    {
        return replace;
    }
    return x;
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

    z = map_to_elev(x, y, elev, map_size_px, res_inv);
    fl[2] = map_to_elev(fl[0], fl[1], elev, map_size_px, res_inv);
    fr[2] = map_to_elev(fr[0], fr[1], elev, map_size_px, res_inv);
    bl[2] = map_to_elev(bl[0], bl[1], elev, map_size_px, res_inv);
    br[2] = map_to_elev(br[0], br[1], elev, map_size_px, res_inv);
}

__global__ void rollout(float* state, const float* controls, const float* BEVmap_height, const float* BEVmap_normal, const float dt, const int rollouts, const int timesteps, const int NX, const int NC,
                        const float D, const float B, const float C, const float lf, const float lr, const float Iz, const float throttle_to_wheelspeed, const float steering_max,
                        const int BEVmap_size_px, const float BEVmap_res, const float BEVmap_size, float car_l2, const float car_w2, const float cg_height)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int state_index = k*timesteps*NX;
    int control_index = k*timesteps*NC;

    int curr, next, ctrl_base;

    float x, y, z, roll, pitch, last_roll, last_pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz;
    float st, w;

    float vf, vr, Kr, Kf, alphaf, alphar, alpha_z, sigmaf, sigmar, sigmaf_x, sigmaf_y, sigmar_x, sigmar_y, Fr, Ff, Frx, Fry, Ffx, Ffy;
    float cp, sp, cr, sr, cy, sy, ct;
    float fl[3], fr[3], bl[3], br[3];
    float res_inv = 1.0f/BEVmap_res;

    for(int t = 0; t < timesteps-1; t++)
    {
        curr = t*NX + state_index;
        next = (t + 1)*NX + state_index;

        ctrl_base = t*NC + control_index;
        
        st = controls[ctrl_base + st_index] * steering_max;
        w = controls[ctrl_base + th_index] * throttle_to_wheelspeed;

        x = state[curr + x_index];
        y = state[curr + y_index];

        vx = state[curr + vx_index];
        vy = state[curr + vy_index];
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
        ct = sqrtf(1 - (sp*sp) - (sr*sr));

        vf = (vx * cosf(st) + vy * sinf(st));
        vr = vx;

        Kr = (w - vr) / vr;
        Kf = (w - vf) / vf;

        alphaf = st - atan2f(wz * lf + vy, vx);
        alphar = atan2f(wz * lr - vy, vx);

        sigmaf_x = nan_to_num( Kf / (1 + Kf), 0.01);
        sigmaf_y = nan_to_num( tanf(alphaf) / (1 + Kf), 0.01);
        sigmaf = sqrtf(sigmaf_x * sigmaf_x + sigmaf_y * sigmaf_y);

        sigmar_x = nan_to_num( Kr / (1 + Kr), 0.01);
        sigmar_y = nan_to_num( tanf(alphar) / (1 + Kr), 0.01);
        sigmar = sqrtf(sigmar_x * sigmar_x + sigmar_y * sigmar_y);

        float Nf, Nr;
        Nf = az*0.5 - ax*cg_height/(car_l2*2);
        Nr = az*0.5 + ax*cg_height/(car_l2*2);

        Fr = Nr * D * sinf(C * atanf(B * sigmar));
        Ff = Nf * D * sinf(C * atanf(B * sigmaf));

        Frx = Fr * sigmar_x / sigmar;
        Fry = Fr * sigmar_y / sigmar;
        Ffx = Ff * sigmaf_x / sigmaf;
        Ffy = Ff * sigmaf_y / sigmaf;

        ax = Frx + Ffx * cosf(st) - Ffy * sinf(st) - sp*GRAVITY;
        ay = Fry + Ffy * cosf(st) + Ffx * sinf(st) - sr*GRAVITY;
        az = GRAVITY*ct - vx*wy + vy*wx; // don't integrate this acceleration

        alpha_z = (Ffx * sinf(st) * lf + Ffy * lf * cosf(st) - Fry * lr) / Iz;

        vx += (ax + vy*wz) * dt;
        vy += (ay - vx*wz) * dt;
        wz += alpha_z * dt;
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