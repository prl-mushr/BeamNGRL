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
#define GRAVITY 9.8

__device__ float nan_to_num(float x, float replace)
{
    if (std::isnan(x) or std::isinf(x)) 
    {
        return replace;
    }
    return x;
}

__global__ void rollout(float* state, float* controls, float* BEVmap_height, float* BEVmap_normal, float dt, int rollouts, int timesteps, int NX, int NC,
                        float D, float B, float C, float lf, float lr, float Iz, float throttle_to_wheelspeed, float steering_max,
                        int BEVmap_size_px, float BEVmap_res, float BEVmap_size )
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int state_index = k*timesteps*NX;
    int control_index = k*timesteps*NC;

    int curr, next, ctrl_base;

    float x, y, z, roll, pitch, last_roll, last_pitch, yaw, vx, vy, vz, ax, ay, az, wx, wy, wz;
    float st, w;

    float vf, vr, Kr, Kf, alphaf, alphar, alpha_z, sigmaf, sigmar, sigmaf_x, sigmaf_y, sigmar_x, sigmar_y, Fr, Ff, Frx, Fry, Ffx, Ffy;
    float cp, sp, cr, sr, cy, sy, ct;
    float normal_x, normal_y, normal_z, heading_x, heading_y, heading_z;
    float left_x, left_y, left_z, forward_x, forward_y, forward_z;
    int img_X, img_Y;
    int norm_base;
    float bevmap_size_b2 = BEVmap_size*0.5f, res_inv = 1.0f/BEVmap_res;

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

        img_X = fminf(fmaxf((int)((x + bevmap_size_b2)*res_inv), 0), BEVmap_size_px - 1);
        img_Y = fminf(fmaxf((int)((y + bevmap_size_b2)*res_inv), 0), BEVmap_size_px - 1);

        z = BEVmap_height[img_Y * BEVmap_size_px + img_X];

        yaw = state[curr + yaw_index];

        norm_base = img_Y * BEVmap_size_px * 3 + img_X * 3;
        normal_x = BEVmap_normal[norm_base + 0];
        normal_y = BEVmap_normal[norm_base + 1];
        normal_z = BEVmap_normal[norm_base + 2];
        
        cy = cosf(yaw);
        sy = sinf(yaw);
        heading_x = cy;
        heading_y = sy;
        heading_z = 0.0;

        left_x = normal_y * heading_z - normal_z * heading_y;
        left_y = normal_z * heading_x - normal_x * heading_z;
        left_z = normal_x * heading_y - normal_y * heading_x;

        forward_x = left_y * normal_z - left_z * normal_y;
        forward_y = left_z * normal_x - left_x * normal_z;
        forward_z = left_x * normal_y - left_y * normal_x;

        roll = asinf(left_z);
        pitch = -asinf(forward_z);

        wx = (roll - last_roll)/dt;
        wy = (pitch - last_pitch)/dt;

        cp = cosf(pitch);
        sp = sinf(pitch);
        cr = cosf(roll);
        sr = sinf(roll);
        ct = sqrtf(cp*cp + cr*cr);

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

        Fr = 0.5 * D * sinf(C * atanf(B * sigmar));
        Ff = 0.5 * D * sinf(C * atanf(B * sigmaf));

        Frx = Fr * sigmar_x / sigmar;
        Fry = Fr * sigmar_y / sigmar;
        Ffx = Ff * sigmaf_x / sigmaf;
        Ffy = Ff * sigmaf_y / sigmaf;

        ax = Frx + Ffx * cos(st) - Ffy * sin(st) + sp*GRAVITY;
        ay = Fry + Ffy * cos(st) + Ffx * sin(st) + sr*GRAVITY;
        az = GRAVITY*ct;

        alpha_z = (Ffx * sin(st) * lf + Ffy * lf * cos(st) - Fry * lr) / Iz;

        vx += ax * dt;
        vy += ay * dt;
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