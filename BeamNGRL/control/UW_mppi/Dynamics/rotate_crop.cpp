__global__ void rotate_crop(const float* in, float* out, float* rotation_angles, int* cent, int in_H, int in_W, int out_H, int out_W, int N)
{
	// we launched crop_size x crop_size threads, these coordinates correspond to the coordinate in the target image
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= out_W || j >= out_H || k > N) 
		return;

	int centerX = cent[k*2 + 0];
	int centerY = cent[k*2 + 1];
	float ct = cosf(rotation_angles[k]);
	float st = sinf(rotation_angles[k]);
	// Translate coordinates to be relative to the center
	float translatedX = float(i) - out_W/2;
	float translatedY = float(j) - out_H/2;
	// Rotate the coordinates
	int I = int(translatedX*ct - translatedY*st) + centerX;
	int J = int(translatedX*st + translatedY*ct) + centerY;

	int out_index = j*out_W*N + i*N + k;
	int in_index  = J*in_W + I;

	out[out_index] = in[in_index];
}