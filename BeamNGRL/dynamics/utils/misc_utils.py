import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import torch

kernel_code = '''
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
'''
mod = SourceModule(kernel_code)
cuda_kernel = mod.get_function('rotate_crop')
block_size = (32, 32, 1)

def crop_rotate_batch(input_array, output_array, center, angle):
	input_array = input_array.clone()
	output_array = output_array.clone().transpose(0,2)
	in_H = input_array.shape[0]
	in_W = input_array.shape[1]
	out_H = output_array.shape[0]
	out_W = output_array.shape[1]
	N = center.shape[0]
	
	input_images = gpuarray.to_gpu(input_array.cpu().numpy())
	output_images = gpuarray.to_gpu(output_array.cpu().numpy())
	angles = gpuarray.to_gpu(angle.clone().cpu().numpy())
	centers = gpuarray.to_gpu(center.clone().cpu().numpy())
	grid_size = ((out_W + block_size[0] - 1) // block_size[0], (out_H + block_size[1] - 1) // block_size[1], N)

	cuda_kernel(input_images, output_images, angles, centers, np.int32(in_H), np.int32(in_W), np.int32(out_H), np.int32(out_W), np.int32(N), block=block_size, grid=grid_size)
	
	output_images = torch.from_numpy(output_images.get()).transpose(0,2).to(torch.device('cuda'))
	
	return output_images