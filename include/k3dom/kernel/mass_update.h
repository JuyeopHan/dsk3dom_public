#pragma once

#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct Particle;
struct pseudo_vector;

__global__ void gridCellPredictionUpdateKernel(
    GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
    float* __restrict__ born_masses_array, float p_B, int cell_count,
    const float* __restrict__ meas_array_x, const float* __restrict__ meas_array_y, const float* __restrict__ meas_array_z,
    const int* __restrict__ pc_idx_arr, int meas_len, int meas_slot,
    const float* __restrict__ free_array_x, const float* __restrict__ free_array_y, const float* __restrict__ free_array_z,
    const int* __restrict__ free_idx_arr,  const int* __restrict__ source_beam_idx, int free_len, int free_slot,
    int grid_size, int grid_size_z, float resolution,
    float sigma, float ls, float gamma_pow, float mass_scale,
    float sensor_x, float sensor_y, float sensor_z);

} /* namespace dom */
