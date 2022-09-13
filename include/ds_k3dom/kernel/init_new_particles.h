#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct Particle;

__global__ void initNewParticlesKernel2_cluster(ParticlesSoA birth_particle_array,
                                        const GridCell* __restrict__ grid_cell_array,
                                        curandState* __restrict__ global_state,
                                        ClusterSetCuda cluster_set_cuda,
                                        float sensor_x, float sensor_y, float sensor_z,
                                        float stddev_velocity, float max_velocity,
                                        float min_vel, int grid_size, int grid_size_z, float resolution, int particle_count);

}
