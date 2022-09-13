#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/cluster_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "ds_k3dom/kernel/init_new_particles.h"

namespace dom
{
__global__ void initNewParticlesKernel2_cluster(ParticlesSoA birth_particle_array,
                                        const GridCell* __restrict__ grid_cell_array,
                                        curandState* __restrict__ global_state,
                                        ClusterSetCuda cluster_set_cuda,
                                        float sensor_x, float sensor_y, float sensor_z,
                                        float stddev_velocity, float max_velocity,
                                        float min_vel, int grid_size, int grid_size_z, float resolution, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        int cell_idx = birth_particle_array.grid_cell_idx[i];
        const GridCell& grid_cell = grid_cell_array[cell_idx];
        bool associated = birth_particle_array.associated[i];

        float x = cell_idx % grid_size + 0.5f;
        float y = (cell_idx % (grid_size * grid_size)) / grid_size + 0.5f;
        float z = cell_idx / (grid_size * grid_size) + 0.5f;

        // coordinate from sensor
        float x_pos = (x - (float)grid_size / 2.0f) * resolution - sensor_x; 
        float y_pos = (y - (float)grid_size / 2.0f) * resolution - sensor_y;
        float z_pos = (z - (float)grid_size_z / 2.0f) * resolution - sensor_z;

        glm::vec3 bbox_min;
        glm::vec3 bbox_max;
        glm::vec3 vel;
        bool in_cluster = false;
        for(int j = 0; j < cluster_set_cuda.size; ++j){
            bbox_min = cluster_set_cuda.bbox_min[j];
            bbox_max = cluster_set_cuda.bbox_max[j];

            if (x_pos <= bbox_max.x && bbox_min.x <= x_pos && 
                y_pos <= bbox_max.y && bbox_min.y <= y_pos &&
                z_pos <= bbox_max.z && bbox_min.z <= z_pos) {
                in_cluster = true;
                vel = cluster_set_cuda.vel[j];
                continue;
            }
        }
        
        float vel_x;
        float vel_y;
        float vel_z;

        if(in_cluster){
            // uniform distribution
            // float vel_x_min = min(0.0f, max(-max_velocity, vel.x - max_velocity));
            // float vel_y_min = min(0.0f, max(-max_velocity, vel.y - max_velocity));
            // float vel_z_min = min(0.0f, max(-max_velocity, vel.z - max_velocity));

            // float vel_x_max = max(0.0f, min(max_velocity, vel.x + max_velocity));
            // float vel_y_max = max(0.0f, min(max_velocity, vel.y + max_velocity));
            // float vel_z_max = max(0.0f, min(max_velocity, vel.z + max_velocity));

            // vel_x = curand_uniform(&local_state, vel_x_min, vel_x_max);
            // vel_y = curand_uniform(&local_state, vel_y_min, vel_y_max);
            // vel_z = curand_uniform(&local_state, vel_z_min, vel_z_max);

            // gaussian
            vel_x = curand_normal(&local_state, vel.x,  stddev_velocity);
            vel_y = curand_normal(&local_state, vel.y,  stddev_velocity);
            vel_z = curand_normal(&local_state, vel.z,  stddev_velocity);
        } else {
            // may employ different model along with association
            vel_x = curand_uniform(&local_state, -max_velocity, max_velocity);
            vel_y = curand_uniform(&local_state, -max_velocity, max_velocity);
            vel_z = curand_uniform(&local_state, -max_velocity, max_velocity);
        }

        // minimum velocity requirement to prevent static particles
        float vel_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;
        if (vel_sq < min_vel * min_vel && vel_sq != 0.0f) {
            float rate = min_vel / sqrtf(vel_sq);
            vel_x *= rate;
            vel_y *= rate;
            vel_z *= rate;
        }

        if (associated)
        {
            birth_particle_array.weight[i] = grid_cell.w_A;
        }
        else
        {
            birth_particle_array.weight[i] = grid_cell.w_UA;
        }

        birth_particle_array.state_pos[i] = glm::vec3(x, y, z);
        birth_particle_array.state_vel[i] = glm::vec3(vel_x, vel_y, vel_z);
    }

    global_state[thread_id] = local_state;
}

}