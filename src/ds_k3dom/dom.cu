#include "common/common.h"
#include "common/cuda_utils.h"
#include "ds_k3dom/dom.h"
#include "common/KDTree.h"
#include "common/dom_types.h"
#include "common/cluster_cuda.h"

#include "common/kernel/init_new_particles.h"
#include "ds_k3dom/kernel/mass_update.h"
#include "ds_k3dom/kernel/init_new_particles.h"

#include <thrust/sort.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>



namespace dom
{

DOM::DOM(const Params& params)
    : DOM_c(params)
{
}

DOM::~DOM()
{
}
void DOM::updateGrid(float dt, KDTreeArr& kdtree_arr_pc, KDTreeArr& kdtree_arr_free, int*& source_beam_arr, ClusterSetCuda& cluster_set_cuda){
    if (updated_time > 0) {
        /*
        following ds-pnd/mib algorithm process
            1. Particle Prediction
            2. Assignment of Particles to Grid Cells
            3. Grid Cell Occupancy Prediction and Update
            4. Update of Persistent particles
            5. initialization of new particles (<- MAIN CHANGE)
            6. statistical moments of grid cells
            7. resampling 
        */
        // ClusterSetCuda cluster_set_cuda??
        particlePrediction(dt);
        particleAssignment();
        gridCellOccupancyUpdate(dt, kdtree_arr_pc, kdtree_arr_free, source_beam_arr);
        updatePersistentParticles();
        initializeNewParticles_cluster(cluster_set_cuda); // <- here
        statisticalMoments();
        resampling();

        // particle_array_test.copy(particle_array, cudaMemcpyDeviceToHost);
        particle_array = particle_array_next;

        
    } else {initializeParticles(kdtree_arr_pc, kdtree_arr_free, source_beam_arr);}
    
    CHECK_ERROR(cudaDeviceSynchronize());
}
void DOM::updateGrid(float t, KDTreeArr& kdtree_arr_pc, KDTreeArr& kdtree_arr_free, int*& source_beam_arr)
{
    if (updated_time > 0) { // skip the first time w/ updated_time = -1.0f
        float dt = t - updated_time;

        /*
        following ds-pnd/mib algorithm process
            1. Particle Prediction
            2. Assignment of Particles to Grid Cells
            3. Grid Cell Occupancy Prediction and Update
            4. Update of Persistent particles
            5. initialization of new particles
            6. statistical moments of grid cells
            7. resampling 
        */

        particlePrediction(dt);
        particleAssignment();

        // k3dom cell occupancy updates, TODO: change poincloud data structure parameter in the function
        gridCellOccupancyUpdate(dt, kdtree_arr_pc, kdtree_arr_free, source_beam_arr);

        updatePersistentParticles();
        initializeNewParticles();
        statisticalMoments(); // required only for simulations
        resampling();
        
        
        // particle_array_test.copy(particle_array, cudaMemcpyDeviceToHost);
        particle_array = particle_array_next;
    }
    else {initializeParticles(kdtree_arr_pc, kdtree_arr_free, source_beam_arr);} // should be exist??

    CHECK_ERROR(cudaDeviceSynchronize());

    updated_time = t;
}

// TODO: change poincloud data structure parameter in the function
void DOM::initializeParticles(KDTreeArr& kdtree_arr_pc, KDTreeArr& kdtree_arr_free, int*& source_beam_arr)
{
    /// reflect measurements first without any assigned particles
    gridCellOccupancyUpdate(0.0f, kdtree_arr_pc, kdtree_arr_free, source_beam_arr);
    
    CHECK_ERROR(cudaGetLastError());

    thrust::device_vector<float> particle_orders_accum(grid_cell_count);
    accumulate(born_masses_array, particle_orders_accum);
    float* particle_orders_array_accum = thrust::raw_pointer_cast(particle_orders_accum.data());

    float new_weight = 1.0f / particle_count;
    // uniformly assign particles into maps
    //float new_weight = 0.005f * grid_cell_count / particle_count;

    normalize_particle_orders(particle_orders_array_accum, grid_cell_count, particle_count);

    initParticlesKernel1<<<grid_map_grid, block_dim>>>(grid_cell_array, particle_array,
                                                       particle_orders_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    initParticlesKernel2<<<particles_grid, block_dim>>>(
        particle_array, grid_cell_array, rng_states, params.init_max_velocity, params.particle_min_vel, grid_size, new_weight, particle_count);

    CHECK_ERROR(cudaGetLastError());
}
// TODO: new particle initialization overload ///////////////
void DOM::initializeNewParticles_cluster(ClusterSetCuda& cluster_set_cuda){
    thrust::device_vector<float> particle_orders_accum(grid_cell_count);
    accumulate(born_masses_array, particle_orders_accum);
    float* particle_orders_array_accum = thrust::raw_pointer_cast(particle_orders_accum.data());

    normalize_particle_orders(particle_orders_array_accum, grid_cell_count, new_born_particle_count);

    // assign particles to each cell
    initNewParticlesKernel1<<<grid_map_grid, block_dim>>>(grid_cell_array,
                                                          born_masses_array, birth_particle_array,
                                                          particle_orders_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    initNewParticlesKernel2_cluster<<<birth_particles_grid, block_dim>>>(birth_particle_array, grid_cell_array, rng_states, cluster_set_cuda,
                                                                 sensor_pos_x - center_pos_x, sensor_pos_y - center_pos_y, sensor_pos_z - center_pos_z,
                                                                 params.stddev_velocity, params.init_max_velocity, params.particle_min_vel,
                                                                 grid_size, grid_size_z, params.resolution, new_born_particle_count);

    CHECK_ERROR(cudaGetLastError());
}
////////////////////////////////////////////////////////////

void DOM::gridCellOccupancyUpdate(float dt, KDTreeArr& kdtree_arr_pc, KDTreeArr& kdtree_arr_free, int*& source_beam_arr)
{
    meas_len = kdtree_arr_pc.length;
    free_len = kdtree_arr_free.length;

    meas_slot = kdtree_arr_pc.slot;
    free_slot = kdtree_arr_free.slot;

    float gamma_pow = powf(params.gamma, dt);
    float alpha_pow = powf(params.alpha, dt);

    CHECK_ERROR(cudaMalloc(&meas_x, meas_len * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&meas_y, meas_len * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&meas_z, meas_len * sizeof(float)));
    CHECK_ERROR(cudaMemcpy(meas_x, kdtree_arr_pc.x_coord, meas_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(meas_y, kdtree_arr_pc.y_coord, meas_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(meas_z, kdtree_arr_pc.z_coord, meas_len * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc(&pc_idx_arr, meas_slot * sizeof(int)));
    CHECK_ERROR(cudaMemcpy(pc_idx_arr, kdtree_arr_pc.idx_arr, meas_slot * sizeof(int), cudaMemcpyHostToDevice));
    
    CHECK_ERROR(cudaMalloc(&free_x, free_len * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&free_y, free_len * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&free_z, free_len * sizeof(float)));
    CHECK_ERROR(cudaMemcpy(free_x, kdtree_arr_free.x_coord, free_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(free_y, kdtree_arr_free.y_coord, free_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(free_z, kdtree_arr_free.z_coord, free_len * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc(&free_idx_arr, free_slot * sizeof(int)));
    CHECK_ERROR(cudaMemcpy(free_idx_arr, kdtree_arr_free.idx_arr, free_slot * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc(&source_beam_idx, free_len * sizeof(int)));
    CHECK_ERROR(cudaMemcpy(source_beam_idx, source_beam_arr, free_len* sizeof(int), cudaMemcpyHostToDevice));

    // TODO: change poincloud data structure parameter in the function
    gridCellPredictionUpdateKernel<<<grid_map_grid, block_dim>>>(grid_cell_array, particle_array,
                                                                born_masses_array, params.birth_prob, grid_cell_count,
                                                                meas_x, meas_y, meas_z,
                                                                pc_idx_arr, meas_len, meas_slot,
                                                                free_x, free_y, free_z, free_idx_arr,
                                                                source_beam_idx, free_len, free_slot,
                                                                grid_size, grid_size_z, params.resolution,
                                                                params.sigma, params.ls, gamma_pow, alpha_pow, params.prior_all,
                                                                sensor_pos_x - center_pos_x,
                                                                sensor_pos_y - center_pos_y,
                                                                sensor_pos_z - center_pos_z,
                                                                center_pos_z);


    CHECK_ERROR(cudaGetLastError());

    CHECK_ERROR(cudaFree(meas_x));
    CHECK_ERROR(cudaFree(meas_y));
    CHECK_ERROR(cudaFree(meas_z));
    CHECK_ERROR(cudaFree(pc_idx_arr));

    CHECK_ERROR(cudaFree(free_x));
    CHECK_ERROR(cudaFree(free_y));
    CHECK_ERROR(cudaFree(free_z));
    CHECK_ERROR(cudaFree(free_idx_arr));
    CHECK_ERROR(cudaFree(source_beam_idx));
}

} /* namespace dom */
