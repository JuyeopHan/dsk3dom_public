#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "ds_k3dom/kernel/mass_update.h"

#include "common/KDTree_cu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

inline __device__ float kernel_func(float d, float ls, float sigma) {
    return max(0.0f, sigma * ((2.0f + cospif(2.0f * d / ls)) * (1.0f - d / ls) / 3.0f + sinpif(2.0f * d / ls) / (2.0f * 3.141592f)));
}

// x, y, z : grid cell position
__device__ float3 getLidarMeasurementGrid(
    const float* __restrict__ meas_array_x,
    const float* __restrict__ meas_array_y,
    const float* __restrict__ meas_array_z,
    const int* __restrict__ pc_idx_arr, int meas_len, int meas_slot,
    const float* __restrict__ free_array_x,
    const float* __restrict__ free_array_y,
    const float* __restrict__ free_array_z,
    const int* __restrict__ free_idx_arr, const int* __restrict__ source_beam_idx_arr , int free_len, int free_slot,
    float x, float y, float z, float resolution, float sigma, float ls, float prior_all)
{
    float del_occ = 0.0f;
    float del_free = 0.0f;

    float d, d_query, d_proj, d_source, x_meas, y_meas, z_meas, x_source, y_source, z_source;
    int meas_idx, free_idx, source_beam_idx; 
    //int flag = 0;

    // range search for occupied points
    pseudo_vector meas_neighbor_idx_vec =
        neighborhoodIndices(meas_array_x, meas_array_y, meas_array_z, pc_idx_arr, meas_len, meas_slot, x, y, z, ls);

    // range search for free points
    float l_free = 1.0f;
    pseudo_vector free_neighbor_idx_vec =
        neighborhoodIndices(free_array_x, free_array_y, free_array_z, free_idx_arr, free_len, free_slot, x, y, z, l_free);

    // update for occ
    for (int i = 0 ; i < meas_neighbor_idx_vec.length; ++i){
        meas_idx = pc_idx_arr[meas_neighbor_idx_vec.vector[i]];
        x_meas = meas_array_x[meas_idx];
        y_meas = meas_array_y[meas_idx];
        z_meas = meas_array_z[meas_idx];
        d = sqrtf((x - x_meas) * (x - x_meas) + (y - y_meas) * (y - y_meas) + (z - z_meas) * (z - z_meas));
        if (d < ls){
            del_occ += kernel_func(d, ls, sigma);
        }
    }

    // update for free (line based)
    // 0. initialize beam search list
    pseudo_vector beam_search_vec;
    beam_search_vec.length = free_neighbor_idx_vec.length;

    for (int i = 0; i < beam_search_vec.length; ++i){
        free_idx = free_idx_arr[free_neighbor_idx_vec.vector[i]];
        source_beam_idx = source_beam_idx_arr[free_idx];
        beam_search_vec.vector[i] = source_beam_idx;
    }

    pseudo_vector unique_arr;
    int unique_arr_len = beam_search_vec.length;
    unique_arr.length = 0;
    for (int i = 0; i < unique_arr_len; ++i){
        if(!searchKeyInArr(unique_arr.vector, unique_arr.length, beam_search_vec.vector[i])){
            unique_arr.vector[unique_arr.length++] = beam_search_vec.vector[i];
        }
    }

    for (int i = 0; i < unique_arr.length; ++i){
        beam_search_vec.vector[i] = 0;
    }
    for (int i = 0; i < unique_arr.length; ++i){
        beam_search_vec.vector[i] = unique_arr.vector[i];
    }
    beam_search_vec.length = unique_arr.length;

    
    // 2. by using search beam list, update the free measure.
    if (beam_search_vec.length > 0){
        
        for (int i = 0; i < beam_search_vec.length; ++i){
        source_beam_idx = beam_search_vec.vector[i];
        x_source = meas_array_x[source_beam_idx];
        y_source = meas_array_y[source_beam_idx];
        z_source = meas_array_z[source_beam_idx];

        d_source = sqrtf(x_source * x_source + y_source * y_source + z_source * z_source);

        d_proj = (x * x_source + y * y_source + z * z_source) / d_source;

        // update for free (use closest point in each measurement ray as free measurement)
        if ((0 < d_proj) && (d_proj < d_source)) {
            d_query = sqrtf(x * x + y * y + z * z);
            d = sqrtf(d_query * d_query - d_proj * d_proj);
            if (d < ls) {
                    del_free += kernel_func(d, ls, sigma);
                } 
            }
        }
    }
    
    assert(del_free >= 0.0f && del_occ >= 0.0f);  // include checking nan

    float sum_meas = del_free + del_occ + prior_all;
    float occ_mass_meas = del_occ / sum_meas;
    float free_mass_meas = del_free / sum_meas;
    float all_mass_meas = prior_all / sum_meas; 

    return make_float3(occ_mass_meas, free_mass_meas, all_mass_meas);
}

__device__ void DSRuleCombinationLidar(GridCell& grid_cell, float free_mass_meas, float occ_mass_meas, float all_mass_meas){
    float all_mass_pred = max(0.0f, 1.0f - grid_cell.free_mass - grid_cell.static_mass - grid_cell.dynamic_mass - grid_cell.occ_mass);
    float occ_bel_meas = occ_mass_meas + all_mass_meas;
    float den = 1 - grid_cell.free_mass * occ_mass_meas - (grid_cell.static_mass + grid_cell.dynamic_mass + grid_cell.occ_mass) * free_mass_meas;
    assert(den <= 1.0f && den >= 0.0f);
    
    float free_mass_update = (grid_cell.free_mass * free_mass_meas + grid_cell.free_mass * all_mass_meas + all_mass_pred * free_mass_meas) / den;
    float static_mass_update = grid_cell.static_mass * occ_bel_meas / den;
    float dynamic_mass_update = grid_cell.dynamic_mass * occ_bel_meas / den;
    float occ_mass_update = (grid_cell.occ_mass * occ_mass_meas + grid_cell.occ_mass * all_mass_meas + all_mass_pred * occ_mass_meas) / den;

    grid_cell.free_mass = free_mass_update;
    grid_cell.static_mass = static_mass_update;
    grid_cell.dynamic_mass = dynamic_mass_update;
    grid_cell.occ_mass = occ_mass_update;

    assert(grid_cell.static_mass >= 0.0f && grid_cell.dynamic_mass >= 0.0f && grid_cell.free_mass >= 0.0f && grid_cell.occ_mass >= 0.0f);
    // include checking nan
}

__device__ void predict_mass(GridCell& grid_cell, float free_mass_pred, float static_mass_pred, float& dynamic_mass_pred, float occ_mass_pred,
 float gamma_pow, float beta) {
    float beta_mass = 0.98f * occ_mass_pred;

    float total_mass = max(0.0f, 1.0f - free_mass_pred - static_mass_pred - dynamic_mass_pred - occ_mass_pred);
    float pignistic_dynamic = dynamic_mass_pred + 0.5f * occ_mass_pred + 0.3333 * total_mass;
    float pignistic_static = dynamic_mass_pred + 0.5f * occ_mass_pred + 0.3333 * total_mass;
    float delta = pignistic_dynamic/(pignistic_dynamic + pignistic_static + 1e-6);

    grid_cell.dynamic_mass = min(dynamic_mass_pred + gamma_pow * delta * beta_mass, 0.99999f);
    grid_cell.static_mass = max(0.0f, min(gamma_pow * (static_mass_pred  + (1.0f - delta) * beta_mass), 0.99999f - grid_cell.dynamic_mass));
    grid_cell.occ_mass = max(0.0f, min(gamma_pow * (1.0f - 0.98f) * occ_mass_pred, 0.99999f - grid_cell.dynamic_mass - grid_cell.static_mass));
    grid_cell.free_mass = max(0.0f, min(gamma_pow * free_mass_pred, 0.99999f - grid_cell.dynamic_mass - grid_cell.static_mass - grid_cell.occ_mass));
    dynamic_mass_pred = grid_cell.dynamic_mass;

    assert(grid_cell.static_mass >= 0.0f && grid_cell.dynamic_mass >= 0.0f && grid_cell.free_mass >= 0.0f && grid_cell.occ_mass >= 0.0f);
    // include checking nan
    assert(grid_cell.static_mass + grid_cell.occ_mass +  grid_cell.dynamic_mass + grid_cell.free_mass <= 1.0f);
}

// for now only for lidar sensor
__device__ void update_mass(GridCell& grid_cell, int cell_idx, 
                            const float* __restrict__ meas_array_x,
                            const float* __restrict__ meas_array_y,
                            const float* __restrict__ meas_array_z,
                            const int* __restrict__ pc_idx_arr, int meas_len, int meas_slot,
                            const float* __restrict__ free_array_x,
                            const float* __restrict__ free_array_y,
                            const float* __restrict__ free_array_z,
                            const int* __restrict__ free_idx_arr,const int* __restrict__ source_beam_idx , int free_len, int free_slot,
                            int grid_size, int grid_size_z, float resolution, float sigma, float ls, float prior_all,
                            float sensor_x, float sensor_y, float sensor_z) {
    float x_idx = cell_idx % grid_size + 0.5f;
    float y_idx = (cell_idx % (grid_size * grid_size)) / grid_size + 0.5f;
    float z_idx = cell_idx / (grid_size * grid_size) + 0.5f;

    // coordinates from sensor
    float x = (x_idx - (float)grid_size / 2.0f) * resolution - sensor_x;
    float y = (y_idx - (float)grid_size / 2.0f) * resolution - sensor_y;
    float z = (z_idx - (float)grid_size_z / 2.0f) * resolution - sensor_z;
    

    // kernel accumulation
    // TODO: change poincloud data structure parameter in the function
    float3 masses =getLidarMeasurementGrid(meas_array_x, meas_array_y, meas_array_z, pc_idx_arr, meas_len, meas_slot,
    free_array_x, free_array_y, free_array_z, free_idx_arr, source_beam_idx, free_len, free_slot, x, y, z, resolution, sigma, ls, prior_all);
    float occ_mass_meas = masses.x;
    float free_mass_meas = masses.y;
    float all_mass_meas = masses.z;
    assert(grid_cell.static_mass >= 0.0f && grid_cell.dynamic_mass >= 0.0f && grid_cell.free_mass >= 0.0f && grid_cell.occ_mass >= 0.0f);
    // include checking nan
    assert(grid_cell.static_mass + grid_cell.occ_mass +  grid_cell.dynamic_mass + grid_cell.free_mass <= 1.0f);

    if (free_mass_meas > 0.0f || occ_mass_meas > 0.0f || all_mass_meas > 0.0f) {      

        DSRuleCombinationLidar(grid_cell, free_mass_meas, occ_mass_meas, all_mass_meas);
    }
}

__device__ float separate_newborn_part(float m_dyn_pred, float occ_bel_pred, float m_dyn_up, float p_B)
{
    if (m_dyn_pred <= 0.0f) {   // (0,0) case is included here
        return 0.0f;
    } else if (occ_bel_pred <= m_dyn_pred) {
        return 0.0f;
    } else {
        return (m_dyn_up * p_B * (1 - occ_bel_pred)) / (m_dyn_pred + p_B * (1 - occ_bel_pred));
    }
 }


__global__ void gridCellPredictionUpdateKernel(
    GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
    float* __restrict__ born_masses_array, float p_B, int cell_count,
    const float* __restrict__ meas_array_x, const float* __restrict__ meas_array_y, const float* __restrict__ meas_array_z,
    const int* __restrict__ pc_idx_arr, int meas_len, int meas_slot,
    const float* __restrict__ free_array_x, const float* __restrict__ free_array_y, const float* __restrict__ free_array_z,
    const int* __restrict__ free_idx_arr,  const int* __restrict__ source_beam_idx, int free_len, int free_slot,
    int grid_size, int grid_size_z, float resolution,
    float sigma, float ls, float gamma_pow, float beta, float prior_all,
    float sensor_x, float sensor_y, float sensor_z, float center_pos_z)
    {

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
        {
            int start_idx = grid_cell_array[i].start_idx;
            int end_idx = grid_cell_array[i].end_idx;

            float free_mass_pred = grid_cell_array[i].free_mass;
            float static_mass_pred = grid_cell_array[i].static_mass;
            float occ_mass_pred = grid_cell_array[i].occ_mass;
            float dynamic_mass_pred = 0.0f;
            if (start_idx != -1)
            {
                for (int j = start_idx; j < end_idx + 1; j++) {
                    dynamic_mass_pred += particle_array.weight[j];
                }
                assert(dynamic_mass_pred >= 0.0f);
            }
            predict_mass(grid_cell_array[i], free_mass_pred, static_mass_pred, dynamic_mass_pred, occ_mass_pred, gamma_pow, beta);
            
            float occ_bel_pred = grid_cell_array[i].static_mass + grid_cell_array[i].dynamic_mass;
            //float occ_bel_pred = grid_cell_array[i].dynamic_mass;

            update_mass(grid_cell_array[i], i, meas_array_x, meas_array_y, meas_array_z, pc_idx_arr, meas_len, meas_slot,
            free_array_x, free_array_y, free_array_z, free_idx_arr, source_beam_idx , free_len, free_slot,
            grid_size, grid_size_z, resolution, sigma, ls, prior_all, sensor_x, sensor_y, sensor_z);

            float z_idx = i / (grid_size * grid_size) + 0.5f;
            float z = (z_idx - (float)grid_size_z / 2.0f) * resolution + center_pos_z;

            // if (z < 0.2f) { // remove new born particle at bottom
            //     grid_cell_array[i].static_mass = min(1.0f, grid_cell_array[i].dynamic_mass + grid_cell_array[i].static_mass + born_masses_array[i]);
            //     born_masses_array[i] = 0.0f;
            //     grid_cell_array[i].dynamic_mass = 0.0f;
            //     grid_cell_array[i].pers_mass = grid_cell_array[i].dynamic_mass;
            // } else {
            //     born_masses_array[i] = separate_newborn_part(dynamic_mass_pred, occ_bel_pred, grid_cell_array[i].dynamic_mass, p_B);
            //     grid_cell_array[i].pers_mass = grid_cell_array[i].dynamic_mass - born_masses_array[i];
            // }

            born_masses_array[i] = separate_newborn_part(dynamic_mass_pred, occ_bel_pred, grid_cell_array[i].dynamic_mass, p_B);
            grid_cell_array[i].pers_mass = grid_cell_array[i].dynamic_mass - born_masses_array[i];
            
            grid_cell_array[i].pred_mass = dynamic_mass_pred;
            
        }
    }
} /* namespace dom */
