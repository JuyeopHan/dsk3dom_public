#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "k3dom/kernel/mass_update.h"

#include "common/KDTree_cu.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

inline __device__ float calc_credit(float mass, float mass_scale = 3.0f) {
    return 1.0f - 2.0f / (1.0f + expf(2*mass/mass_scale));
}

inline __device__ float positive_diff(float a, float b) {
    return max(0.0f, a - b);
}

inline __device__ float calc_ratio(float numerator, float rest) {
    return (numerator <= 0.0f || rest < 0.0f)? 0.0f : (1.0f / (1.0f + rest / numerator));
}

inline __device__ float kernel_func(float d, float ls, float sigma) {
    return max(0.0f, sigma * ((2.0f + cospif(2.0f * d / ls)) * (1.0f - d / ls) / 3.0f + sinpif(2.0f * d / ls) / (2.0f * 3.141592f)));
}

// since very close free and occupancy measurements cancel each other
inline __device__ float distance_with_margin(float distance, float resolution) {
    return max(0.9f * distance, distance - 2.0f * resolution);
}

/*
 * x, y, z : grid cell position
 *
 */
__device__ float2 BGKI(
    const float* __restrict__ meas_array_x,
    const float* __restrict__ meas_array_y,
    const float* __restrict__ meas_array_z,
    const int* __restrict__ pc_idx_arr, int meas_len, int meas_slot,
    const float* __restrict__ free_array_x,
    const float* __restrict__ free_array_y,
    const float* __restrict__ free_array_z,
    const int* __restrict__ free_idx_arr, const int* __restrict__ source_beam_idx_arr , int free_len, int free_slot,
    float x, float y, float z, float resolution, float sigma, float ls)
{
    float del_occ = 0.0f;
    float del_free = 0.0f;

    // //TODO: efficient search considering kernel function (ex: rtree)
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

    // update for free (sample based)

    // float x_free, y_free, z_free; 

    // for (int i = 0 ; i < free_neighbor_idx_vec.length; ++i){
    //     free_idx = free_idx_arr[free_neighbor_idx_vec.vector[i]];
    //     x_free = free_array_x[free_idx];
    //     y_free = free_array_y[free_idx];
    //     z_free = free_array_z[free_idx];
    //     d = sqrtf((x - x_free) * (x - x_free) + (y - y_free) * (y - y_free) + (z - z_free) * (z - z_free));
    //     if (d < ls){
    //         del_free += kernel_func(d, ls, sigma);
    //     }
    // }

    // update for free (line based)

    // 0. initialize beam search list
    pseudo_vector beam_search_vec;
    beam_search_vec.length = free_neighbor_idx_vec.length;

    // 잘 뽑힘
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
        // 여기서 문제가 생기는 듯.
        if ((0 < d_proj) && (d_proj < d_source)) {
            d_query = sqrtf(x * x + y * y + z * z);
            //printf(" d_query : %f, d_proj : %f \n", d_query, d_proj);
            d = sqrtf(d_query * d_query - d_proj * d_proj);
            if (d < ls) {
                    del_free += kernel_func(d, ls, sigma);
                } 
            }
        }
    }
    

     assert(del_free >= 0.0f && del_occ >= 0.0f);  // include checking nan

    return make_float2(del_occ, del_free);
}

__device__ void predict_mass(GridCell& grid_cell, float dynamic_mass_pred, float d_max, float gamma_pow, float mass_scale) {

    grid_cell.static_mass *= gamma_pow;
    grid_cell.dynamic_mass = min(dynamic_mass_pred, max(0.0f, d_max - grid_cell.static_mass));
    grid_cell.free_mass = d_max - grid_cell.dynamic_mass;
}

__device__ void update_mass(GridCell& grid_cell, int cell_idx, 
                            const float* __restrict__ meas_array_x,
                            const float* __restrict__ meas_array_y,
                            const float* __restrict__ meas_array_z,
                            const int* __restrict__ pc_idx_arr, int meas_len, int meas_slot,
                            const float* __restrict__ free_array_x,
                            const float* __restrict__ free_array_y,
                            const float* __restrict__ free_array_z,
                            const int* __restrict__ free_idx_arr,const int* __restrict__ source_beam_idx , int free_len, int free_slot,
                            int grid_size, int grid_size_z, float resolution, float sigma, float ls, float mass_scale,
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
    float2 del = BGKI(meas_array_x, meas_array_y, meas_array_z,pc_idx_arr, meas_len, meas_slot,
    free_array_x, free_array_y, free_array_z, free_idx_arr, source_beam_idx, free_len, free_slot, x, y, z, resolution, sigma, ls);
    float del_occ = del.x;
    float del_free = del.y;

    if (del_free > 0.0f || del_occ > 0.0f) {      
        float m_f = grid_cell.free_mass;
        float m_s = grid_cell.static_mass;
        float m_d = grid_cell.dynamic_mass;

        assert(m_f >= 0.0f && m_s >= 0.0f && m_d >= 0.0f);

        /////////////// update step1: rebalancing ///////////////
        float del_sigma = del_free + del_occ;
        float prime_sigma = m_f + m_s + m_d;
        
        float credit = calc_credit(sqrtf(del_sigma * prime_sigma), mass_scale);

        float lamda_DtoF = credit * positive_diff(del_free, del_occ) / del_sigma * m_d / prime_sigma;
        float lamda_StoFD = credit * positive_diff(del_free, del_occ) / del_sigma * positive_diff(m_s + m_d, m_f) / prime_sigma;
        float lamda_FtoD = credit * positive_diff(del_occ, del_free) / del_sigma * positive_diff(m_f, m_s + m_d) / prime_sigma;

        float diff_f = lamda_StoFD / 2.0f * m_s + lamda_DtoF * m_d - lamda_FtoD * m_f;
        float diff_s = -lamda_StoFD * m_s;
        float diff_d = lamda_FtoD * m_f + lamda_StoFD / 2.0f * m_s - lamda_DtoF * m_d;

        m_f += diff_f;
        m_s += diff_s;
        m_d += diff_d;
        
        /////////////// update step2: kernel inference ///////////////
        credit = calc_credit(m_s + m_d, mass_scale);
        float beta = (1.0f - credit) + credit * calc_ratio(m_s, m_d);
        grid_cell.static_mass = m_s + beta * del_occ;
        grid_cell.dynamic_mass = m_d + (1.0f - beta) * del_occ;
        grid_cell.free_mass = m_f + del_free;
    }
    assert(grid_cell.static_mass >= 0.0f && grid_cell.dynamic_mass >= 0.0f && grid_cell.free_mass >= 0.0f);  // include checking nan
}

__device__ float separate_newborn_part(float m_dyn_pred, float m_total_pred, float m_dyn_up, float p_B)
{
    if (m_dyn_pred <= 0.0f) {   // (0,0) case is included here
        return m_dyn_up;
    } else if (m_total_pred <= m_dyn_pred) {
        return 0.0f;
    } else {
        return (m_dyn_up * p_B * (m_total_pred - m_dyn_pred)) / (m_dyn_pred + p_B * (m_total_pred - m_dyn_pred));
    }
}

// TODO: change poincloud data structure parameter in the function
__global__ void gridCellPredictionUpdateKernel(
    GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
    float* __restrict__ born_masses_array, float p_B, int cell_count,
    const float* __restrict__ meas_array_x, const float* __restrict__ meas_array_y, const float* __restrict__ meas_array_z,
    const int* __restrict__ pc_idx_arr, int meas_len, int meas_slot,
    const float* __restrict__ free_array_x, const float* __restrict__ free_array_y, const float* __restrict__ free_array_z,
    const int* __restrict__ free_idx_arr,  const int* __restrict__ source_beam_idx, int free_len, int free_slot,
    int grid_size, int grid_size_z, float resolution,
    float sigma, float ls, float gamma_pow, float mass_scale,
    float sensor_x, float sensor_y, float sensor_z)
    {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
        {
            int start_idx = grid_cell_array[i].start_idx;
            int end_idx = grid_cell_array[i].end_idx;

            float pred_total = (grid_cell_array[i].free_mass + grid_cell_array[i].dynamic_mass) * gamma_pow;
            float dynamic_mass_pred = 0.0f;
            if (start_idx != -1)
            {
                for (int j = start_idx; j < end_idx + 1; j++) {
                    dynamic_mass_pred += particle_array.weight[j];
                }
                assert(dynamic_mass_pred >= 0.0f);
            }
            predict_mass(grid_cell_array[i], dynamic_mass_pred, pred_total, gamma_pow, mass_scale);
            
            // 실험
            dynamic_mass_pred = grid_cell_array[i].dynamic_mass;

            //////// TODO: change poincloud data structure parameter in the function ///////////////////////////////////////
            update_mass(grid_cell_array[i], i, meas_array_x, meas_array_y, meas_array_z, pc_idx_arr, meas_len, meas_slot
            ,free_array_x, free_array_y, free_array_z,free_idx_arr, source_beam_idx, free_len, free_slot, grid_size, grid_size_z, 
                        resolution, sigma, ls, mass_scale, sensor_x, sensor_y, sensor_z);
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            float z_idx = i / (grid_size * grid_size) + 0.5f;
            float z = (z_idx - (float)grid_size_z / 2.0f) * resolution - sensor_z;

            // if (z < 0.10f) { // remove new born particle at bottom
            //     born_masses_array[i] = 0.0f;
            //     grid_cell_array[i].pers_mass = grid_cell_array[i].dynamic_mass;}
            // else {
            //     born_masses_array[i] = separate_newborn_part(dynamic_mass_pred, pred_total, grid_cell_array[i].dynamic_mass, p_B);
            //     grid_cell_array[i].pers_mass = grid_cell_array[i].dynamic_mass - born_masses_array[i];
            // }
            born_masses_array[i] = separate_newborn_part(dynamic_mass_pred, pred_total, grid_cell_array[i].dynamic_mass, p_B);
            grid_cell_array[i].pers_mass = grid_cell_array[i].dynamic_mass - born_masses_array[i];
            grid_cell_array[i].pred_mass = dynamic_mass_pred;
        }
    }
} /* namespace dom */
