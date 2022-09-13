#pragma once

#include "common/dom_types.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <memory>

#include <vector>

namespace dom
{

class DOM_c // 'c' indicates 'common'
{
public:
    struct Params
    {
        float size;
        float size_z;   // size for z is usually different from those for x,y dimensions
        float mass_scale;
        float resolution;
        int particle_count;
        int new_born_particle_count;
        float persistence_prob;
        float stddev_process_noise_position;
        float stddev_process_noise_velocity;
        float birth_prob;
        float stddev_velocity;
        float init_max_velocity;
        float particle_min_vel;
        float sensor_off_x;
        float sensor_off_y;
        float sensor_off_z;
        int map_shift_thresh;

        //downsampling resolution
        float ds_resolution;

        float prior_free;
        float prior_static;
        float prior_dynamic;
        float prior_occ;
        float prior_all; // only for ds-k3dom

        // only for ds-k3dom
        float alpha; // transfering factor for occupied mass -> alpha^(dt) * occ_mass

        // only for k3dom
        float sigma; // sigma_0 in sparse kernel
        float ls; // length scale of sparse kernel
        float gamma; // decaying factor for masses: mass -> gamma^(dt) * mass

        // only for ds-phd/mib (to generate meas_grid)
        float max_range;
        float meas_occ_stddev;
    };

    DOM_c(const Params& params);
    ~DOM_c();

    void updatePose(float new_x, float new_y, float new_z);

    int getGridSize() const { return grid_size; }
    float getResolution() const { return params.resolution; }

    float getCenterPositionX() const { return center_pos_x; }
    float getCenterPositionY() const { return center_pos_y; }
    float getCenterPositionZ() const { return center_pos_z; }

    float getSensorPositionX() const { return sensor_pos_x; }
    float getSensorPositionY() const { return sensor_pos_y; }
    float getSensorPositionZ() const { return sensor_pos_z; }

    float indexToCoordX(int i) {
        return (i % grid_size + 0.5f) * params.resolution - params.size/2.0f
                + getCenterPositionX();
    }
    float indexToCoordY(int i) {
        return ((i % (grid_size * grid_size)) / grid_size + 0.5f) * params.resolution
                - params.size/2.0f + getCenterPositionY();
    }
    float indexToCoordZ(int i) {
        return (i / (grid_size * grid_size) + 0.5f) * params.resolution
                - params.size_z/2.0f + getCenterPositionZ();
    }

    int getIteration() const { return iteration; }

private:
    void initialize();

public:
    void particlePrediction(float dt);
    void particleAssignment();
    void updatePersistentParticles();
    void initializeNewParticles();
    void statisticalMoments();
    void resampling();

public:
    Params params;

    GridCell* grid_cell_array;
    GridCell* grid_cell_array_host;
    ParticlesSoA particle_array;
    ParticlesSoA particle_array_next;
    ParticlesSoA particle_array_test;   // for debuging in host
    ParticlesSoA birth_particle_array;

    // point cloud for occupied meaurements
    float* meas_x;
    float* meas_y;
    float* meas_z;
    
    ///// added for K3DOM and DS-K3DOM /////
    int* pc_idx_arr;
    int* free_idx_arr;
    int* source_beam_arr;
    int* source_beam_idx;
    // point cloud for free meaurements
    float* free_x;
    float* free_y;
    float* free_z;
    ////////////////////////////////////////

    float* born_masses_array;

    float* vel_x_array;
    float* vel_y_array;
    float* vel_z_array;

    float* rand_array;

    curandState* rng_states;

    int grid_size;
    int grid_size_z;

    int grid_cell_count; // the number of all grid cells
    int particle_count; // the number of particles
    int new_born_particle_count; // the number of new borm particles
    int meas_len; // the number of occupied measurements

    ///// added for K3DOM & DS-K3DOM /////
    int free_len; // the number of free measurements
    int meas_slot; // the number of slot of k-d tree for occupied measurements
    int free_slot; // the number of slot of k-d tree for free measurements
    ////////////////////////////////////////
    
    // cuda kernel numbers
    dim3 block_dim;
    dim3 particles_grid;
    dim3 birth_particles_grid;
    dim3 grid_map_grid;
    
    float updated_time;
// private:
protected:
    int iteration;

    bool first_pose_received;
    bool first_measurement_received;
    float sensor_pos_x;   // sensor position
    float sensor_pos_y;
    float sensor_pos_z;

    float center_pos_x;    // map center position
    float center_pos_y;
    float center_pos_z;

    
};

} /* namespace dom */
