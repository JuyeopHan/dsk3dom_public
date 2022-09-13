#pragma once

#include "common/cuda_utils.h"
#include <glm/vec3.hpp>

namespace dom{
struct ClusterSetCuda
{
    glm::vec3* bbox_min;
    glm::vec3* bbox_max;
    glm::vec3* vel;
    int size;
    // bool device; we do not set this since we suppose device == true

    ClusterSetCuda() : size(0) {}

    ClusterSetCuda(std::vector<glm::vec3*> arrays, int size_new) {init(arrays, size_new);};
    
    void init(std::vector<glm::vec3*> arrays, int size_new){
        size = size_new;

        // mem allocation
        CHECK_ERROR(cudaMalloc((void**)&bbox_min , size * sizeof(glm::vec3)));
        CHECK_ERROR(cudaMalloc((void**)&bbox_max , size * sizeof(glm::vec3)));
        CHECK_ERROR(cudaMalloc((void**)&vel , size * sizeof(glm::vec3)));

        if (size != 0){
            // memcpy
            CHECK_ERROR(cudaMemcpy(bbox_min ,arrays[0], size * sizeof(glm::vec3), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(bbox_max ,arrays[1], size * sizeof(glm::vec3), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(vel ,arrays[2], size * sizeof(glm::vec3), cudaMemcpyHostToDevice));
        }
    }

    void free(){
        CHECK_ERROR(cudaFree(bbox_min));
        CHECK_ERROR(cudaFree(bbox_max));
        CHECK_ERROR(cudaFree(vel));
    }
};
}