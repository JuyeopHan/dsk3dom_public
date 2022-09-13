#pragma once

#include "common/dom_types.h"
#include "common/dom.h"
#include "common/KDTree.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <memory>

#include <vector>

namespace dom
{

class DOM : public DOM_c
{
public:
    DOM(const Params& params);
    ~DOM();
    void updateGrid(float t, KDTreeArr& kdtree_arr_pc, KDTreeArr& kdtree_arr_free, int*& source_beam_arr);

    void initializeParticles(KDTreeArr& kdtree_arr_pc, KDTreeArr& kdtree_arr_free, int*& source_beam_arr);

    void gridCellOccupancyUpdate(float dt, KDTreeArr& kdtree_arr_pc, KDTreeArr& kdtree_arr_free, int*& source_beam_arr);
};

} /* namespace dom */
