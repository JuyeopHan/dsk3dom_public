#pragma once

#include <device_launch_parameters.h>

namespace dom
{
    const int max_queue_size = 150;
    const int max_length = 100;

struct pseudo_vector
{
    int vector[max_length];
    int length;
};

class CircularQueue {

    private:
    
    int queue_level[max_queue_size]; // queue contains level
    int queue_idx[max_queue_size]; // queue contains idx
    int front = 0;
    int rear = 0;

    public:
    inline __device__ bool isEmpty() { if (front == rear) { return true; } else { return false; } }
    inline __device__ bool isFull() { if (front == (rear + 1) % max_queue_size) { return true;} else {return false;} }
    inline __device__ void enQueue(int level, int idx)
    {
        if (isFull()) {return;
        } else {
            rear = (rear + 1) % max_queue_size;
            queue_level[rear] = level;
            queue_idx[rear] = idx;
        }
    }
    inline __device__ int2 deQueue()
    {
        if(isEmpty()) {
            int2 arr = make_int2(-1, -1);
            return arr;
        } else {
            front = (front + 1) % max_queue_size;
            int2 arr = make_int2(queue_level[front], queue_idx[front]);
            return arr;
        }
    }
};

inline __device__ bool isEmpty(const int* __restrict__ idx_arr ,const int idx) { return idx_arr[idx] < 0;}
inline __device__  bool isInvalid(const int idx, const int slot) {return idx >= slot;}
inline __device__  int left_child_idx(const int idx){return 2*idx + 1;}
inline __device__  int right_child_idx(const int idx){return 2*idx + 2;}

inline __device__ float dist2(float x1, float y1, float z1, 
    float x2, float y2, float z2)
{
    float distc = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
    return distc;
}

inline __device__ bool searchKeyInArr(int* arr, int arr_capacity, int key){
    
    for (int i = 0; i < arr_capacity; i++){
        if(arr[i] == key){
            return true;
        }
    }
    return false;
}

/* 
 * neighborhood_indices_ : k-d tree range search in CUDA w/o recursive function 
 * x_coord, y_coord, z_coord : (occupied or free) point data array for each coordinate
 * idx_arr : kd tree array indicated indices of point arrays
 * length : the length of coordinate arrays
 * slot : the total length of idx_arr
 * x_pos, y_pos, z_pos : query point coodinate for each coordinate
 * rad : radius for range search
 */

inline __device__ pseudo_vector neighborhoodIndices(
    const float* __restrict__ x_coord,
    const float* __restrict__ y_coord,
    const float* __restrict__ z_coord,
    const int* __restrict__ idx_arr,
    int length, int slot,
    float x_pos,
    float y_pos,
    float z_pos,
    float rad)
{

    // initializing variables and parameters
    /*
    level : 0 : x-axis 1 : y-axis, 2: z-axis
    */
    float d, dx, dx2, level_pos, level_pos_branch, r2 = rad * rad;
    int branch_idx = 0, dim = 3, level = 0, section_idx, other_idx;
    //int k = 0;
    pseudo_vector nbh;
    int2 qval;
    nbh.length = 0;
    

    // queue implementation instead of recursive function
    CircularQueue circualr_queue;
    circualr_queue.enQueue(level, branch_idx);

    while (!circualr_queue.isEmpty()){

        qval = circualr_queue.deQueue();
        level = qval.x;
        branch_idx = qval.y;

        if ((idx_arr[branch_idx] >= length) || (nbh.length >= max_length)){
            break;
        }

        float x_pos_branch = x_coord[idx_arr[branch_idx]];
        float y_pos_branch = y_coord[idx_arr[branch_idx]];
        float z_pos_branch = z_coord[idx_arr[branch_idx]];

        d = dist2(x_pos, y_pos, z_pos, x_pos_branch, y_pos_branch, z_pos_branch);

        switch (level)
        {
        case 0:
            level_pos = x_pos, level_pos_branch = x_pos_branch;
            break;

        case 1:
            level_pos = y_pos, level_pos_branch = y_pos_branch;
            break;

        case 2:
            level_pos = z_pos, level_pos_branch = z_pos_branch;
            break;  

        default:
            break;
        }

        dx = level_pos_branch - level_pos;
        dx2 = dx * dx;

        if ((d <= r2) && (nbh.length < max_length)) {
            nbh.vector[nbh.length] = branch_idx;
            nbh.length = nbh.length + 1;
        }

        if (dx > 0){
            section_idx = left_child_idx(branch_idx);
            other_idx = right_child_idx(branch_idx);
        } else {
            section_idx = right_child_idx(branch_idx);
            other_idx = left_child_idx(branch_idx);
        }


        if (!isInvalid(section_idx, slot) && !isEmpty(idx_arr, section_idx)){
            circualr_queue.enQueue((level + 1) % dim, section_idx);
        }

        if ((dx2 < r2) && !isInvalid(other_idx, slot) && !isEmpty(idx_arr, other_idx)) {
            circualr_queue.enQueue((level + 1) % dim, other_idx);
        }

    }

    return nbh;
}

}