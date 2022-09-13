#include <glm/vec3.hpp>
#include <utility>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "common/cluster_matching.h"
#include "common/munkres.h"
namespace dom {


std::vector<Cluster> buildClusters(const std::vector<PCLPointCloud::Ptr> cluster_pcls){
    std::vector<Cluster> clusters;
    for (const auto& cluster_pcl : cluster_pcls){
        Cluster cluster(cluster_pcl);
        clusters.push_back(cluster);
    }
    return clusters;
}

Cluster::Cluster(const PCLPointCloud::Ptr cluster_pcl){
    float num = 0.0;
    float x_mean = 0.0f, y_mean = 0.0f, z_mean = 0.0f;

    float x_min = 1e3;
    float y_min = 1e3;
    float z_min = 1e3;

    float x_max = -1e3;
    float y_max = -1e3;
    float z_max = -1e3;

    for (const auto& pt: *cluster_pcl){
        x_mean += pt.x;
        y_mean += pt.y;
        z_mean += pt.z;

        checkMinMax(x_min, x_max, pt.x);
        checkMinMax(y_min, y_max, pt.y);
        checkMinMax(z_min, z_max, pt.z);
        num += 1.0f;
    }
    
    x_mean /= num;
    y_mean /= num;
    z_mean /= num;

    x_min = std::min(x_mean - 0.4f, x_min);
    y_min = std::min(y_mean - 0.4f, y_min);
    z_min = std::min(z_mean - 0.4f, z_min);

    x_max = std::max(x_mean + 0.4f, x_max);
    y_max = std::max(y_mean + 0.4f, y_max);
    z_max = std::max(z_mean + 0.4f, z_max);

    glm::vec3 bbox_min = glm::vec3(x_min, y_min, z_min);
    glm::vec3 bbox_max = glm::vec3(x_max, y_max, z_max);
    bbox = std::make_pair(bbox_min, bbox_max);
    center = glm::vec3(x_mean, y_mean, z_mean);
}

void Cluster::checkMinMax(float& min, float& max, float pt_coord){
    if (min > pt_coord) min = pt_coord;
    if (max < pt_coord) max = pt_coord;
}

void MatchingMunkres::setValueMat(glm::vec3 sensor_pos, glm::vec3 sensor_pos_prev){
    int num_prev = clusters_prev.size();
    int num_curr = clusters.size();
    int row, col;
        if (num_prev >= num_curr){
        row = num_prev;
        col = num_prev; 
    } else {
        row = num_curr;
        col = num_curr;
    }

    float value_arr[row][col];
    std::fill(&value_arr[0][0], &value_arr[row-1][col], (-0.0f) * 1e2);
    int i = 0;
    for (const auto& cluster_prev : clusters_prev){
        int j = 0;
        for (const auto& cluster : clusters){
            value_arr[i][j] 
            = (1.0f) * sqrt((cluster_prev.center.x + sensor_pos_prev.x - cluster.center.x - sensor_pos.x) * (cluster_prev.center.x + sensor_pos_prev.x - cluster.center.x - sensor_pos.x) +
            (cluster_prev.center.y + sensor_pos_prev.y - cluster.center.y - sensor_pos.y) * (cluster_prev.center.y + sensor_pos_prev.y - cluster.center.y - sensor_pos.y) + 
            (cluster_prev.center.z + sensor_pos_prev.z - cluster.center.z - sensor_pos.z) * (cluster_prev.center.z + sensor_pos_prev.z - cluster.center.z - sensor_pos.z));
        ++j;
        }
        ++i;
    }

    for (int i = 0; i < row; ++i){
        for (int j = 0; j < col; ++j){
            value_vec.push_back(value_arr[i][j]);
        }
    }

    row_mat = (unsigned) row;
    col_mat = (unsigned) col;
}

std::vector<std::pair<unsigned, unsigned>> MatchingMunkres::getMatch(){

    auto f = [&](unsigned r, unsigned c) {return value_vec[r * col_mat + c];};
    auto matching = munkres_algorithm<float>(row_mat, col_mat, f);
    return matching;
}

void MatchingMunkres::setVel(const float dt, glm::vec3 sensor_pos, glm::vec3 sensor_pos_prev){
    int num_prev = clusters_prev.size();
    int num = clusters.size();
    std::vector<std::pair<unsigned, unsigned>> matching = getMatch();

    std::vector<std::pair<unsigned, glm::vec3>> cluster_vels;
    for (const auto& pair: matching){
        unsigned idx_prev = pair.first;
        unsigned idx = pair.second;

        if (idx_prev >= num_prev || idx >= num) {continue;}
        else {
            float vel_x = (clusters[idx].center.x + sensor_pos.x - clusters_prev[idx_prev].center.x - sensor_pos_prev.x) / dt;
            float vel_y = (clusters[idx].center.y + sensor_pos.y - clusters_prev[idx_prev].center.y - sensor_pos_prev.y) / dt;
            float vel_z = (clusters[idx].center.z + sensor_pos.z - clusters_prev[idx_prev].center.z - sensor_pos_prev.z) / dt;
            if (abs(vel_x) > vel_max || abs(vel_y) > vel_max || abs(vel_z) > vel_max){continue;}
            else {
                glm::vec3 vel(vel_x, vel_y, vel_z);
                std::pair<unsigned, glm::vec3> cluster_vel = std::make_pair(idx, vel);
                cluster_vels.push_back(cluster_vel);
            }
        }
    }
    cluster_vel_pair = cluster_vels;
}


} /* namespace dom*/