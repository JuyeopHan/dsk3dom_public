#pragma once

#include <utility>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <glm/vec3.hpp>

namespace dom{

const float vel_max = 5.0f;

    /// PCL PointCloud types as input
typedef pcl::PointXYZ PCLPointType;
typedef pcl::PointCloud<PCLPointType> PCLPointCloud;

class Cluster
{
public:
    glm::vec3 center;
    std::pair<glm::vec3, glm::vec3> bbox;

    Cluster(const PCLPointCloud::Ptr cluster_pcl);
    ~Cluster() {}

private:
    void checkMinMax(float& min, float& max, float pt_coord);

};

std::vector<Cluster> buildClusters(const std::vector<PCLPointCloud::Ptr> cluster_pcls);

class MatchingMunkres
{
public:
    std::vector<Cluster> clusters_prev;
    std::vector<Cluster> clusters;
    std::vector<std::pair<unsigned, glm::vec3>> cluster_vel_pair;

private:
    std::vector<float> value_vec;
    unsigned row_mat;
    unsigned col_mat;

public:
    MatchingMunkres(const std::vector<Cluster> clusters_prev, const std::vector<Cluster> clusters, const float dt, glm::vec3 sensor_pos, glm::vec3 sensor_pos_prev) :
    clusters_prev(clusters_prev), clusters(clusters) { setValueMat(sensor_pos, sensor_pos_prev); setVel(dt, sensor_pos, sensor_pos_prev); }
    ~MatchingMunkres() {};

private:
    void setValueMat(glm::vec3 sensor_pos, glm::vec3 sensor_pos_prev);
    void setVel(const float dt, glm::vec3 sensor_pos, glm::vec3 sensor_pos_prev);
    std::vector<std::pair<unsigned, unsigned>> getMatch();
};

struct ClusterSet
{
    std::vector<glm::vec3*> arrays;
    int size;

    ClusterSet() : size(0) {}

    ClusterSet(MatchingMunkres matching_munkres) {init(matching_munkres);};
    
    void init(MatchingMunkres matching_munkres){
        size = matching_munkres.cluster_vel_pair.size();

        if (size != 0){
            // generate array
            arrays = genArray(matching_munkres);
        }
    };

    std::vector<glm::vec3*> genArray(MatchingMunkres matching_munkres){
        
        std::vector<glm::vec3*> output;

        if (size != 0){
            glm::vec3 bbox_min_set[size];
            glm::vec3 bbox_max_set[size];
            glm::vec3 vel_set[size];
            int i = 0;
            for (const auto& pair : matching_munkres.cluster_vel_pair){
                unsigned idx = pair.first;
                std::pair<glm::vec3, glm::vec3> bbox = matching_munkres.clusters[idx].bbox;

                glm::vec3 bbox_min_ = bbox.first;
                glm::vec3 bbox_max_ = bbox.second;
                glm::vec3 vel_ = pair.second;

                bbox_min_set[i] = bbox_min_;
                bbox_max_set[i] = bbox_max_;
                vel_set[i] = vel_;

                i++;
            }

            output.push_back(bbox_min_set);
            output.push_back(bbox_max_set);
            output.push_back(vel_set);
        }
        
        return output;
    }
};

} /* namespace dom */