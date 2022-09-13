#pragma once

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include "common/mappoint.h"

namespace dom {

    /// PCL PointCloud types as input
typedef pcl::PointXYZ PCLPointType;
typedef pcl::PointCloud<PCLPointType> PCLPointCloud;
typedef pcl::PointCloud<MapPoint> MapPointCloud;

PCLPointCloud::Ptr downSample(const PCLPointCloud in, float ds_resolution);
    // distance between sensor - point cloud
float getSensorPointDist(float x, float y, float z);
    // simple heuristics for preventing excessive free points
int getDivideNumber(float length);
    // store free point and its source beam idx into 'free_point_vector' 
void getFreePoints(
        std::vector<float>& free_x,
        std::vector<float>& free_y,
        std::vector<float>& free_z,
        std::vector<int>& source_beam_vec,
        std::vector<std::vector<float>>& free_point_vector,
        const pcl::PointXYZ& point,
        const float& length,
        const float& dist,
        const int& idx);
    // remove floor points at a point cloud
PCLPointCloud removeGroundPts(const PCLPointCloud& in);
    // remove wall points at a point cloud
PCLPointCloud::Ptr removeWallPts(PCLPointCloud::Ptr in);
    // cluster extraction
std::vector<PCLPointCloud::Ptr> extractCluster(PCLPointCloud::Ptr in);
} // namespace dom