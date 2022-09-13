#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include "common/mappoint.h"

#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include "common/pcl_ftns.h"

namespace dom {

PCLPointCloud::Ptr downSample(const PCLPointCloud in, float ds_resolution)
    {
        PCLPointCloud::Ptr out(new PCLPointCloud);
        PCLPointCloud::Ptr pcl_in(new PCLPointCloud(in));

        pcl::VoxelGrid<PCLPointType> sor;
        sor.setInputCloud(pcl_in);
        sor.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        sor.filter(*out);

        return out;
    }

    // distance between sensor - point cloud
float getSensorPointDist(float x, float y, float z) { return sqrt(x*x + y*y + z*z); }

    // simple heuristics for preventing excessive free points
int getDivideNumber(float length){return std::max(2, (int)(18.0/length));}

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
        const int& idx)
    {
        float fp_x, fp_y, fp_z;
        fp_x = length / dist * point.x;
        fp_y = length / dist * point.y;
        fp_z = length / dist * point.z;
        free_x.push_back(fp_x);
        free_y.push_back(fp_y);
        free_z.push_back(fp_z);
        source_beam_vec.push_back(idx);
        free_point_vector.push_back({fp_x, fp_y, fp_z});

    }

//caution: do not sure which z-level is a ground critera    
PCLPointCloud removeGroundPts(const PCLPointCloud& in){
    PCLPointCloud::Ptr cloud (new PCLPointCloud(in));
    PCLPointCloud::Ptr cloud_filtered (new PCLPointCloud);

    // generate object
    pcl::PassThrough<PCLPointType> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0f, std::numeric_limits<float>::max());
    pass.filter (*cloud_filtered);

    return *cloud_filtered;
}

PCLPointCloud::Ptr removeWallPts(PCLPointCloud::Ptr in){
    PCLPointCloud::Ptr cloud (new PCLPointCloud(*in));
    PCLPointCloud::Ptr cloud_f (new PCLPointCloud);

    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentation<PCLPointType> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    PCLPointCloud::Ptr cloud_plane (new PCLPointCloud ());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.02);

    int nr_points = (int) cloud->size ();

    while(cloud->size() > 0.20 * nr_points){
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size() == 0){
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<PCLPointType> extract;
        extract.setInputCloud (cloud);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        *cloud = *cloud_f;
    }
    return cloud;
}

std::vector<PCLPointCloud::Ptr> extractCluster(PCLPointCloud::Ptr in){
    pcl::search::KdTree<PCLPointType>::Ptr tree (new pcl::search::KdTree<PCLPointType>);
    tree->setInputCloud (in);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (1.5f);
    ec.setMinClusterSize (10);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (in);
    ec.extract (cluster_indices);

    std::vector<PCLPointCloud::Ptr> output;

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : it->indices)
            cloud_cluster->push_back ((*in)[idx]);
            cloud_cluster->width = cloud_cluster->size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;

            output.push_back(cloud_cluster);
    }

    return output;
}

}
