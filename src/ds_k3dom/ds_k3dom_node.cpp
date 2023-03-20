#include "ds_k3dom/dom.h"
#include "common/KDTree.h"
#include "common/dom_types.h"

#include "common/markerarray_pub.h"
#include "common/mappoint.h"
#include "common/pcl_ftns.h"
#include "common/cluster_matching.h"
#include "common/cluster_cuda.h"

#include <memory>
#include <string>
#include <cmath>
#include <ctime>
#include <chrono>

#include <glm/vec3.hpp>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// TEST
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

namespace dom 
{
/// PCL PointCloud types as input
typedef pcl::PointXYZ PCLPointType;
typedef pcl::PointCloud<PCLPointType> PCLPointCloud;
typedef pcl::PointCloud<MapPoint> MapPointCloud;

class myMap {
public:
    myMap(std::string map_topic,
          std::string lidar_frame,
          float occupied_thresh,
          float valid_thresh,
          float min_z,
          float max_z,
          float min_var,
          float max_var,
          int map_pub_freq,
          float imu_pose_offset,
          bool do_eval,
          int map_vis_mode,
          ros::NodeHandle nh,
          DOM::Params grid_params)
    : grid_map(grid_params), lidar_frame(lidar_frame),
      m_pub(nh, map_topic, "lidar_traj", grid_params.resolution, "local_map", "global_map"),
      initial(true), br(), listener(), count(0), min_z(min_z), max_z(max_z), min_var(min_var), max_var(max_var), ds_resolution(grid_params.ds_resolution),
      map_pub_freq(map_pub_freq), imu_pose_offset(imu_pose_offset), do_eval(do_eval),
      msg_mapdata(new MapPointCloud), map_vis_mode(map_vis_mode),
      min_occ_threshold(occupied_thresh), min_mass_threshold(valid_thresh)
    {
        mapdata_pub = nh.advertise<MapPointCloud>("/map_data", 1);
        msg_mapdata->header.frame_id = "map_data_gazebo";
        msg_mapdata->height = 1;
    }
    
    ~myMap() {}
 
    void updateFromROSTopic(const PCLPointCloud::ConstPtr& msg_pc,
                              const geometry_msgs::PoseStamped::ConstPtr& msg_p)
    {
        ROS_INFO_STREAM("Well received pointcloud with " << msg_pc->width << " points");
        auto start1 = std::chrono::high_resolution_clock::now();
           
        if (initial) {
            initial = false;
        }

        // tf::Transform tf_lidar2lmap: to modify point clouds based on the robot frame
        tf::Transform tf_lidar2lmap = reflectNewPose(msg_p);

        // point cloud acquired from a lidar sensor
        PCLPointCloud cloud_in_map_pre;
        pcl_ros::transformPointCloud (*msg_pc, cloud_in_map_pre, tf_lidar2lmap.inverse());

        // downsample point clouds
        PCLPointCloud::Ptr cloud_in_map = downSample(cloud_in_map_pre, ds_resolution);
        ROS_INFO_STREAM("Filtered pointcloud with " << cloud_in_map->size() << " points");

        // free_dist : free points sample distance for each beams
        // should put this var into param later

        float free_dist = 1.0f;

        /*
         *
         * pc_vector : vector of point cloud 
         * pc_x, pc_y, pc_z : vector of each coordinate of point cloud
         * idx_arr : index array of point cloud
         * 
         * free_point_vector : vector of point cloud
         * free_x, free_y, free_z : vector of each coordinate of free points
         * source_beam_vec, source_beam_arr : array(vector) of source beam index of free points
         * free_idx_arr : index array of free points
         * 
         * idx, free_idx : index of point cloud (or free points)
         * 
         */
        
        std::vector<std::vector<float>> pc_vector;
        std::vector<float> pc_x, pc_y, pc_z;
        int idx = 0;
        std::vector<std::vector<float>> free_point_vector;
        std::vector<float> free_x, free_y, free_z;
        std::vector<int> source_beam_vec;

        for (const auto& point : *cloud_in_map) {
            
            // Add Occupied Point

            pc_x.push_back(point.x);
            pc_y.push_back(point.y);
            pc_z.push_back(point.z);
            pc_vector.push_back({point.x, point.y, point.z});
            
            
            float dist = getSensorPointDist(point.x, point.y, point.z); 

            // Add Free Point Array
            if ((dist > free_dist)){
                for (float length = free_dist; length < dist; length += free_dist){
                        int num = getDivideNumber(length);
                        if ((idx % num == 0)){
                            getFreePoints(free_x, free_y, free_z, source_beam_vec, free_point_vector, point,length, dist, idx);
                    }
                }
            } 
            else if ((dist < free_dist) && (free_dist/2 < dist)) {
                float length = free_dist - dist;
                getFreePoints(free_x, free_y, free_z, source_beam_vec, free_point_vector, point, length, dist, idx);
            }
            idx++;

            
        }

        float* pc_arr_x = &pc_x[0];
        float* pc_arr_y = &pc_y[0];
        float* pc_arr_z = &pc_z[0];

        float* free_arr_x = &free_x[0];
        float* free_arr_y = &free_y[0];
        float* free_arr_z = &free_z[0];
        int* source_beam_arr = &source_beam_vec[0]; 

        //make k-d tree (array) of point cloud and free points
        KDTree kdtree_pc(pc_vector);
        KDTree kdtree_free_point(free_point_vector);

        KDTreeArr kdtree_arr_pc(kdtree_pc, pc_arr_x, pc_arr_y, pc_arr_z);
        KDTreeArr kdtree_arr_free(kdtree_free_point, free_arr_x, free_arr_y, free_arr_z);

        // update occupancy grid
        grid_map.updateGrid(msg_p->header.stamp.toSec(), kdtree_arr_pc, kdtree_arr_free, source_beam_arr);

        auto stop1 = std::chrono::high_resolution_clock::now();
        auto timespan1 = std::chrono::duration<double>(stop1 - start1);
        ROS_INFO_STREAM("free " << free_z.size() << " points.");
        ROS_INFO_STREAM("it took " << timespan1.count() << " seconds.");
       
        // evalutation
        if (do_eval) {
            pcl_conversions::toPCL(msg_p->header.stamp, msg_mapdata->header.stamp);
            publish_mapdata();
        }
        
        if (map_vis_mode != -1 && ++count % map_pub_freq == 0) {
            m_pub.update_time(msg_p->header.stamp);
            publish_map();
        }
    }

    tf::Transform reflectNewPose(const geometry_msgs::PoseStamped::ConstPtr& msg_p)
    {
        // gmap: global map
        // lmap: local map
        tf::Transform tf_gmap2base;
        tf_gmap2base.setOrigin(tf::Vector3(msg_p->pose.position.x, msg_p->pose.position.y,
                                            msg_p->pose.position.z));
        tf::Quaternion orient;
        quaternionMsgToTF(msg_p->pose.orientation, orient);
        tf::Quaternion offset;  // offset between "imu" frame and pose estimation
        offset.setRPY(0,0,imu_pose_offset);
        tf_gmap2base.setRotation(offset*orient);
        tf::Transform tf_base2gmap(tf_gmap2base.inverse());
        br.sendTransform(tf::StampedTransform(tf_base2gmap, msg_p->header.stamp, "base_link", "global_map"));
        
        tf::StampedTransform tf_lidar2base;
        listener.lookupTransform(lidar_frame, "base_link", msg_p->header.stamp, tf_lidar2base);
        tf::Transform tf_lidar2gmap = tf_base2gmap * tf_lidar2base;
        tf::Transform tf_lidar2lmap = tf_lidar2gmap;
        tf_lidar2lmap.setOrigin(tf::Vector3(0, 0, 0)); // local map with fixed orientation
        br.sendTransform(tf::StampedTransform(tf_lidar2lmap, msg_p->header.stamp, lidar_frame, "local_map"));
        
        auto tmp = tf_lidar2gmap.inverse().getOrigin();
        grid_map.updatePose(tmp.x(),tmp.y(),tmp.z());

        m_pub.insert_trajectory(msg_p);

        return tf_lidar2lmap;
    }

    void publish_map() {
        if (!do_eval) { // else, done in publish_mapdata
            CHECK_ERROR(cudaMemcpy(grid_map.grid_cell_array_host, grid_map.grid_cell_array, 
                                    grid_map.grid_cell_count * sizeof(GridCell), cudaMemcpyDeviceToHost));
        }
        
        float x, y, z;
        for (int i = 0; i < grid_map.grid_cell_count; ++i) {
            if (grid_map.grid_cell_array_host[i].total_belief() > min_mass_threshold && //
                grid_map.grid_cell_array_host[i].occ_belief() > min_occ_threshold) {
                x = grid_map.indexToCoordX(i) - grid_map.getSensorPositionX();
                y = grid_map.indexToCoordY(i) - grid_map.getSensorPositionY();
                z = grid_map.indexToCoordZ(i) - grid_map.getSensorPositionZ();

                if (map_vis_mode == 0) {    // classification result
                    m_pub.insert_point3d_class(x, y, z,
                                                grid_map.grid_cell_array_host[i].free_mass,
                                                grid_map.grid_cell_array_host[i].static_mass,
                                                grid_map.grid_cell_array_host[i].dynamic_mass,
                                                grid_map.grid_cell_array_host[i].occ_mass);
                }
                else if (map_vis_mode == 1) {   // height map
                    m_pub.insert_point3d_height(x, y, z, min_z, max_z, z + grid_map.getSensorPositionZ());
                }else { // variance map
                    m_pub.insert_point3d_var(x, y, z, 
                                                grid_map.grid_cell_array_host[i].free_mass,
                                                grid_map.grid_cell_array_host[i].static_mass,
                                                grid_map.grid_cell_array_host[i].dynamic_mass,
                                                grid_map.grid_cell_array_host[i].occ_mass);
                }
            }
        }
        m_pub.publish();
        ROS_INFO("Map published");
        m_pub.clear();
    }

    void publish_mapdata() {
            CHECK_ERROR(cudaMemcpy(grid_map.grid_cell_array_host, grid_map.grid_cell_array, 
                                    grid_map.grid_cell_count * sizeof(GridCell), cudaMemcpyDeviceToHost));
            float res = grid_map.getResolution();
            MapPoint pt;
            for (int i = 0; i < grid_map.grid_cell_count; ++i) {
                pt.x = grid_map.indexToCoordX(i);
                pt.y = grid_map.indexToCoordY(i);
                pt.z = grid_map.indexToCoordZ(i);
                pt.v_x = grid_map.grid_cell_array_host[i].mean_x_vel * res;
                pt.v_y = grid_map.grid_cell_array_host[i].mean_y_vel * res;
                pt.v_z = grid_map.grid_cell_array_host[i].mean_z_vel * res;
                pt.occ_val = grid_map.grid_cell_array_host[i].occ_belief();
                pt.dyn_val = grid_map.grid_cell_array_host[i].dyn_belief();
                pt.eval_aug = grid_map.grid_cell_array_host[i].total_belief();
                pt.dyn_aug = grid_map.grid_cell_array_host[i].pers_mass;

                msg_mapdata->points.push_back(pt);                
            }
            msg_mapdata->width = grid_map.grid_cell_count;
            mapdata_pub.publish(msg_mapdata);
            msg_mapdata->points.clear();
            ROS_INFO("Map data published");
        }

public:
    DOM grid_map;
    MarkerArrayPub m_pub;
    std::string lidar_frame;
    bool initial;
    tf::TransformBroadcaster br;
    tf::TransformListener listener;
    int count;
    float min_occ_threshold;
    float min_mass_threshold;
    float min_z;
    float max_z;
    float min_var;
    float max_var;
    float ds_resolution;
    int map_pub_freq;
    float imu_pose_offset;
    bool do_eval;
    int map_vis_mode;
    ros::Publisher mapdata_pub;
    MapPointCloud::Ptr msg_mapdata;
    
};
};
 

int main(int argc, char** argv)
{
    ros::init(argc, argv, "k3dom_demo_node");
    ros::NodeHandle nh("~");

    std::string map_topic("/occupied_cells_vis_array");
    std::string lidar_topic("/velodyne_points");
    std::string pose_topic("/pose");
    std::string lidar_frame("velodyne_left");

    // default parameters
    float resolution = 0.1f; 
    float ds_resolution = 0.025f; // Downsampling
    float free_thresh = 0.3f;
    float occupied_thresh = 0.7f;
    float valid_thresh = 1.0f; // Threshold on sum of masses to distinguish known/unknown
    float min_z = 0.0f;
    float max_z = 0.0f;
    float min_var = 0.0f;
    float max_var = 0.0f;
    int map_pub_freq = 5; // map publishing frequency for Rviz
    float imu_pose_offset = 0.0f;
    float size = 10.0f;
    float size_z = 5.0f;
    float mass_scale = 3.0f;
    bool do_eval = false; // whether publish evaluation
    int map_vis_mode = 0; // choose how to visualize(color) the map
    // -1: not visualize / 0: dynamic classification / 1: height map / 2: variance map

    // sensor_off_.. : initial sensor offset related to the local map frame 
    float sensor_off_x = 0.0f;
    float sensor_off_y = 0.0f;
    float sensor_off_z = 0.0f;
    float map_shift_thresh = 1.0f; // shift local map when sensor is far from its default position

    // parameters for particles
    float particle_count = 2 * static_cast<int>(10e6); // 2 or 3 millions
    float new_born_particle_count = 3 * static_cast<int>(10e5);
    float persistence_prob = 0.99f;
    float stddev_process_noise_position = 0.1f; // [m] per dimension
    float stddev_process_noise_velocity = 1.0f; // [m/s] per dimension
    float birth_prob = 0.02f;
    float stddev_velocity = 1.0f; // [m/s] per dimension for birth distribution (not used now)
    float init_max_velocity = 5.0f; // [m/s] per dimension
    float particle_min_vel = 1.0f; // [m/s]

    // parameters for kernel & dirichlets
    float sigma = 1.0; // Actually sigma_0 in sparse kernel
    float ls = 1.0; //  Length scale of the sparse kernel

    float gamma = 0.99f; // decaying factor for prediction update
    float beta = 0.9f;

    // initial concentration parameter
    float prior_free = 1.0f; // unoccupied
    float prior_static = 1.0f; // occupied_static
    float prior_dynamic = 1.0f; // occupied_dynamic
    float prior_occ = 0.0f; // occupied
    float prior_all = 0.001f; // all states


    // calling parameters from .yaml  file (./config/...)
    nh.param<std::string>("map_topic", map_topic, map_topic);
    nh.param<std::string>("lidar_topic", lidar_topic, lidar_topic);
    nh.param<std::string>("pose_topic", pose_topic, pose_topic);
    nh.param<std::string>("lidar_frame", lidar_frame, lidar_frame);
    nh.param<float>("resolution", resolution, resolution);
    nh.param<float>("ds_resolution", ds_resolution, ds_resolution);
    nh.param<float>("free_thresh", free_thresh, free_thresh);
    nh.param<float>("occupied_thresh", occupied_thresh, occupied_thresh);
    nh.param<float>("valid_thresh", valid_thresh, valid_thresh);
    nh.param<float>("min_z", min_z, min_z);
    nh.param<float>("max_z", max_z, max_z);
    nh.param<float>("min_var", min_var, min_var);
    nh.param<float>("max_var", max_var, max_var);
    nh.param<int>("map_pub_freq", map_pub_freq, map_pub_freq);
    nh.param<float>("imu_pose_offset", imu_pose_offset, imu_pose_offset);
    nh.param<float>("size", size, size);
    nh.param<float>("size_z", size_z, size_z);
    nh.param<float>("mass_scale", mass_scale, mass_scale);
    nh.param<bool>("do_eval", do_eval, do_eval);
    nh.param<int>("map_vis_mode", map_vis_mode, map_vis_mode);
    nh.param<float>("sensor_off_x", sensor_off_x, sensor_off_x);
    nh.param<float>("sensor_off_y", sensor_off_y, sensor_off_y);
    nh.param<float>("sensor_off_z", sensor_off_z, sensor_off_z);
    nh.param<float>("map_shift_thresh", map_shift_thresh, map_shift_thresh);
    nh.param<float>("particle_count", particle_count, particle_count);
    nh.param<float>("new_born_particle_count", new_born_particle_count, new_born_particle_count);
    nh.param<float>("persistence_prob", persistence_prob, persistence_prob);
    nh.param<float>("stddev_process_noise_position", stddev_process_noise_position, stddev_process_noise_position);
    nh.param<float>("stddev_process_noise_velocity", stddev_process_noise_velocity, stddev_process_noise_velocity);
    nh.param<float>("birth_prob", birth_prob, birth_prob);
    nh.param<float>("stddev_velocity", stddev_velocity, stddev_velocity);
    nh.param<float>("init_max_velocity", init_max_velocity, init_max_velocity);
    nh.param<float>("particle_min_vel", particle_min_vel, particle_min_vel);
    nh.param<float>("sigma", sigma, sigma);
    nh.param<float>("ls", ls, ls);
    nh.param<float>("gamma", gamma, gamma);
    nh.param<float>("beta", beta, beta);
    nh.param<float>("prior_free", prior_free, prior_free);
    nh.param<float>("prior_static", prior_static, prior_static);
    nh.param<float>("prior_dynamic", prior_dynamic, prior_dynamic);
    nh.param<float>("prior_occ", prior_occ, prior_occ);
    nh.param<float>("prior_all", prior_all, prior_all);

    ROS_INFO_STREAM("Parameters:" << std::endl <<
            "map_topic: " << map_topic << std::endl <<
            "lidar_topic: " << lidar_topic << std::endl <<
            "pose_topic: " << pose_topic << std::endl <<
            "lidar_frame: " << lidar_frame << std::endl <<
            "resolution: " << resolution << std::endl <<
            "ds_resolution: " << ds_resolution << std::endl <<
            "free_thresh: " << free_thresh << std::endl <<
            "occupied_thresh: " << occupied_thresh << std::endl <<
            "valid_thresh: " << valid_thresh << std::endl <<
            "min_z: " << min_z << std::endl <<
            "max_z: " << max_z << std::endl <<
            "min_var: " << min_var << std::endl <<
            "max_var: " << max_var << std::endl <<
            "map_pub_freq: " << map_pub_freq << std::endl <<
            "imu_pose_offset: " << imu_pose_offset << std::endl <<
            "size: " << size << std::endl <<
            "size_z: " << size_z << std::endl <<
            "mass_scale: " << mass_scale << std::endl <<
            "do_eval: " << do_eval << std::endl <<
            //"particle_initialize: " << particle_initialize << std::endl <<
            "map_vis_mode: " << map_vis_mode << std::endl <<
            "sensor_off_x: " << sensor_off_x << std::endl <<
            "sensor_off_y: " << sensor_off_y << std::endl <<
            "sensor_off_z: " << sensor_off_z << std::endl <<
            "map_shift_thresh: " << map_shift_thresh << std::endl <<
            "particle_count: " << particle_count << std::endl <<
            "new_born_particle_count: " << new_born_particle_count << std::endl <<
            "persistence_prob: " << persistence_prob << std::endl <<
            "stddev_process_noise_position: " << stddev_process_noise_position << std::endl <<
            "stddev_process_noise_velocity: " << stddev_process_noise_velocity << std::endl <<
            "birth_prob: " << birth_prob << std::endl <<
            "stddev_velocity: " << stddev_velocity << std::endl <<
            "init_max_velocity: " << init_max_velocity << std::endl <<
            "particle_min_vel: " << particle_min_vel << std::endl <<
            "sigma: " << sigma << std::endl <<
            "ls: " << ls << std::endl <<
            "gamma: " << gamma << std::endl <<
            "beta: " << beta << std::endl <<
            "prior_free: " << prior_free << std::endl <<
            "prior_static: " << prior_static << std::endl <<
            "prior_dynamic: " << prior_dynamic << std::endl <<
            "prior_occ: " << prior_occ << std::endl <<
            "prior_all: " << prior_all
            );
    
    dom::DOM::Params grid_params;
    grid_params.ds_resolution = ds_resolution;
    grid_params.size = size;
    grid_params.size_z = size_z;
    grid_params.mass_scale = mass_scale;
    grid_params.resolution = resolution;
    grid_params.particle_count = particle_count;
    grid_params.new_born_particle_count = new_born_particle_count;
    grid_params.persistence_prob = persistence_prob;
    grid_params.stddev_process_noise_position = stddev_process_noise_position / resolution;  // in idx unit
    grid_params.stddev_process_noise_velocity = stddev_process_noise_velocity / resolution;  // in idx unit
    grid_params.birth_prob = birth_prob;
    grid_params.stddev_velocity = stddev_velocity / resolution; // in idx unit
    grid_params.init_max_velocity = init_max_velocity / resolution; // in idx unit
    grid_params.particle_min_vel = particle_min_vel / resolution;    // in idx unit
    grid_params.sigma = sigma;
    grid_params.ls = ls;
    grid_params.gamma = gamma;
    grid_params.beta = beta;
    grid_params.prior_free = prior_free;
    grid_params.prior_static = prior_static;
    grid_params.prior_dynamic = prior_dynamic;
    grid_params.prior_occ = prior_occ;
    grid_params.prior_all = prior_all;
    grid_params.sensor_off_x = sensor_off_x;
    grid_params.sensor_off_y = sensor_off_y;
    grid_params.sensor_off_z = sensor_off_z;
    grid_params.map_shift_thresh = static_cast<int>(map_shift_thresh / resolution);    // in idx unit

    // Just to init cuda
    cudaDeviceSynchronize();

    dom::myMap mymap(map_topic, lidar_frame, occupied_thresh, valid_thresh, min_z, max_z, min_var, max_var, map_pub_freq, imu_pose_offset, do_eval, map_vis_mode, nh, grid_params);
 
    message_filters::Subscriber<dom::PCLPointCloud> sub_lidar(nh, lidar_topic, 100);
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_pose(nh, pose_topic, 100);

    //  message_filters::sync_policies::ApproximateTime, message_filters::Synchronizer<mySyncPolicy>: match sync. of topic data
    
    typedef message_filters::sync_policies::ApproximateTime<dom::PCLPointCloud, geometry_msgs::PoseStamped> mySyncPolicy;  
    
    message_filters::Synchronizer<mySyncPolicy> sync(mySyncPolicy(100), sub_lidar, sub_pose);
    sync.registerCallback(boost::bind(&dom::myMap::updateFromROSTopic, &mymap, _1, _2));
    
    ros::spin();

    return 0;
}
