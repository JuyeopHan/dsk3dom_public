#pragma once

/*
 * file: KDTree.hpp
 * author: J. Frederico Carvalho
 *
 * This is an adaptation of the KD-tree implementation in rosetta code
 *  https://rosettacode.org/wiki/K-d_tree
 * It is a reimplementation of the C code using C++.
 * It also includes a few more queries than the original
 *
 */

/*
 * modified by Juyeop Han
 *
 * The header file is modified to implement a k-d tree in array structure
 * The only function in the k-d tree array class is range search
 * 
 */

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

// point_t : k-dimensional point
// indexArr :  index of a point in array
// pointIndex : point and index pair

using point_t = std::vector< float >;
using indexArr = std::vector< int >;
using pointIndex = typename std::pair< std::vector< float >, int >;


namespace dom {

    /*
     * k-d Tree Node
     *
     * Member Variable
     *  index : point index in the pointcloud
     *  x : point (3d)
     */

    class KDNode {
       public:
        using KDNodePtr = std::shared_ptr< KDNode >;
        int index;
        point_t x;
        KDNodePtr left;
        KDNodePtr right;

        // initializer
        KDNode();
        KDNode(const point_t &, const int &, const KDNodePtr &,
               const KDNodePtr &);
        KDNode(const pointIndex &, const KDNodePtr &, const KDNodePtr &);
        ~KDNode();

        // getter
        float coord(const int &);

        // conversions
        explicit operator bool();
        explicit operator point_t();
        explicit operator int();
        explicit operator pointIndex();
    };

    using KDNodePtr = std::shared_ptr< KDNode >;

    KDNodePtr NewKDNodePtr();

    // square euclidean distance
    inline float dist2(const point_t &, const point_t &);
    inline float dist2(const KDNodePtr &, const KDNodePtr &);
    inline float dist2(
        const float& , const float&, const float&,
        const float&, const float&, const float&
        );

    // euclidean distance
    inline float dist(const point_t &, const point_t &);
    inline float dist(const KDNodePtr &, const KDNodePtr &);
    inline float dist2(
        const float& , const float&, const float&,
        const float&, const float&, const float&
        );

    // Need for sorting
    class comparer {
       public:
        int idx;
        explicit comparer(int idx_);
        inline bool compare_idx(
            const std::pair< std::vector< float >, int > &,  //
            const std::pair< std::vector< float >, int > &   //
        );
    };

    // vector of pointIndex
    using pointIndexArr = typename std::vector< pointIndex >;

    inline void sort_on_idx(const pointIndexArr::iterator &,  //
                            const pointIndexArr::iterator &,  //
                            int idx);

    // point clouds...
    using pointVec = std::vector< point_t >;

    class KDTree {
        KDNodePtr root;
        KDNodePtr leaf;
        int length;

        KDNodePtr make_tree(const pointIndexArr::iterator &begin,  //
                            const pointIndexArr::iterator &end,    //
                            const int &length,                  //
                            const int &level                    //
        );

       public:
        KDTree() = default;
        explicit KDTree(pointVec point_array);
        int getLength(){ return length;}
        KDNodePtr getRoot() {return root;}

       private:
        KDNodePtr nearest_(           //
            const KDNodePtr &branch,  //
            const point_t &pt,        //
            const int &level,      //
            const KDNodePtr &best,    //
            const float &best_dist   //
        );

        // default caller
        KDNodePtr nearest_(const point_t &pt);

       public:
        point_t nearest_point(const point_t &pt);
        int nearest_index(const point_t &pt);
        pointIndex nearest_pointIndex(const point_t &pt);

       private:
        pointIndexArr neighborhood_(  //
            const KDNodePtr &branch,  //
            const point_t &pt,        //
            const float &rad,        //
            const int &level       //
        );

       public:
        pointIndexArr neighborhood(  //
            const point_t &pt,       //
            const float &rad);

        pointVec neighborhood_points(  //
            const point_t &pt,         //
            const float &rad);

        indexArr neighborhood_indices(  //
            const point_t &pt,          //
            const float &rad);
    };

    class KDTreeArr {
        public:
            float* x_coord;
            float* y_coord;
            float* z_coord;
            int* idx_arr;
            int length;
            int slot; // size of the arrays

            void convert2Arr(
                const int& root_idx,
                const KDNodePtr& branch);

            KDTreeArr() = default;
            KDTreeArr(
            KDTree& kdtree,
            float*& x_coord, 
            float*& y_coord,
            float*& z_coord);
            ~KDTreeArr() = default;

            inline bool isEmpty(const int& idx) { return idx_arr[idx] < 0;}
            inline bool isInvalid(const int& idx) {return idx + 1 > slot;}
            inline int left_child_idx(const int& idx){return 2*idx + 1;}
            inline int right_child_idx(const int& idx){return 2*idx + 2;}
            inline int parent_idx(const int& idx){return (idx - 1)/2;}

        private:

            indexArr neighborhood_indices_(
                const int& branch_idx,
                const float& x_pos,
                const float& y_pos,
                const float& z_pos,
                const float& rad,
                const int& level);

        public:
            indexArr neighborhood_indices(
                const float& x_pos,
                const float& y_pos,
                const float& z_pos,
                const float &rad);
    };
    // 사실 필요없을 수도 있겠습니다.
    class Free_KDTreeArr : public KDTreeArr {
        public:
            int* source_beam_arr;

        Free_KDTreeArr(
            KDTree& kdtree,
            float*& x_coord, 
            float*& y_coord,
            float*& z_coord,
            int*& source_beam_arr) : source_beam_arr(source_beam_arr) {
                KDTreeArr(kdtree, x_coord, y_coord, z_coord);
            } ;

    };
}