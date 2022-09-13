/*
 * file: KDTree.hpp
 * author: J. Frederico Carvalho
 *
 * This is an adaptation of the KD-tree implementation in rosetta code
 * https://rosettacode.org/wiki/K-d_tree
 *
 * It is a reimplementation of the C code using C++.  It also includes a few
 * more queries than the original, namely finding all points at a distance
 * smaller than some given distance to a point.
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
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>
#include <iostream>
#include <cmath>

#include "common/KDTree.h"

namespace dom{

    KDNode::KDNode() = default; // explicit intializaer

    KDNode::KDNode(const point_t &pt, const int &idx_, const KDNodePtr &left_,
                const KDNodePtr &right_) {
        x = pt;
        index = idx_;
        left = left_;
        right = right_;
    }

    KDNode::KDNode(const pointIndex &pi, const KDNodePtr &left_,
                const KDNodePtr &right_) {
        x = pi.first;
        index = pi.second;
        left = left_;
        right = right_;
    }

    KDNode::~KDNode() = default;

    float KDNode::coord(const int &idx) { return x.at(idx); }
    KDNode::operator bool() { return (!x.empty()); }
    KDNode::operator point_t() { return x; }
    KDNode::operator int() { return index; }
    KDNode::operator pointIndex() { return pointIndex(x, index); }

    KDNodePtr NewKDNodePtr() {
        KDNodePtr mynode = std::make_shared< KDNode >();
        return mynode;
    }

    inline float dist2(const point_t &a, const point_t &b) {
        float distc = 0;
        for (int i = 0; i < a.size(); i++) {
            float di = a.at(i) - b.at(i);
            distc += di * di;
        }
        return distc;
    }

    inline float dist2(const KDNodePtr &a, const KDNodePtr &b) {
        return dist2(a->x, b->x);
    }

    inline float dist2(
        const float& x1, const float& y1, const float& z1,
        const float& x2, const float& y2, const float& z2)
    {
        float distc = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
        return distc;
    }

    inline float dist(const point_t &a, const point_t &b) {
        return std::sqrt(dist2(a, b));
    }

    inline float dist(const KDNodePtr &a, const KDNodePtr &b) {
        return std::sqrt(dist2(a, b));
    }

    inline float dist(
        const float& x1, const float& y1, const float& z1,
        const float& x2, const float& y2, const float& z2
        ) { return std::sqrt(dist2(x1, y1, z1, x2, y2, z2));}

    comparer::comparer(int idx_) : idx{idx_} {};

    // if b is bigger than a
    // 오름차순
    inline bool comparer::compare_idx(const pointIndex &a,  //
                                    const pointIndex &b   //
    ) {
        return (a.first.at(idx) < b.first.at(idx));  //
    }

    inline void sort_on_idx(const pointIndexArr::iterator &begin,  //
                            const pointIndexArr::iterator &end,    //
                            int idx) {
        comparer comp(idx);
        comp.idx = idx;

        using std::placeholders::_1;
        using std::placeholders::_2;

        std::nth_element(begin, begin + std::distance(begin, end) / 2,
                        end, std::bind(&comparer::compare_idx, comp, _1, _2));
    }

    // point clouds...
    using pointVec = std::vector< point_t >;

    KDNodePtr KDTree::make_tree(const pointIndexArr::iterator &begin,  //
                                const pointIndexArr::iterator &end,    //
                                const int &length,                  //
                                const int &level                    //
    ) {
        if (begin == end) {
            return NewKDNodePtr();  // empty tree
        }

        int dim = begin->first.size();

        if (length > 1) {
            sort_on_idx(begin, end, level);
        }

        auto middle = begin + (length / 2);

        auto l_begin = begin;
        auto l_end = middle;
        auto r_begin = middle + 1;
        auto r_end = end;

        int l_len = length / 2;
        int r_len = length - l_len - 1;

        KDNodePtr left;
        if (l_len > 0 && dim > 0) {
            left = make_tree(l_begin, l_end, l_len, (level + 1) % dim);
        } else {
            left = leaf;
        }
        KDNodePtr right;
        if (r_len > 0 && dim > 0) {
            right = make_tree(r_begin, r_end, r_len, (level + 1) % dim);
        } else {
            right = leaf;
        }

        // KDNode result = KDNode();
        return std::make_shared< KDNode >(*middle, left, right);
    }

    KDTree::KDTree(pointVec point_array) {
        leaf = std::make_shared< KDNode >();
        // iterators
        pointIndexArr arr;
        for (int i = 0; i < point_array.size(); i++) {
            arr.push_back(pointIndex(point_array.at(i), i));
        }

        auto begin = arr.begin();
        auto end = arr.end();

        length = arr.size();
        int level = 0;  // starting

        root = KDTree::make_tree(begin, end, length, level);
    }

    KDNodePtr KDTree::nearest_(   //
        const KDNodePtr &branch,  //
        const point_t &pt,        //
        const int &level,      //
        const KDNodePtr &best,    //
        const float &best_dist   //
    ) {
        float d, dx, dx2;

        if (!bool(*branch)) {
            return NewKDNodePtr();  // basically, null
        }

        point_t branch_pt(*branch);
        int dim = branch_pt.size();

        d = dist2(branch_pt, pt);
        dx = branch_pt.at(level) - pt.at(level);
        dx2 = dx * dx;

        KDNodePtr best_l = best;
        float best_dist_l = best_dist;

        if (d < best_dist) {
            best_dist_l = d;
            best_l = branch;
        }

        int next_lv = (level + 1) % dim;
        KDNodePtr section;
        KDNodePtr other;

        // select which branch makes sense to check
        if (dx > 0) {
            section = branch->left;
            other = branch->right;
        } else {
            section = branch->right;
            other = branch->left;
        }

        // keep nearest neighbor from further down the tree
        KDNodePtr further = nearest_(section, pt, next_lv, best_l, best_dist_l);
        if (!further->x.empty()) {
            float dl = dist2(further->x, pt);
            if (dl < best_dist_l) {
                best_dist_l = dl;
                best_l = further;
            }
        }
        // only check the other branch if it makes sense to do so
        if (dx2 < best_dist_l) {
            further = nearest_(other, pt, next_lv, best_l, best_dist_l);
            if (!further->x.empty()) {
                float dl = dist2(further->x, pt);
                if (dl < best_dist_l) {
                    best_dist_l = dl;
                    best_l = further;
                }
            }
        }

        return best_l;
    };

    // default caller
    KDNodePtr KDTree::nearest_(const point_t &pt) {
        int level = 0;
        // KDNodePtr best = branch;
        float branch_dist = dist2(point_t(*root), pt);
        return nearest_(root,          // beginning of tree
                        pt,            // point we are querying
                        level,         // start from level 0
                        root,          // best is the root
                        branch_dist);  // best_dist = branch_dist
    };

    point_t KDTree::nearest_point(const point_t &pt) {
        return point_t(*nearest_(pt));
    };
    int KDTree::nearest_index(const point_t &pt) {
        return int(*nearest_(pt));
    };

    pointIndex KDTree::nearest_pointIndex(const point_t &pt) {
        KDNodePtr Nearest = nearest_(pt);
        return pointIndex(point_t(*Nearest), int(*Nearest));
    }

    pointIndexArr KDTree::neighborhood_(  //
        const KDNodePtr &branch,          //
        const point_t &pt,                //
        const float &rad,                //
        const int &level               //
    ) {
        float d, dx, dx2;

        if (!bool(*branch)) {
            // branch has no point, means it is a leaf,
            // no points to add
            return pointIndexArr();
        }

        int dim = pt.size();

        float r2 = rad * rad;

        d = dist2(point_t(*branch), pt);
        dx = point_t(*branch).at(level) - pt.at(level);
        dx2 = dx * dx;

        pointIndexArr nbh, nbh_s, nbh_o;
        if (d <= r2) {
            nbh.push_back(pointIndex(*branch));
        }

        //
        KDNodePtr section;
        KDNodePtr other;
        if (dx > 0) {
            section = branch->left;
            other = branch->right;
        } else {
            section = branch->right;
            other = branch->left;
        }

        nbh_s = neighborhood_(section, pt, rad, (level + 1) % dim);
        nbh.insert(nbh.end(), nbh_s.begin(), nbh_s.end());
        if (dx2 < r2) {
            nbh_o = neighborhood_(other, pt, rad, (level + 1) % dim);
            nbh.insert(nbh.end(), nbh_o.begin(), nbh_o.end());
        }

        return nbh;
    };

    pointIndexArr KDTree::neighborhood(  //
        const point_t &pt,               //
        const float &rad) {
        int level = 0;
        return neighborhood_(root, pt, rad, level);
    }

    pointVec KDTree::neighborhood_points(  //
        const point_t &pt,                 //
        const float &rad) {
        int level = 0;
        pointIndexArr nbh = neighborhood_(root, pt, rad, level);
        pointVec nbhp;
        nbhp.resize(nbh.size());
        std::transform(nbh.begin(), nbh.end(), nbhp.begin(),
                    [](pointIndex x) { return x.first; });
        return nbhp;
    }

    indexArr KDTree::neighborhood_indices(  //
        const point_t &pt,                  //
        const float &rad) {
        int level = 0;
        pointIndexArr nbh = neighborhood_(root, pt, rad, level);
        indexArr nbhi;
        nbhi.resize(nbh.size());
        std::transform(nbh.begin(), nbh.end(), nbhi.begin(),
                    [](pointIndex x) { return x.second; });
        return nbhi;
    }

    void KDTreeArr::convert2Arr(
        const int& root_idx,
        const KDNodePtr& branch){

        idx_arr[root_idx] = branch->index;
        
        if (!branch->left->x.empty())
            {
                int left_idx = left_child_idx(root_idx);
                convert2Arr(left_idx, branch->left);
            }

        if (!branch->right->x.empty())
            {
                int right_idx = right_child_idx(root_idx);
                convert2Arr(right_idx, branch->right);
            } 
    }
                
    KDTreeArr::KDTreeArr(
        KDTree &kdtree, 
        float*& x_coord, 
        float*& y_coord,
        float*& z_coord) : x_coord(x_coord), y_coord(y_coord), z_coord(z_coord) {
        length = kdtree.getLength();
        float depth = ceil(log2f ((float) length));
        slot = (int) pow(2.0, depth) - 1;

        idx_arr = new int[slot];
        for ( int i = 0; i < slot; ++i){
            idx_arr[i] = -1;
        }
        convert2Arr(0, kdtree.getRoot());
    }



    indexArr KDTreeArr::neighborhood_indices_(
        const int& branch_idx,
        const float& x_pos,
        const float& y_pos,
        const float& z_pos,
        const float& rad,
        const int& level) {

        float d, dx, dx2, level_pos, level_pos_branch;

        // for debugging
        std::cout << "this is index : " << branch_idx << std::endl;
        // if (isEmpty(branch_idx)){
        //     std::vector<int> empty_vec;
        //     return empty_vec;
        // }

        int dim = 3;

        float r2 = rad * rad;

        float x_pos_branch = x_coord[idx_arr[branch_idx]];
        float y_pos_branch = y_coord[idx_arr[branch_idx]];
        float z_pos_branch = z_coord[idx_arr[branch_idx]];

        // d = dist2(point_t(*branch), pt);
        // dx = point_t(*branch).at(level) - pt.at(level);

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

        indexArr nbh, nbh_s, nbh_o;
        if (d <= r2) {
            nbh.push_back(branch_idx);
        }

        int section_idx, other_idx;
        if (dx > 0){
            section_idx = left_child_idx(branch_idx);
            other_idx = right_child_idx(branch_idx);
        } else {
            section_idx = right_child_idx(branch_idx);
            other_idx = left_child_idx(branch_idx);
        }
        if (!isInvalid(section_idx) && !isEmpty(section_idx)){
            nbh_s = neighborhood_indices_(section_idx, x_pos, y_pos, z_pos, rad, (level + 1) % dim);
            nbh.insert(nbh.end(), nbh_s.begin(), nbh_s.end());
        }
        
        if ((dx2 < r2) && !isInvalid(other_idx) && !isEmpty(other_idx)) {
            nbh_o = neighborhood_indices_(other_idx, x_pos, y_pos, z_pos, rad, (level + 1) % dim);
            nbh.insert(nbh.end(), nbh_o.begin(), nbh_o.end());
        }

        return nbh;
    };

    indexArr KDTreeArr::neighborhood_indices(
        const float& x_pos,
        const float& y_pos,
        const float& z_pos,
        const float &rad){
        int level = 0;
        return neighborhood_indices_(0, x_pos, y_pos, z_pos, rad, level);
    }   
}
