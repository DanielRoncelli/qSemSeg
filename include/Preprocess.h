// ##########################################################################
// #                                                                        #
// #                     CLOUDCOMPARE PLUGIN: qSemSeg                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: Daniel Roncelli                                    #
// #                                                                        #
// ##########################################################################

#pragma once

#include "ModelSpec.h"

// nanoflann (vendored in CCCoreLib, include path exported by the nanoflann::nanoflann target)
#include <cstdint>
#include <memory>
#include <nanoflann.hpp>
#include <random>
#include <vector>

//! KDTree over an N*3 point array
struct PointCloudAdaptor
{
	const std::vector<float>* pts = nullptr; // N*3

	inline size_t kdtree_get_point_count() const
	{
		return pts->size() / 3;
	}
	inline float kdtree_get_pt(size_t idx, size_t dim) const
	{
		return (*pts)[idx * 3 + dim];
	}
	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const
	{
		return false;
	}
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3>;

//! k nearest neighbors of query point q (3 floats) in a tree over n points
/** Pads to k by repeating the last index. distsOut (optional) receives the k squared distances. **/
std::vector<int64_t> knnQuery(const KDTree& tree, size_t n, const float* q, int k, std::vector<float>* distsOut = nullptr);

//! kNN of every query point in queries (Q*3) against the tree. Returns Q*k indices, row-major
std::vector<int64_t> knnQueryBatch(const KDTree& tree, size_t n, const std::vector<float>& queries, int k, std::vector<float>* distsOut = nullptr);

//! Farthest point sampling indices
/** fixedStart >= 0 pins the initial sample (used by the parity tests to
    reproduce the Python reference's FPS); -1 picks a random start. **/
std::vector<int64_t> fpsIndices(const std::vector<float>& pts, size_t nSamples, std::mt19937_64& rng, int64_t fixedStart = -1);

//! One-time cloud preprocessing result
struct CloudPre
{
	std::vector<float>                 subPoints;    // M*3
	std::vector<float>                 subRgb;       // M*3 or empty
	std::vector<float>                 subIntensity; // M or empty
	std::vector<int>                   subLabels;    // M or empty
	std::vector<int64_t>               projInds;     // original N -> subsampled M
	std::unique_ptr<PointCloudAdaptor> adaptor;
	std::unique_ptr<KDTree>            tree; // over subPoints
	size_t                             M = 0;
};

//! Subsampling parameters for preprocessCloud (manifest preprocessing.subsampling)
/** type Voxel uses voxelSize; Random/Fps pick `target` points (rng must be
    provided and seeded by the caller for reproducible runs). type None and
    Random/Fps keep original coordinates (their sub points ARE original
    points, so the projection map is the identity). **/
struct SubsamplingParams
{
	ModelSpec::SubsamplingType type   = ModelSpec::SubsamplingType::Voxel;
	int                        target = 0;       // Random/Fps subset size
	std::mt19937_64*           rng    = nullptr; // required for Random/Fps
};

//! Subsamples the cloud (per SubsamplingParams.type) + nearest-neighbor projection map back to the original cloud
/** voxel: barycenter of points / mean of features per voxel (uses voxelSize
    and shiftMin); none / random / fps: subset of the original points
    (random: rng choice without replacement; fps: farthest point sampling). **/
CloudPre preprocessCloud(const std::vector<float>& xyz, const std::vector<float>& rgb, const std::vector<float>& intensity, const std::vector<int>& labels, double voxelSize, bool shiftMin, const SubsamplingParams& subsampling = {});

//! FKAConv multi-scale hierarchy (4 stages of support points + 14 index sets)
struct Hierarchy
{
	std::vector<std::vector<float>>   sp;  // 4 stages, each S*3
	std::vector<std::vector<int64_t>> idx; // 14 index sets, each S*K row-major
};

Hierarchy computeHierarchy(const std::vector<float>& points, const std::vector<int>& stageNpoints, int numNeighbors, std::mt19937_64& rng, const std::vector<int64_t>* fpsStarts = nullptr);
