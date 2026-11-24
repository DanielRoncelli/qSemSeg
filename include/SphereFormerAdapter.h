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

// ModelAdapter for SphereFormer-style fixed-shape graphs (Open3D-ML
// SphereFormerSemSeg exports): 67 inputs per patch (feat + 5 levels of
// SubMConv/window-attention tables + stride-2 down/inverse tables), int32
// index tables, batch axis baked to 1. All structural tensors (voxel grids,
// window partitions, pair tables) are computed per patch OUTSIDE the graph,
// exactly as the reference Python builder (onnx_adapters._sphere_patch_tensors
// + spconv_plugin table builders) does.

#include "ModelAdapter.h"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

//! Lexicographic stable partition of n int64 triples
/** Replica of numpy's np.unique(axis=0, return_inverse=True, return_counts=True)
    combined with the stable argsort of the inverse that the window machinery
    builds on: rows are grouped by their triple in ascending lexicographic
    order, ties keep original row order. All triple components are
    non-negative in every caller (floor of non-negative quotients), so the
    lexicographic order is the numeric one. **/
struct SphereLexPartition
{
	std::vector<int64_t> sort;   //!< n: sorted position -> row (stable lex order)
	std::vector<int64_t> iperm;  //!< n: row -> sorted position
	std::vector<int64_t> rank;   //!< n: row -> unique-group index (lex order)
	std::vector<int64_t> counts; //!< U: group sizes (lex order)
	std::vector<int64_t> uniq;   //!< U*3: the distinct triples (lex order)
};

//! cart2sphere replica (float32 in/out, degrees), bit-exact vs numpy
/** xyz: n*3 floats; out: n*3 (theta, beta, r). The exact op order of the
    reference (_cart2sphere_np) is replicated:
      theta = (atan2_d(y, x) + PI_f32) * f32(180/PI)   (atan2 computed in
              double precision, then rounded to float32 - numpy's float32
              arctan2 is exactly this, MinGW's atan2f is NOT)
      beta  = atan2_d(sqrt_f32(x*x + y*y), z) * f32(180/PI)
      r     = sqrt_f32(x*x + y*y + z*z)                (all f32, left-to-right) **/
void sphereCart2Sphere(const std::vector<float>& xyz, std::vector<float>& out);

//! Lexicographic stable partition (see SphereLexPartition)
/** triples: n*3 int64 values (row-major). Throws on triples.size() % 3. **/
void sphereLexPartition(const std::vector<int64_t>& triples, SphereLexPartition& out);

//! RPE quantization: floor(((coords - min) % window) / quant) per row
/** coords: n*3 float32 (row-major); window/quant: per-axis float32 sizes.
    Values are non-negative (coords >= min), so fmodf == numpy remainder.
    out: n*3 int64. **/
void sphereQuant(const std::vector<float>& coords, const std::array<float, 3>& window, const std::array<float, 3>& quant, std::vector<int64_t>& out);

//! scatter_mean(dim=0) with float64 row-order accumulation -> float32
/** xyz: m*3 floats; index: m values in [0, outRows); out: outRows*3. Each
    row of xyz is accumulated into out[index[i]] IN ROW ORDER in double
    precision, then divided by the hit count (0/0 -> NaN, like numpy). **/
void sphereScatterMean(const std::vector<float>& xyz, const std::vector<int64_t>& index, size_t outRows, std::vector<float>& out);

//! SubMConv3d(k=3, p=1) per-out-row in-row table (cap x 27 int32, ROW-major)
/** voxels: M*3 level voxel indices (row-major). Column k of row v holds the
    in-row feeding kernel offset k at out row v, or `cap` (the zero feature
    row appended in-graph) when the tap contributes nothing. Offset 13
    (center) is the row itself for EVERY row (padding rows included).
    Duplicate voxels resolve to the FIRST row ('first' rule: the smallest
    original row index of the voxel, matching spconv's hash insert order). **/
void sphereSubmGatherTable(const std::vector<int64_t>& voxels, const std::array<int64_t, 3>& spatialShape, int64_t cap, std::vector<int32_t>& out);

//! SparseConv3d(k=2, s=2) gather tables (stride-2 downsampling conv)
/** childPartition: lex partition of the CHILD voxel triples (M rows);
    invp: parent rank per child row (from the partition of voxels//2).
    Returns nin (nextCap x 8 int32, nin[parent*8 + kv] = unique-child-voxel
    row or cap), perm (cap, child rows sorted by unique-voxel rank - the
    partition's own sort), bounds (cap+1, segment boundaries per unique
    child voxel, tail = M). **/
void sphereDownGatherTables(const SphereLexPartition& childPartition, const std::vector<int64_t>& invp, int64_t cap, int64_t nextCap, std::vector<int32_t>& ninOut, std::vector<int32_t>& permOut, std::vector<int32_t>& boundsOut);

//! SparseInverseConv3d(k=2) per-child-row parent table (cap x 8 int32)
/** invp: parent rank per child row. tab[row*8 + kv] = parent rank, where
    kv = the child voxel's odd/even bits (4*x + 2*y + z); every other column
    and every padding row stays at nextCap (the zero parent row appended
    in-graph). **/
void sphereInverseGatherTable(const std::vector<int64_t>& voxels, const std::vector<int64_t>& invp, int64_t nextCap, int64_t cap, std::vector<int32_t>& out);

//! Adapter for SphereFormer-style fixed-shape graphs
/** Requires the manifest adapter.window (per-level window/quant constants)
    and adapter.caps (structural capacities) blocks; the exported graph
    cannot grow its buffers, so exceeding a capacity is a loud error telling
    the user to re-export with larger caps - never a silent truncation. **/
class SphereFormerAdapter : public ModelAdapter
{
  public:
	const char* id() const override
	{
		return "SphereFormer";
	}

	void applyDefaults(ModelSpec& spec) const override;

	std::string validate(const ModelSpec& spec, const IOContract& io) const override;

	std::string validateCloudFeatures(const ModelSpec& spec, bool hasColors, bool hasIntensity) const override;

	Patch buildPatch(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, std::mt19937_64& rng, const std::vector<int64_t>* fpsStarts, const float* fixedMean) const override;

	BatchBuffers assembleBatch(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage) const override;

	//! The 'feat' input is [num_points, channels]: its ROW count is the axis
	int64_t staticNumPointsFromIO(const IOContract& io) const override;

	//! The export bakes the batch axis to 1 (no dynamic batch at all)
	int fixedBatchSize(const ModelSpec& spec) const override
	{
		(void)spec;
		return 1;
	}
};
