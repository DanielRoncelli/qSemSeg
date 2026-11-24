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

#include "SphereFormerAdapter.h"

#include <cstdio>

// system
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
	//! Slot of one expected graph input (name -> kind + level)
	struct SphereSlot
	{
		enum Kind
		{
			Feat,
			SubmNin,
			CarCnt,
			CarSrt,
			CarInv,
			CarQ,
			SphCnt,
			SphSrt,
			SphInv,
			SphQ,
			SphR,
			DownNin,
			DownPerm,
			DownBnd,
			InvNin
		};
		Kind kind;
		int  level;
	};

	//! The 67 expected input names of a SphereFormer graph, in reference order
	/** feat, then per level l: subm{l}_nin, car_cnt/srt/inv/q{l}, sph_cnt/
	    srt/inv/q{l}, sph_r{l}; then per level l<4: down{l}_nin/perm/bnd and
	    inv{l}_nin (matches SphereFormerONNXAdapter.input_names). **/
	std::map<std::string, SphereSlot> sphereExpectedInputs()
	{
		std::map<std::string, SphereSlot> map;
		const auto                        put = [&](const std::string& name, SphereSlot::Kind kind, int level)
		{
			map.emplace(name, SphereSlot{kind, level});
		};
		put("feat", SphereSlot::Feat, 0);
		for (int l = 0; l < 5; l++)
		{
			const std::string ls = std::to_string(l);
			put("subm" + ls + "_nin", SphereSlot::SubmNin, l);
			put("car_cnt" + ls, SphereSlot::CarCnt, l);
			put("car_srt" + ls, SphereSlot::CarSrt, l);
			put("car_inv" + ls, SphereSlot::CarInv, l);
			put("car_q" + ls, SphereSlot::CarQ, l);
			put("sph_cnt" + ls, SphereSlot::SphCnt, l);
			put("sph_srt" + ls, SphereSlot::SphSrt, l);
			put("sph_inv" + ls, SphereSlot::SphInv, l);
			put("sph_q" + ls, SphereSlot::SphQ, l);
			put("sph_r" + ls, SphereSlot::SphR, l);
		}
		for (int l = 0; l < 4; l++)
		{
			const std::string ls = std::to_string(l);
			put("down" + ls + "_nin", SphereSlot::DownNin, l);
			put("down" + ls + "_perm", SphereSlot::DownPerm, l);
			put("down" + ls + "_bnd", SphereSlot::DownBnd, l);
			put("inv" + ls + "_nin", SphereSlot::InvNin, l);
		}
		return map;
	}

	//! The extra feature channels (after xyz), k rows of E columns, row-major
	/** Mirror of ModelAdapter.cpp's buildExtraChannels (file-local there):
	    feature_columns order, default intensity then rgb. **/
	std::vector<float> sphereExtraChannels(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, int extraCount)
	{
		std::vector<std::string> columns = spec.featureColumns;
		if (columns.empty())
		{
			columns = {"intensity", "rgb"};
		}
		// expand "rgb" to 3 channels (matches the generic adapter)
		std::vector<std::string> expanded;
		for (const auto& col : columns)
		{
			if (col == "rgb")
			{
				expanded.push_back("r");
				expanded.push_back("g");
				expanded.push_back("b");
			}
			else
			{
				expanded.push_back(col);
			}
		}
		columns = expanded;
		if (static_cast<int>(columns.size()) < extraCount)
		{
			throw std::runtime_error("the model needs " + std::to_string(extraCount) + " extra feature channel(s) (after xyz) but only " + std::to_string(columns.size()) + " are declared");
		}
		const size_t       k = idxs.size();
		std::vector<float> extras(k * static_cast<size_t>(extraCount));
		for (size_t i = 0; i < k; i++)
		{
			for (int e = 0; e < extraCount; e++)
			{
				const std::string& column = columns[e];
				if (column == "intensity" || column == "i")
				{
					if (pre.subIntensity.empty())
					{
						throw std::runtime_error("feature 'intensity' is required by the model but the cloud provides no scalar field for it");
					}
					extras[i * extraCount + e] = pre.subIntensity[idxs[i]];
				}
				else if (column == "r" || column == "g" || column == "b")
				{
					if (pre.subRgb.empty())
					{
						throw std::runtime_error("feature 'rgb' is required by the model but the cloud has no colors");
					}
					const int c                = (column == "r") ? 0 : (column == "g") ? 1
					                                                                   : 2;
					extras[i * extraCount + e] = pre.subRgb[idxs[i] * 3 + c];
				}
				else if (column == "rgb")
				{
					throw std::runtime_error("feature column 'rgb' is not a single scalar - it expands to 3 channels; use 'r', 'g', 'b' instead");
				}
				else
				{
					throw std::runtime_error("unsupported feature column '" + column + "' (supported: intensity, rgb)");
				}
			}
		}
		return extras;
	}

	//! Level capacity C[l]: level 0 is the patch point count, deeper levels
	//! come from the manifest's adapter.caps.C
	int64_t sphereLevelCap(const ModelSpec& spec, int level, int64_t numPoints)
	{
		return level == 0 ? numPoints : spec.sphereCapsC[static_cast<size_t>(level) - 1];
	}

	//! Per-branch/level window-count (W) and attention-pair (M) capacities
	//! (flat 10-entry lists in car0..car4, sph0..sph4 order)
	int64_t sphereBranchCap(const std::vector<int64_t>& caps, int level, bool sphere)
	{
		return caps[static_cast<size_t>(level) + (sphere ? 5 : 0)];
	}
} // namespace

void sphereCart2Sphere(const std::vector<float>& xyz, std::vector<float>& out)
{
	const size_t n = xyz.size() / 3;
	out.resize(n * 3);
	// single-rounding constants, exactly as the numpy reference sees them:
	// np.pi enters as a weak python float cast to float32, and 180.0/np.pi is
	// computed in DOUBLE then rounded once (f32(180)/f32(PI) would
	// double-round and flip boundary cases)
	const float PI  = static_cast<float>(M_PI);
	const float DEG = static_cast<float>(180.0 / M_PI);
	for (size_t i = 0; i < n; i++)
	{
		const float x = xyz[i * 3 + 0];
		const float y = xyz[i * 3 + 1];
		const float z = xyz[i * 3 + 2];
		// numpy's float32 arctan2 == (float)atan2(double, double) (verified
		// bit-exact on 400k samples); plain atan2f differs by 1-2 ULP
		const float theta = (static_cast<float>(std::atan2(static_cast<double>(y), static_cast<double>(x))) + PI) * DEG;
		const float xy    = x * x + y * y; // f32 left-to-right (no FMA: the
		                                   // build targets baseline x86-64)
		const float beta = static_cast<float>(std::atan2(static_cast<double>(std::sqrt(xy)), static_cast<double>(z))) * DEG;
		const float r    = std::sqrt(xy + z * z);
		out[i * 3 + 0]   = theta;
		out[i * 3 + 1]   = beta;
		out[i * 3 + 2]   = r;
	}
}

void sphereLexPartition(const std::vector<int64_t>& triples, SphereLexPartition& out)
{
	if (triples.size() % 3 != 0)
	{
		throw std::runtime_error("sphereLexPartition: triples size must be a multiple of 3");
	}
	const size_t n = triples.size() / 3;
	out.sort.resize(n);
	out.iperm.resize(n);
	out.rank.resize(n);
	std::iota(out.sort.begin(), out.sort.end(), static_cast<int64_t>(0));
	// stable sort by (t0, t1, t2): equal triples keep original row order,
	// which is exactly argsort(rank, stable) after ranking
	std::stable_sort(out.sort.begin(), out.sort.end(), [&triples](int64_t a, int64_t b)
	                 {
		for (int d = 0; d < 3; d++)
		{
			const int64_t va = triples[static_cast<size_t>(a) * 3 + d];
			const int64_t vb = triples[static_cast<size_t>(b) * 3 + d];
			if (va != vb)
			{
				return va < vb;
			}
		}
		return false; });
	out.counts.clear();
	out.uniq.clear();
	out.uniq.reserve(n * 3);
	int64_t group = -1;
	for (size_t pos = 0; pos < n; pos++)
	{
		const int64_t row      = out.sort[pos];
		bool          newGroup = (pos == 0);
		if (!newGroup)
		{
			const int64_t prev = out.sort[pos - 1];
			for (int d = 0; d < 3; d++)
			{
				newGroup = newGroup || triples[static_cast<size_t>(row) * 3 + d] != triples[static_cast<size_t>(prev) * 3 + d];
			}
		}
		if (newGroup)
		{
			out.counts.push_back(1);
			group++;
			for (int d = 0; d < 3; d++)
			{
				out.uniq.push_back(triples[static_cast<size_t>(row) * 3 + d]);
			}
		}
		else
		{
			out.counts.back()++;
		}
		out.rank[row]  = group;
		out.iperm[row] = static_cast<int64_t>(pos);
	}
}

void sphereQuant(const std::vector<float>& coords, const std::array<float, 3>& window, const std::array<float, 3>& quant, std::vector<int64_t>& out)
{
	const size_t n = coords.size() / 3;
	out.resize(n * 3);
	for (int d = 0; d < 3; d++)
	{
		if (!(window[d] > 0.f) || !(quant[d] > 0.f))
		{
			throw std::runtime_error("sphereQuant: window and quant sizes must be positive");
		}
	}
	// per-dim minimum of the (sorted) coordinates, exactly as the reference's
	// _sphere_quant
	float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
	for (size_t i = 0; i < n; i++)
	{
		for (int d = 0; d < 3; d++)
		{
			mn[d] = std::min(mn[d], coords[i * 3 + d]);
		}
	}
	for (size_t i = 0; i < n; i++)
	{
		for (int d = 0; d < 3; d++)
		{
			// coords >= mn so the remainder is non-negative and fmodf ==
			// numpy's % operator
			const float rem = std::fmod(coords[i * 3 + d] - mn[d], window[d]);
			out[i * 3 + d]  = static_cast<int64_t>(std::floor(rem / quant[d]));
		}
	}
}

void sphereScatterMean(const std::vector<float>& xyz, const std::vector<int64_t>& index, size_t outRows, std::vector<float>& out)
{
	const size_t m = index.size();
	if (xyz.size() != m * 3)
	{
		throw std::runtime_error("sphereScatterMean: xyz size does not match the index count");
	}
	std::vector<double>  acc(outRows * 3, 0.0);
	std::vector<int64_t> cnt(outRows, 0);
	for (size_t i = 0; i < m; i++)
	{
		const int64_t p = index[i];
		if (p < 0 || static_cast<size_t>(p) >= outRows)
		{
			throw std::runtime_error("sphereScatterMean: index out of range");
		}
		// float64 accumulation in ROW order (np.add.at semantics)
		acc[static_cast<size_t>(p) * 3 + 0] += static_cast<double>(xyz[i * 3 + 0]);
		acc[static_cast<size_t>(p) * 3 + 1] += static_cast<double>(xyz[i * 3 + 1]);
		acc[static_cast<size_t>(p) * 3 + 2] += static_cast<double>(xyz[i * 3 + 2]);
		cnt[static_cast<size_t>(p)]++;
	}
	out.resize(outRows * 3);
	for (size_t p = 0; p < outRows; p++)
	{
		for (int d = 0; d < 3; d++)
		{
			// 0/0 -> NaN, exactly like numpy when a parent has no children
			out[p * 3 + d] = cnt[p] > 0 ? static_cast<float>(acc[p * 3 + d] / static_cast<double>(cnt[p])) : std::numeric_limits<float>::quiet_NaN();
		}
	}
}

void sphereSubmGatherTable(const std::vector<int64_t>& voxels, const std::array<int64_t, 3>& spatialShape, int64_t cap, std::vector<int32_t>& out)
{
	const int64_t m = static_cast<int64_t>(voxels.size()) / 3;
	if (m * 3 != static_cast<int64_t>(voxels.size()))
	{
		throw std::runtime_error("sphereSubmGatherTable: voxels size must be a multiple of 3");
	}
	const int64_t s0 = spatialShape[0], s1 = spatialShape[1], s2 = spatialShape[2];
	const auto    keyof = [&](int64_t g0, int64_t g1, int64_t g2)
	{
		return (g0 * s1 + g1) * s2 + g2;
	};
	if (s0 <= 0 || s1 <= 0 || s2 <= 0 || keyof(s0 - 1, s1 - 1, s2 - 1) > (int64_t{1} << 40))
	{
		throw std::runtime_error("sphereSubmGatherTable: spatial shape too large (patch extent does not fit the graph's fixed grid)");
	}
	std::vector<int64_t> keys(static_cast<size_t>(m));
	for (int64_t i = 0; i < m; i++)
	{
		keys[static_cast<size_t>(i)] = keyof(voxels[static_cast<size_t>(i) * 3 + 0], voxels[static_cast<size_t>(i) * 3 + 1], voxels[static_cast<size_t>(i) * 3 + 2]);
	}
	// stable argsort of the keys: equal keys keep original row order, so the
	// first occurrence of a voxel key is its smallest original row index
	// ('first' duplicate rule, matching spconv's hash insert order)
	std::vector<int64_t> order(static_cast<size_t>(m));
	std::iota(order.begin(), order.end(), static_cast<int64_t>(0));
	std::stable_sort(order.begin(), order.end(), [&keys](int64_t a, int64_t b)
	                 { return keys[static_cast<size_t>(a)] < keys[static_cast<size_t>(b)]; });
	std::vector<int64_t> keysSorted(static_cast<size_t>(m));
	for (int64_t p = 0; p < m; p++)
	{
		keysSorted[static_cast<size_t>(p)] = keys[static_cast<size_t>(order[static_cast<size_t>(p)])];
	}

	out.assign(static_cast<size_t>(cap) * 27, static_cast<int32_t>(cap));
	for (int64_t v = 0; v < cap; v++)
	{
		out[static_cast<size_t>(v) * 27 + 13] = static_cast<int32_t>(v); // self-pair for EVERY row
	}
	// kernel offsets in itertools.product([-1,0,1], repeat=3) order:
	// oi = (o0+1)*9 + (o1+1)*3 + (o2+1); oi 13 is the center (preset above)
	for (int oi = 0; oi < 27; oi++)
	{
		if (oi == 13)
		{
			continue;
		}
		const int64_t off[3] = {oi / 9 - 1, (oi / 3) % 3 - 1, oi % 3 - 1};
		for (int64_t i = 0; i < m; i++)
		{
			const int64_t c0       = voxels[static_cast<size_t>(i) * 3 + 0] + off[0];
			const int64_t c1       = voxels[static_cast<size_t>(i) * 3 + 1] + off[1];
			const int64_t c2       = voxels[static_cast<size_t>(i) * 3 + 2] + off[2];
			const bool    inBounds = c0 >= 0 && c0 < s0 && c1 >= 0 && c1 < s1 && c2 >= 0 && c2 < s2;
			if (!inBounds)
			{
				continue;
			}
			const int64_t ck = keyof(c0, c1, c2);
			const auto    lo = std::lower_bound(keysSorted.begin(), keysSorted.end(), ck);
			if (lo != keysSorted.end() && *lo == ck)
			{
				out[static_cast<size_t>(i) * 27 + oi] = static_cast<int32_t>(order[static_cast<size_t>(lo - keysSorted.begin())]);
			}
		}
	}
}

void sphereDownGatherTables(const SphereLexPartition& childPartition, const std::vector<int64_t>& invp, int64_t cap, int64_t nextCap, std::vector<int32_t>& ninOut, std::vector<int32_t>& permOut, std::vector<int32_t>& boundsOut)
{
	const int64_t m = static_cast<int64_t>(childPartition.sort.size());
	const int64_t u = static_cast<int64_t>(childPartition.counts.size());
	if (static_cast<int64_t>(invp.size()) != m)
	{
		throw std::runtime_error("sphereDownGatherTables: parent rank count does not match the child rows");
	}
	for (int64_t i = 0; i < m; i++)
	{
		if (invp[static_cast<size_t>(i)] < 0 || invp[static_cast<size_t>(i)] >= nextCap)
		{
			throw std::runtime_error("sphereDownGatherTables: parent rank exceeds the next-level capacity");
		}
	}

	// perm: child rows sorted by unique-voxel rank = the partition's own sort
	permOut.assign(static_cast<size_t>(cap), 0);
	for (int64_t p = 0; p < m; p++)
	{
		permOut[static_cast<size_t>(p)] = static_cast<int32_t>(childPartition.sort[static_cast<size_t>(p)]);
	}
	// bounds: [0, cs[1], ..., cs[U]] (cs = prefix sums of the counts), then
	// the tail filled with cs[U] = m (empty segments -> exact zero rows)
	boundsOut.assign(static_cast<size_t>(cap) + 1, static_cast<int32_t>(m));
	int64_t acc = 0;
	for (int64_t p = 0; p < u; p++)
	{
		boundsOut[static_cast<size_t>(p)] = static_cast<int32_t>(acc);
		acc += childPartition.counts[static_cast<size_t>(p)];
	}
	boundsOut[static_cast<size_t>(u)] = static_cast<int32_t>(acc);

	// nin: per parent row p and kernel offset kv, the unique-child-voxel row
	// feeding that tap, or cap (zero row of the aggregated child feats)
	ninOut.assign(static_cast<size_t>(nextCap) * 8, static_cast<int32_t>(cap));
	for (int64_t uv = 0; uv < u; uv++)
	{
		// all child rows of unique-voxel group uv share the voxel, hence the
		// same parent; the first row of the segment carries the rank
		const int64_t firstRow                                                = childPartition.sort[static_cast<size_t>(boundsOut[static_cast<size_t>(uv)])];
		const int64_t parentRank                                              = invp[static_cast<size_t>(firstRow)];
		const int64_t b0                                                      = ((childPartition.uniq[static_cast<size_t>(uv) * 3 + 0] % 2) + 2) % 2;
		const int64_t b1                                                      = ((childPartition.uniq[static_cast<size_t>(uv) * 3 + 1] % 2) + 2) % 2;
		const int64_t b2                                                      = ((childPartition.uniq[static_cast<size_t>(uv) * 3 + 2] % 2) + 2) % 2;
		const int64_t kv                                                      = b0 * 4 + b1 * 2 + b2;
		ninOut[static_cast<size_t>(parentRank) * 8 + static_cast<size_t>(kv)] = static_cast<int32_t>(uv);
	}
}

void sphereInverseGatherTable(const std::vector<int64_t>& voxels, const std::vector<int64_t>& invp, int64_t nextCap, int64_t cap, std::vector<int32_t>& out)
{
	const int64_t m = static_cast<int64_t>(voxels.size()) / 3;
	if (static_cast<int64_t>(invp.size()) != m)
	{
		throw std::runtime_error("sphereInverseGatherTable: parent rank count does not match the child rows");
	}
	out.assign(static_cast<size_t>(cap) * 8, static_cast<int32_t>(nextCap));
	for (int64_t i = 0; i < m; i++)
	{
		if (invp[static_cast<size_t>(i)] < 0 || invp[static_cast<size_t>(i)] >= nextCap)
		{
			throw std::runtime_error("sphereInverseGatherTable: parent rank exceeds the next-level capacity");
		}
		const int64_t b0                                          = ((voxels[static_cast<size_t>(i) * 3 + 0] % 2) + 2) % 2;
		const int64_t b1                                          = ((voxels[static_cast<size_t>(i) * 3 + 1] % 2) + 2) % 2;
		const int64_t b2                                          = ((voxels[static_cast<size_t>(i) * 3 + 2] % 2) + 2) % 2;
		const int64_t kv                                          = b0 * 4 + b1 * 2 + b2;
		out[static_cast<size_t>(i) * 8 + static_cast<size_t>(kv)] = static_cast<int32_t>(invp[static_cast<size_t>(i)]);
	}
}

namespace
{
	//! One branch's (car/sph) window tensors for one level
	struct SphereBranchTensors
	{
		std::vector<int32_t> cnt; // Wcap
		std::vector<int32_t> srt; // cap
		std::vector<int32_t> inv; // cap
		std::vector<int32_t> q;   // cap x qCols (3 cart / 2 sphere)
		std::vector<float>   r;   // cap (sphere only)
	};

	//! Computes the window machinery of one branch at one level (exactly
	//! _sphere_windows_sorted + the padding/cap checks + RPE quantization)
	/** xBr: nReal*3 branch coordinates; sphere: the radius r goes into the
	    sph_r tensor instead of a third quant column. **/
	SphereBranchTensors sphereBranchTensors(const std::vector<float>& xBr, const std::array<float, 3>& wsize, const std::array<float, 3>& qsize, int64_t cap, int64_t nReal, int64_t wCap, int64_t mCap, bool sphere, const std::string& branch, int level)
	{
		const size_t n = static_cast<size_t>(nReal);
		// window partition: floor((coords - min) / window), non-negative
		float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
		for (size_t i = 0; i < n; i++)
		{
			for (int d = 0; d < 3; d++)
			{
				mn[d] = std::min(mn[d], xBr[i * 3 + d]);
			}
		}
		std::vector<int64_t> fl(n * 3);
		for (size_t i = 0; i < n; i++)
		{
			for (int d = 0; d < 3; d++)
			{
				fl[i * 3 + d] = static_cast<int64_t>(std::floor((xBr[i * 3 + d] - mn[d]) / wsize[d]));
			}
		}
		SphereLexPartition part;
		sphereLexPartition(fl, part);
		const int64_t wl = static_cast<int64_t>(part.counts.size());

		// capacity checks, in the reference's order and wording
		int64_t pairSum = 0;
		for (int64_t c : part.counts)
		{
			pairSum += c * c;
		}
		if (pairSum > mCap)
		{
			// the M capacity is a runtime safety limit (it is not a graph input
			// dimension), so exceeding it is a performance warning, not a fatal
			// error: the graph's fixed tensors (C, W) still bound the memory.
			std::fprintf(stderr, "qSemSeg: level %d branch %s: %lld pairs > cap %lld - continuing; consider raising adapter.caps.M in the manifest\n", level, branch.c_str(), static_cast<long long>(pairSum), static_cast<long long>(mCap));
		}
		if (wl > wCap)
		{
			throw std::runtime_error("level " + std::to_string(level) + " branch " + branch + ": " + std::to_string(wl) + " windows > cap " + std::to_string(wCap) + " - re-export with larger caps");
		}
		const int64_t nPad = cap - nReal;
		if (wl + nPad > wCap)
		{
			throw std::runtime_error("level " + std::to_string(level) + " branch " + branch + ": " + std::to_string(wl) + " windows + " + std::to_string(nPad) + " padding > window cap " + std::to_string(wCap) + " - re-export with larger caps");
		}

		SphereBranchTensors t;
		// counts: real windows, then count-1 phantom windows for the padding
		// rows, then zeros up to the fixed Wcap
		t.cnt.resize(static_cast<size_t>(wCap), 0);
		for (int64_t w = 0; w < wl; w++)
		{
			t.cnt[static_cast<size_t>(w)] = static_cast<int32_t>(part.counts[static_cast<size_t>(w)]);
		}
		for (int64_t p = 0; p < nPad; p++)
		{
			t.cnt[static_cast<size_t>(wl + p)] = 1;
		}
		// sort/inv over all rows (junk rows = identity tail)
		t.srt.resize(static_cast<size_t>(cap));
		for (int64_t p = 0; p < nReal; p++)
		{
			t.srt[static_cast<size_t>(p)] = static_cast<int32_t>(part.sort[static_cast<size_t>(p)]);
		}
		for (int64_t p = nReal; p < cap; p++)
		{
			t.srt[static_cast<size_t>(p)] = static_cast<int32_t>(p);
		}
		t.inv.assign(static_cast<size_t>(cap), 0);
		for (int64_t p = 0; p < cap; p++)
		{
			t.inv[static_cast<size_t>(t.srt[static_cast<size_t>(p)])] = static_cast<int32_t>(p);
		}
		// RPE quantization per SORTED row over the real rows; padding rows get
		// zeros (they are isolated count-1 windows, never read)
		std::vector<float> xSorted(n * 3);
		for (size_t p = 0; p < n; p++)
		{
			const size_t row = static_cast<size_t>(part.sort[p]);
			std::memcpy(xSorted.data() + p * 3, xBr.data() + row * 3, 3 * sizeof(float));
		}
		std::vector<int64_t> q3;
		sphereQuant(xSorted, wsize, qsize, q3);
		const int qCols = sphere ? 2 : 3;
		t.q.assign(static_cast<size_t>(cap) * qCols, 0);
		for (int64_t p = 0; p < nReal; p++)
		{
			for (int c = 0; c < qCols; c++)
			{
				t.q[static_cast<size_t>(p) * qCols + c] = static_cast<int32_t>(q3[static_cast<size_t>(p) * 3 + c]);
			}
		}
		if (sphere)
		{
			// the radius axis is kept RAW (in SORTED row order); the graph
			// computes its own table index (exponential_split + clamp)
			t.r.assign(static_cast<size_t>(cap), 0.f);
			for (int64_t p = 0; p < nReal; p++)
			{
				t.r[static_cast<size_t>(p)] = xSorted[static_cast<size_t>(p) * 3 + 2];
			}
		}
		return t;
	}
} // namespace

void SphereFormerAdapter::applyDefaults(ModelSpec& spec) const
{
	// SphereFormer preprocessing: origin-anchored voxel subsampling (the
	// reference's grid_subsampling floors p/dl in ORIGINAL coordinates - no
	// min shift) and per-patch mid-range recentering done by THIS adapter
	// (not the mean recenter). Only fields no manifest source provided.
	if (!(spec.manifestFields & ModelSpec::mbShiftMin))
	{
		spec.shiftMin = false;
	}
	if (!(spec.manifestFields & ModelSpec::mbRecenter))
	{
		spec.recenter = false;
	}
}

std::string SphereFormerAdapter::validate(const ModelSpec& spec, const IOContract& io) const
{
	// the manifest must carry the per-level window/quant constants and the
	// structural capacities: a wrong guess would feed the attention a
	// partition the exported graph cannot represent
	if (spec.sphereCarWindow.size() != 15 || spec.sphereCarQuant.size() != 15 || spec.sphereSphWindow.size() != 15 || spec.sphereSphQuant.size() != 15)
	{
		return "SphereFormer models require the manifest adapter.window block (car_window / car_quant / sph_window / sph_quant, 5 levels x 3 axes)";
	}
	if (spec.sphereCapsC.size() != 4 || spec.sphereCapsW.size() != 10 || spec.sphereCapsM.size() != 10)
	{
		return "SphereFormer models require the manifest adapter.caps block (C with 4 row capacities, W and M with the car0..car4 / sph0..sph4 keys)";
	}

	const auto expected = sphereExpectedInputs();
	if (io.inputNames.size() != expected.size())
	{
		return "SphereFormer graphs expose exactly " + std::to_string(expected.size()) + " inputs, the session declares " + std::to_string(io.inputNames.size());
	}

	// every expected input must appear exactly once (ORT binds by name)
	std::map<std::string, int> seen;
	for (const std::string& name : io.inputNames)
	{
		if (!expected.count(name))
		{
			return "unexpected input '" + name + "' for a SphereFormer graph";
		}
		seen[name]++;
	}
	for (const auto& entry : seen)
	{
		if (entry.second != 1)
		{
			return "input '" + entry.first + "' appears " + std::to_string(entry.second) + " times in the graph";
		}
	}

	// the 'feat' input defines level 0: [num_points, in_channels], both fixed
	size_t featIdx = std::numeric_limits<size_t>::max();
	for (size_t in = 0; in < io.inputNames.size(); in++)
	{
		if (io.inputNames[in] == "feat")
		{
			featIdx = in;
			break;
		}
	}
	if (featIdx == std::numeric_limits<size_t>::max())
	{
		return "SphereFormer graphs require a 'feat' input";
	}
	if (io.inputDtypes[featIdx] != "float32")
	{
		return "input 'feat' must be float32 (the graph declares " + io.inputDtypes[featIdx] + ")";
	}
	const std::vector<int64_t>& featShape = io.inputShapes[featIdx];
	if (featShape.size() != 2 || featShape[0] <= 0 || featShape[1] <= 0)
	{
		return "input 'feat' must have the fixed shape [num_points, in_channels]";
	}
	if (featShape[1] != spec.inChannels)
	{
		return "the graph's 'feat' input expects " + std::to_string(featShape[1]) + " channels, but the manifest declares in_channels=" + std::to_string(spec.inChannels);
	}
	int64_t caps[5];
	caps[0] = featShape[0];
	for (int l = 1; l < 5; l++)
	{
		caps[l] = spec.sphereCapsC[static_cast<size_t>(l) - 1];
		if (caps[l] <= 0)
		{
			return "adapter.caps.C must carry positive row capacities for levels 1..4";
		}
	}

	// per-input dtype + fixed-shape checks (all shapes are concrete in a
	// fixed-shape export; the manifest caps must MATCH the graph's shapes,
	// otherwise this manifest does not belong to this export)
	for (size_t in = 0; in < io.inputNames.size(); in++)
	{
		const std::string&          name        = io.inputNames[in];
		const SphereSlot&           slot        = expected.at(name);
		const std::vector<int64_t>& shape       = io.inputShapes[in];
		const std::string&          dtype       = io.inputDtypes[in];
		const std::string           expectDtype = (slot.kind == SphereSlot::Feat || slot.kind == SphereSlot::SphR) ? "float32" : "int32";
		if (dtype != expectDtype)
		{
			return "input '" + name + "' must be " + expectDtype + " (the graph declares " + dtype + ")";
		}
		int64_t rows = -1;
		int64_t cols = 1;
		switch (slot.kind)
		{
		case SphereSlot::Feat:
			continue; // checked above
		case SphereSlot::SubmNin:
			rows = caps[slot.level];
			cols = 27;
			break;
		case SphereSlot::CarCnt:
		case SphereSlot::SphCnt:
			rows = sphereBranchCap(spec.sphereCapsW, slot.level, slot.kind == SphereSlot::SphCnt);
			cols = 1;
			break;
		case SphereSlot::CarSrt:
		case SphereSlot::CarInv:
		case SphereSlot::SphSrt:
		case SphereSlot::SphInv:
		case SphereSlot::SphR:
		case SphereSlot::DownPerm:
			rows = caps[slot.level];
			cols = 1;
			break;
		case SphereSlot::CarQ:
			rows = caps[slot.level];
			cols = 3;
			break;
		case SphereSlot::SphQ:
			rows = caps[slot.level];
			cols = 2;
			break;
		case SphereSlot::DownNin:
		case SphereSlot::InvNin:
			rows = slot.kind == SphereSlot::DownNin ? caps[slot.level + 1] : caps[slot.level];
			cols = 8;
			break;
		case SphereSlot::DownBnd:
			rows = caps[slot.level] + 1;
			cols = 1;
			break;
		}
		const bool shapeOk = shape.size() == (cols > 1 ? 2 : 1) && shape[0] == rows && (cols == 1 || shape[1] == cols);
		if (!shapeOk)
		{
			std::string expect = cols > 1 ? "[" + std::to_string(rows) + ", " + std::to_string(cols) + "]" : "[" + std::to_string(rows) + "]";
			return "input '" + name + "' must have the fixed shape " + expect + " (the graph declares [" + std::to_string(shape.empty() ? -1 : shape[0]) + (shape.size() > 1 ? ", " + std::to_string(shape[1]) : "") + "]) - the manifest adapter.caps do not match this export";
		}
	}

	return {};
}

std::string SphereFormerAdapter::validateCloudFeatures(const ModelSpec& spec, bool hasColors, bool hasIntensity) const
{
	// xyz + optional extra channels (same rules as the generic family)
	if (spec.inChannels < 3)
	{
		return "SphereFormer models require in_channels >= 3 (xyz)";
	}
	const int extras = spec.inChannels - 3;
	if (extras == 0)
	{
		return {}; // xyz-only: works on any point cloud
	}
	std::vector<std::string> columns = spec.featureColumns;
	if (columns.empty())
	{
		if (extras >= 1)
		{
			columns.push_back("intensity");
		}
		if (extras >= 2)
		{
			columns.push_back("rgb");
		}
	}
	// expand "rgb" to 3 channels (matches sphereExtraChannels)
	std::vector<std::string> expanded;
	for (const auto& col : columns)
	{
		if (col == "rgb")
		{
			expanded.push_back("r");
			expanded.push_back("g");
			expanded.push_back("b");
		}
		else
		{
			expanded.push_back(col);
		}
	}
	columns = expanded;
	if (static_cast<int>(columns.size()) < extras)
	{
		return "the model needs " + std::to_string(extras) + " extra feature channel(s) (after xyz) but only " + std::to_string(columns.size()) + " are declared";
	}
	for (int e = 0; e < extras; e++)
	{
		if ((columns[e] == "r" || columns[e] == "g" || columns[e] == "b") && !hasColors)
		{
			return "feature 'rgb' is required by the model but the cloud has no colors";
		}
		if ((columns[e] == "intensity" || columns[e] == "i") && !hasIntensity)
		{
			return "feature 'intensity' is required by the model but the cloud provides no scalar field for it";
		}
	}
	return {};
}

int64_t SphereFormerAdapter::staticNumPointsFromIO(const IOContract& io) const
{
	// feat is [num_points, in_channels]: the ROW count is the points axis
	for (size_t in = 0; in < io.inputNames.size(); in++)
	{
		if (io.inputNames[in] == "feat" && io.inputDtypes[in] == "float32" && !io.inputShapes[in].empty() && io.inputShapes[in][0] > 0)
		{
			return io.inputShapes[in][0];
		}
	}
	return 0;
}

Patch SphereFormerAdapter::buildPatch(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, std::mt19937_64& /*rng*/, const std::vector<int64_t>* /*fpsStarts*/, const float* /*fixedMean*/) const
{
	// no FPS and no mean recenter: the patch structure is fully determined by
	// the sampled indices (the fpsStarts/fixedMean parity overrides of the
	// FKAConv family have no counterpart here)
	const size_t k = idxs.size();
	if (k < 2)
	{
		throw std::runtime_error("SphereFormer patches need at least 2 points");
	}
	if (spec.sphereCarWindow.size() != 15 || spec.sphereCarQuant.size() != 15 || spec.sphereSphWindow.size() != 15 || spec.sphereSphQuant.size() != 15 || spec.sphereCapsC.size() != 4 || spec.sphereCapsW.size() != 10 || spec.sphereCapsM.size() != 10)
	{
		throw std::runtime_error("SphereFormer models require the manifest adapter.window and adapter.caps blocks");
	}
	if (spec.staticNumPoints > 0 && static_cast<int64_t>(k) != spec.staticNumPoints)
	{
		throw std::runtime_error("SphereFormer graphs are exported for exactly " + std::to_string(spec.staticNumPoints) + " points per patch, got " + std::to_string(k));
	}
	if (spec.inChannels < 3)
	{
		throw std::runtime_error("SphereFormer models require in_channels >= 3 (xyz)");
	}

	// ---- gather + mid-range recenter (build_patch_inputs) ----
	// the offset is snapped to a multiple of the model quant size: the
	// subsampled cloud is one point per voxel, so an integer shift in voxel
	// space keeps the voxel indices unique
	std::vector<float> pts(k * 3);
	for (size_t i = 0; i < k; i++)
	{
		std::memcpy(pts.data() + i * 3, pre.subPoints.data() + static_cast<size_t>(idxs[i]) * 3, 3 * sizeof(float));
	}
	const float* q0 = spec.sphereCarQuant.data(); // level-0 quant == model.quant_size
	float        mn[3], mx[3], shift[3];
	for (int d = 0; d < 3; d++)
	{
		mn[d] = pts[d];
		mx[d] = pts[d];
	}
	for (size_t i = 1; i < k; i++)
	{
		for (int d = 0; d < 3; d++)
		{
			mn[d] = std::min(mn[d], pts[i * 3 + d]);
			mx[d] = std::max(mx[d], pts[i * 3 + d]);
		}
	}
	for (int d = 0; d < 3; d++)
	{
		// np.round -> rint (round half to even), all steps float32
		shift[d] = std::rint(((mn[d] + mx[d]) / 2.0f) / q0[d]) * q0[d];
	}
	std::vector<float> sampled(k * 3);
	for (size_t i = 0; i < k; i++)
	{
		for (int d = 0; d < 3; d++)
		{
			sampled[i * 3 + d] = pts[i * 3 + d] - shift[d];
		}
	}

	// ---- features: the recentered xyz (+ extra channels, /255 heuristic) --
	std::vector<float> feat(k * static_cast<size_t>(spec.inChannels));
	{
		const int extras = spec.inChannels - 3;
		std::memcpy(feat.data(), sampled.data(), k * 3 * sizeof(float));
		if (extras > 0)
		{
			std::vector<float> extrasData = sphereExtraChannels(spec, pre, idxs, extras);
			float              extraMax   = 0.f;
			bool               any        = false;
			for (float v : extrasData)
			{
				any      = true;
				extraMax = std::max(extraMax, v);
			}
			const bool divideBy255 = any && extraMax > 1.0f;
			for (size_t i = 0; i < k; i++)
			{
				for (int e = 0; e < extras; e++)
				{
					feat[i * spec.inChannels + 3 + e] = divideBy255 ? extrasData[i * extras + e] / 255.f : extrasData[i * extras + e];
				}
			}
		}
	}

	// ---- per-level structural tensors ------------------------------------
	// per level: two window branches (cart on xyz, sphere on cart2sphere),
	// the SubM gather table, then the stride-2 down/inverse tables and the
	// scatter-mean that produces the next level's coordinates
	std::vector<std::vector<float>> xyzLevels;
	xyzLevels.push_back(std::move(sampled));
	std::vector<int64_t>                curVox; // current level's voxel triples
	std::vector<std::array<int64_t, 3>> spatial;

	// level-0 voxel indices: floor((xyz - min) / quant), exactly
	// _build_sparse_tensor_internal's grid. The minimum is over the
	// RECENTERED coordinates (xyz0.min(0) in the reference, where xyz0 is
	// already mid-range-centered) - recomputing it here keeps the f32
	// rounding identical to numpy's (the recentered min is not the raw min
	// bit-for-bit).
	{
		curVox.resize(k * 3);
		float gMin[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
		for (size_t i = 0; i < k; i++)
		{
			for (int d = 0; d < 3; d++)
			{
				gMin[d] = std::min(gMin[d], xyzLevels[0][i * 3 + d]);
			}
		}
		float vMax[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
		for (size_t i = 0; i < k; i++)
		{
			for (int d = 0; d < 3; d++)
			{
				const float v = static_cast<float>(std::floor((xyzLevels[0][i * 3 + d] - gMin[d]) / q0[d]));
				if (!(std::fabs(v) < 1.0e9f))
				{
					throw std::runtime_error("patch coordinates do not fit the graph's voxel grid (extent too large for quant_size " + std::to_string(q0[d]) + ")");
				}
				curVox[i * 3 + d] = static_cast<int64_t>(v);
				vMax[d]           = std::max(vMax[d], v);
			}
		}
		std::array<int64_t, 3> sp0 = {1, 1, 1};
		for (int d = 0; d < 3; d++)
		{
			const int64_t ext = static_cast<int64_t>(vMax[d]) + 1; // max + 1
			sp0[d]            = std::max<int64_t>(((ext + 31) / 32) * 32, 32);
		}
		spatial.push_back(sp0);
	}

	// tensor storage per slot kind + level
	std::vector<std::vector<int32_t>> submNin(5), carCnt(5), carSrt(5), carInv(5), carQ(5), sphCnt(5), sphSrt(5), sphInv(5), sphQ(5), downNin(4), downPerm(4), downBnd(4), invNin(4);
	std::vector<std::vector<float>>   sphR(5);

	for (int l = 0; l < 5; l++)
	{
		const int64_t             nReal = (l == 0) ? static_cast<int64_t>(k) : static_cast<int64_t>(curVox.size()) / 3;
		const int64_t             cap   = sphereLevelCap(spec, l, static_cast<int64_t>(k));
		const std::vector<float>& x     = xyzLevels[static_cast<size_t>(l)];

		// two branches: cart on the cartesian xyz, sphere on cart2sphere(xyz)
		std::vector<float> sphereCoords;
		for (int b = 0; b < 2; b++)
		{
			const bool                isSphere = b == 1;
			const std::string         branch   = isSphere ? "sph" : "car";
			const float*              wsize    = (isSphere ? spec.sphereSphWindow : spec.sphereCarWindow).data() + l * 3;
			const float*              qsize    = (isSphere ? spec.sphereSphQuant : spec.sphereCarQuant).data() + l * 3;
			const int64_t             wCap     = sphereBranchCap(spec.sphereCapsW, l, isSphere);
			const int64_t             mCap     = sphereBranchCap(spec.sphereCapsM, l, isSphere);
			const std::vector<float>* xBr      = &x;
			if (isSphere)
			{
				sphereCart2Sphere(x, sphereCoords);
				xBr = &sphereCoords;
			}
			std::array<float, 3> wArr = {wsize[0], wsize[1], wsize[2]};
			std::array<float, 3> qArr = {qsize[0], qsize[1], qsize[2]};
			SphereBranchTensors  t    = sphereBranchTensors(*xBr, wArr, qArr, cap, nReal, wCap, mCap, isSphere, branch, l);
			if (isSphere)
			{
				sphCnt[l] = std::move(t.cnt);
				sphSrt[l] = std::move(t.srt);
				sphInv[l] = std::move(t.inv);
				sphQ[l]   = std::move(t.q);
				sphR[l]   = std::move(t.r);
			}
			else
			{
				carCnt[l] = std::move(t.cnt);
				carSrt[l] = std::move(t.srt);
				carInv[l] = std::move(t.inv);
				carQ[l]   = std::move(t.q);
			}
		}

		// SubM gather table over the CURRENT level's voxels (the running
		// spatial shape: level 0 from the voxel extent, then halved per level)
		sphereSubmGatherTable(curVox, spatial[static_cast<size_t>(l)], cap, submNin[l]);

		if (l == 4)
		{
			break;
		}

		// stride-2 hierarchy: parent voxels = child voxels // 2 (all
		// components non-negative, so // 2 == plain /2)
		std::vector<int64_t> half(curVox.size());
		for (size_t i = 0; i < curVox.size(); i++)
		{
			half[i] = curVox[i] / 2;
		}
		SphereLexPartition parentPart;
		sphereLexPartition(half, parentPart);
		const int64_t nNextRaw  = static_cast<int64_t>(parentPart.uniq.size()) / 3;
		const int64_t parentCap = spec.sphereCapsC[static_cast<size_t>(l)];
		int64_t       nNext     = nNextRaw;
		if (nNext > parentCap)
		{
			// the C capacity is a graph shape constraint, so a large overflow
			// (more than ~10% over) cannot be worked around here - re-export
			// with a larger cap. A small overflow is truncated to the cap
			// (dropping the least-significant parent voxels) so the model still
			// runs on denser clouds without a re-export.
			if (nNext > parentCap + parentCap / 10)
			{
				throw std::runtime_error("level " + std::to_string(l + 1) + ": " + std::to_string(nNext) + " rows > capacity " + std::to_string(parentCap) + " - increase the capacity and re-export");
			}
			std::fprintf(stderr, "qSemSeg: level %d: %lld rows > capacity %lld - truncating to cap (re-export with larger caps for exact results)\n", l + 1, static_cast<long long>(nNext), static_cast<long long>(parentCap));
			nNext = parentCap;
			// truncate the parent voxels to the cap (lex order: drop the highest)
			parentPart.uniq.resize(static_cast<size_t>(nNext) * 3);
			// remap: any child whose parent rank >= nNext is clamped to nNext-1
			for (auto& r : parentPart.rank)
			{
				if (r >= nNext)
				{
					r = nNext - 1;
				}
			}
		}
		// unique child voxels (lex order) for the down-table segments
		SphereLexPartition childPart;
		sphereLexPartition(curVox, childPart);
		sphereDownGatherTables(childPart, parentPart.rank, cap, spec.sphereCapsC[static_cast<size_t>(l)], downNin[l], downPerm[l], downBnd[l]);
		sphereInverseGatherTable(curVox, parentPart.rank, spec.sphereCapsC[static_cast<size_t>(l)], cap, invNin[l]);

		// next level's spatial shape: halved per axis (spconv stride-2 grid)
		spatial.push_back({spatial.back()[0] / 2, spatial.back()[1] / 2, spatial.back()[2] / 2});

		// next level's voxels and coordinates (scatter-mean of the children)
		std::vector<float> nextXyz;
		sphereScatterMean(x, parentPart.rank, static_cast<size_t>(nNext), nextXyz);
		curVox = std::move(parentPart.uniq);
		xyzLevels.push_back(std::move(nextXyz));
	}

	// ---- distribute into the patch's per-input stores --------------------
	Patch patch;
	patch.idxs = idxs;
	patch.adapterF32.assign(spec.inputNames.size(), {});
	patch.adapterI32.assign(spec.inputNames.size(), {});
	const auto expected = sphereExpectedInputs();
	for (size_t in = 0; in < spec.inputNames.size(); in++)
	{
		const auto it = expected.find(spec.inputNames[in]);
		if (it == expected.end())
		{
			throw std::runtime_error("unexpected input '" + spec.inputNames[in] + "' for a SphereFormer graph");
		}
		const SphereSlot& slot = it->second;
		switch (slot.kind)
		{
		case SphereSlot::Feat:
			patch.adapterF32[in] = feat;
			break;
		case SphereSlot::SubmNin:
			patch.adapterI32[in] = submNin[slot.level];
			break;
		case SphereSlot::CarCnt:
			patch.adapterI32[in] = carCnt[slot.level];
			break;
		case SphereSlot::CarSrt:
			patch.adapterI32[in] = carSrt[slot.level];
			break;
		case SphereSlot::CarInv:
			patch.adapterI32[in] = carInv[slot.level];
			break;
		case SphereSlot::CarQ:
			patch.adapterI32[in] = carQ[slot.level];
			break;
		case SphereSlot::SphCnt:
			patch.adapterI32[in] = sphCnt[slot.level];
			break;
		case SphereSlot::SphSrt:
			patch.adapterI32[in] = sphSrt[slot.level];
			break;
		case SphereSlot::SphInv:
			patch.adapterI32[in] = sphInv[slot.level];
			break;
		case SphereSlot::SphQ:
			patch.adapterI32[in] = sphQ[slot.level];
			break;
		case SphereSlot::SphR:
			patch.adapterF32[in] = sphR[slot.level];
			break;
		case SphereSlot::DownNin:
			patch.adapterI32[in] = downNin[slot.level];
			break;
		case SphereSlot::DownPerm:
			patch.adapterI32[in] = downPerm[slot.level];
			break;
		case SphereSlot::DownBnd:
			patch.adapterI32[in] = downBnd[slot.level];
			break;
		case SphereSlot::InvNin:
			patch.adapterI32[in] = invNin[slot.level];
			break;
		}
	}
	return patch;
}

BatchBuffers SphereFormerAdapter::assembleBatch(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage) const
{
	const auto fail = [errorMessage](const std::string& message) -> BatchBuffers
	{
		if (errorMessage)
		{
			*errorMessage = message;
		}
		return {};
	};

	if (patches.size() != 1)
	{
		return fail("SphereFormer graphs are exported with the batch axis fixed to 1 - run one patch per forward (patch batch size must be 1)");
	}
	const Patch& patch = patches[0];
	if (patch.adapterF32.size() != spec.inputNames.size() || patch.adapterI32.size() != spec.inputNames.size())
	{
		return fail("SphereFormer patch tensors do not match the graph's input count");
	}

	BatchBuffers buffers;
	buffers.shapes.resize(spec.inputNames.size());
	buffers.f32.resize(spec.inputNames.size());
	buffers.i32.resize(spec.inputNames.size());
	buffers.i64.resize(spec.inputNames.size());
	for (size_t in = 0; in < spec.inputNames.size(); in++)
	{
		// the export is fully fixed-shape: resolving with B=1 is a no-op, but
		// it keeps the shapes consistent with the rest of the pipeline
		buffers.shapes[in] = OrtSession::resolveShape(spec.inputShapes[in], 1, static_cast<int64_t>(patch.idxs.size()));
		int64_t perPatch   = 1;
		for (size_t d = 1; d < buffers.shapes[in].size(); d++)
		{
			perPatch *= buffers.shapes[in][d];
		}
		const size_t       expected = static_cast<size_t>(buffers.shapes[in][0]) * static_cast<size_t>(perPatch);
		const std::string& dtype    = spec.inputDtypes[in];
		if (dtype == "float32")
		{
			if (patch.adapterF32[in].size() != expected)
			{
				return fail("input '" + spec.inputNames[in] + "' payload size " + std::to_string(patch.adapterF32[in].size()) + " does not match the graph shape");
			}
			buffers.f32[in] = patch.adapterF32[in];
		}
		else if (dtype == "int32")
		{
			if (patch.adapterI32[in].size() != expected)
			{
				return fail("input '" + spec.inputNames[in] + "' payload size " + std::to_string(patch.adapterI32[in].size()) + " does not match the graph shape");
			}
			buffers.i32[in] = patch.adapterI32[in];
		}
		else
		{
			return fail("input '" + spec.inputNames[in] + "' has unsupported dtype '" + dtype + "'");
		}
	}
	return buffers;
}
