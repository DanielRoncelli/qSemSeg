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

#include "Preprocess.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace
{
	//! Minimum index count for a parallel-for to be worth its wake-up overhead
	constexpr size_t kParallelMinCount = 4096;

	//! `count` distinct indices of 0..n-1, drawn uniformly with the given rng
	/** Partial Fisher-Yates over an index array: deterministic for a given
	    seed (the parity suite pins counts and sets, not orders). When count
	    >= n the result is simply every index. **/
	std::vector<int64_t> randomSubset(size_t n, size_t count, std::mt19937_64& rng)
	{
		std::vector<int64_t> idxs(n);
		for (size_t p = 0; p < n; ++p)
		{
			idxs[p] = static_cast<int64_t>(p);
		}
		if (count < n)
		{
			for (size_t i = 0; i < count; ++i)
			{
				std::uniform_int_distribution<size_t> d(i, n - 1);
				std::swap(idxs[i], idxs[d(rng)]);
			}
			idxs.resize(count);
		}
		return idxs;
	}

	//! Minimal persistent worker pool for parallel-for loops
	/** Workers are created once and reused: spawning a thread per iteration
	    would dominate the per-iteration work (FPS updates one distance per
	    point per iteration). Each parallelFor call hands worker w the disjoint
	    range [w*chunk, min(count, (w+1)*chunk)); results are exact because the
	    bodies never share output elements. Concurrent parallelFor calls are
	    serialized. NEVER call parallelFor from inside a task body: the nested
	    call would block on the call mutex the outer dispatch still holds
	    (self-deadlock). Worker exceptions are stashed and rethrown by the
	    caller instead of escaping the thread (which would terminate the
	    process). **/
	class ParallelPool
	{
	  public:
		static ParallelPool& instance()
		{
			static ParallelPool pool;
			return pool;
		}

		void parallelFor(size_t count, const std::function<void(size_t, size_t)>& body)
		{
			if (count == 0)
			{
				return;
			}
			std::lock_guard<std::mutex> callLock(m_callMutex);
			if (count < kParallelMinCount || m_workers.empty())
			{
				body(0, count);
				return;
			}
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_task  = &body;
				m_count = count;
				m_chunk = (count + m_workers.size() - 1) / m_workers.size();
				m_generation++;
				m_remaining      = static_cast<int>(m_workers.size());
				m_firstException = nullptr;
			}
			m_startCv.notify_all();

			std::unique_lock<std::mutex> lock(m_mutex);
			m_doneCv.wait(lock, [this]
			              { return m_remaining == 0; });
			if (m_firstException)
			{
				std::exception_ptr first = m_firstException;
				m_firstException         = nullptr;
				std::rethrow_exception(first);
			}
		}

	  private:
		ParallelPool()
		{
			unsigned int hw = std::thread::hardware_concurrency();
			if (hw == 0)
			{
				hw = 4;
			}
			const unsigned int nWorkers = std::min(hw, 16u);
			m_workers.reserve(nWorkers);
			for (unsigned int w = 0; w < nWorkers; w++)
			{
				m_workers.emplace_back([this, w]
				                       { workerLoop(w); });
			}
			// Do not return before every worker is parked with a consistent
			// generation snapshot. A worker thread that starts late would
			// otherwise read m_generation AFTER the first dispatch and sleep
			// through that call, leaving m_remaining stuck above zero.
			std::unique_lock<std::mutex> lock(m_mutex);
			m_regCv.wait(lock, [this]
			             { return m_registered == static_cast<int>(m_workers.size()); });
		}

		~ParallelPool()
		{
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_stop = true;
			}
			m_startCv.notify_all();
			for (auto& worker : m_workers)
			{
				worker.join();
			}
		}

		void workerLoop(unsigned int id)
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			// Register with the constructor and take the generation snapshot
			// under the SAME lock hold: the constructor returns only once
			// every worker is parked with seen == m_generation, so no worker
			// can miss a dispatched call.
			++m_registered;
			m_regCv.notify_all();
			uint64_t seen = m_generation;
			for (;;)
			{
				m_startCv.wait(lock, [this, &seen]
				               { return m_generation != seen || m_stop; });
				if (m_stop)
				{
					return;
				}
				seen                     = m_generation;
				const auto         task  = *m_task;
				const size_t       begin = id * m_chunk;
				const size_t       end   = std::min(m_count, begin + m_chunk);
				std::exception_ptr localError;
				lock.unlock();
				try
				{
					if (begin < end)
					{
						task(begin, end);
					}
				}
				catch (...)
				{
					// do not let the exception escape the thread: keep the
					// remaining-accounting invariant and hand it to the caller
					localError = std::current_exception();
				}
				lock.lock();
				if (localError && !m_firstException)
				{
					m_firstException = localError;
				}
				if (--m_remaining == 0)
				{
					m_doneCv.notify_all();
				}
			}
		}

		std::vector<std::thread>                   m_workers;
		std::mutex                                 m_callMutex;
		std::mutex                                 m_mutex;
		std::condition_variable                    m_startCv;
		std::condition_variable                    m_doneCv;
		std::condition_variable                    m_regCv;
		int                                        m_registered     = 0;
		const std::function<void(size_t, size_t)>* m_task           = nullptr;
		size_t                                     m_count          = 0;
		size_t                                     m_chunk          = 0;
		uint64_t                                   m_generation     = 0;
		int                                        m_remaining      = 0;
		bool                                       m_stop           = false;
		std::exception_ptr                         m_firstException = nullptr;
	};
} // namespace

std::vector<int64_t> knnQuery(const KDTree& tree, size_t n, const float* q, int k, std::vector<float>* distsOut)
{
	const int            kq = std::min<int64_t>(k, static_cast<int64_t>(n));
	std::vector<int64_t> out(k, 0);
	if (kq <= 0)
	{
		return out;
	}
	std::vector<size_t>            idx(kq);
	std::vector<float>             dist(kq);
	nanoflann::KNNResultSet<float> rs(kq);
	rs.init(idx.data(), dist.data());
	tree.findNeighbors(rs, q, nanoflann::SearchParameters());
	for (int i = 0; i < kq; i++)
	{
		out[i] = static_cast<int64_t>(idx[i]);
	}
	for (int i = kq; i < k; i++)
	{
		out[i] = out[kq - 1]; // pad by repeating last
	}
	if (distsOut)
	{
		distsOut->assign(k, 0.f);
		for (int i = 0; i < kq; i++)
		{
			(*distsOut)[i] = dist[i];
		}
		for (int i = kq; i < k; i++)
		{
			(*distsOut)[i] = dist[kq - 1];
		}
	}
	return out;
}

std::vector<int64_t> knnQueryBatch(const KDTree& tree, size_t n, const std::vector<float>& queries, int k, std::vector<float>* distsOut)
{
	const size_t         q = queries.size() / 3;
	std::vector<int64_t> out(q * static_cast<size_t>(k), 0);
	if (distsOut)
	{
		distsOut->assign(q * static_cast<size_t>(k), 0.f);
	}
	// queries are independent and nanoflann's findNeighbors is const, so the
	// batch parallelizes with EXACT per-query results (disjoint output slices)
	auto queryRange = [&](size_t begin, size_t end)
	{
		for (size_t i = begin; i < end; i++)
		{
			std::vector<float> d;
			auto               r = knnQuery(tree, n, queries.data() + i * 3, k, distsOut ? &d : nullptr);
			std::memcpy(out.data() + i * k, r.data(), r.size() * sizeof(int64_t));
			if (distsOut)
			{
				std::memcpy(distsOut->data() + i * k, d.data(), d.size() * sizeof(float));
			}
		}
	};
	ParallelPool::instance().parallelFor(q, queryRange);
	return out;
}

std::vector<int64_t> fpsIndices(const std::vector<float>& pts, size_t nSamples, std::mt19937_64& rng, int64_t fixedStart)
{
	const size_t n = pts.size() / 3;
	if (n <= nSamples)
	{
		// same as the Python reference (_farthest_point_sample): keep all points
		std::vector<int64_t> all(n);
		for (size_t i = 0; i < n; i++)
		{
			all[i] = static_cast<int64_t>(i);
		}
		return all;
	}
	size_t farthest = 0;
	if (fixedStart >= 0 && static_cast<size_t>(fixedStart) < n)
	{
		farthest = static_cast<size_t>(fixedStart);
	}
	else
	{
		std::uniform_int_distribution<size_t> pick(0, n - 1);
		farthest = pick(rng);
	}
	std::vector<int64_t> centroids(nSamples);
	std::vector<float>   distances(n, std::numeric_limits<float>::infinity());
	for (size_t i = 0; i < nSamples; i++)
	{
		centroids[i]   = static_cast<int64_t>(farthest);
		const float* c = pts.data() + farthest * 3;
		// kept serial on purpose: the update is one distance per point per
		// FPS iteration, so a parallel-for here would pay a pool round-trip
		// (16 wakeups) for ~16 microseconds of arithmetic
		for (size_t p = 0; p < n; p++)
		{
			float dx = pts[p * 3 + 0] - c[0];
			float dy = pts[p * 3 + 1] - c[1];
			float dz = pts[p * 3 + 2] - c[2];
			float d  = dx * dx + dy * dy + dz * dz;
			if (d < distances[p])
			{
				distances[p] = d;
			}
		}
		farthest = static_cast<size_t>(std::distance(distances.begin(), std::max_element(distances.begin(), distances.end())));
	}
	return centroids;
}

namespace
{
	struct VoxelKey
	{
		int64_t x, y, z;
		bool    operator==(const VoxelKey& o) const
		{
			return x == o.x && y == o.y && z == o.z;
		}
	};

	struct VoxelKeyHash
	{
		size_t operator()(const VoxelKey& k) const
		{
			size_t h = std::hash<int64_t>()(k.x);
			h        = h * 1000003 ^ std::hash<int64_t>()(k.y);
			h        = h * 1000003 ^ std::hash<int64_t>()(k.z);
			return h;
		}
	};
} // namespace

CloudPre preprocessCloud(const std::vector<float>& xyz, const std::vector<float>& rgb, const std::vector<float>& intensity, const std::vector<int>& labels, double voxelSize, bool shiftMin, const SubsamplingParams& subsampling)
{
	const size_t n = xyz.size() / 3;
	CloudPre     pre;
	pre.projInds.resize(n);

	// ---- subset modes (none / random / fps): keep ORIGINAL points ----
	// Their projection map back to the original cloud is the identity (the
	// nearest "sub" point of an original point is itself - exact, no kNN).
	// Coordinates stay unshifted: patch features are built from subPoints, and
	// shifting would change them (shift_min is a voxel-hash preprocessing).
	const bool useSubset = subsampling.type != ModelSpec::SubsamplingType::Voxel;
	if (useSubset)
	{
		std::vector<int64_t> subsetIdxs;
		switch (subsampling.type)
		{
		case ModelSpec::SubsamplingType::None:
			subsetIdxs.resize(n);
			for (size_t p = 0; p < n; ++p)
			{
				subsetIdxs[p] = static_cast<int64_t>(p);
			}
			break;
		case ModelSpec::SubsamplingType::Random:
			if (!subsampling.rng || subsampling.target <= 0)
			{
				throw std::runtime_error("random subsampling needs a positive target and a seeded rng");
			}
			subsetIdxs = randomSubset(n, static_cast<size_t>(subsampling.target), *subsampling.rng);
			break;
		case ModelSpec::SubsamplingType::Fps:
			if (!subsampling.rng || subsampling.target <= 0)
			{
				throw std::runtime_error("fps subsampling needs a positive target and a seeded rng");
			}
			subsetIdxs = fpsIndices(xyz, static_cast<size_t>(subsampling.target), *subsampling.rng);
			break;
		default:
			break;
		}

		const size_t m = subsetIdxs.size();
		pre.subPoints.resize(m * 3);
		if (!rgb.empty())
		{
			pre.subRgb.resize(m * 3);
		}
		if (!intensity.empty())
		{
			pre.subIntensity.resize(m);
		}
		if (!labels.empty())
		{
			pre.subLabels.resize(m);
		}
		for (size_t o = 0; o < m; ++o)
		{
			const int64_t src = subsetIdxs[o];
			for (int d = 0; d < 3; ++d)
			{
				pre.subPoints[o * 3 + d] = xyz[src * 3 + d];
			}
			if (!rgb.empty())
			{
				for (int d = 0; d < 3; ++d)
				{
					pre.subRgb[o * 3 + d] = rgb[src * 3 + d];
				}
			}
			if (!intensity.empty())
			{
				pre.subIntensity[o] = intensity[src];
			}
			if (!labels.empty())
			{
				pre.subLabels[o] = labels[src];
			}
		}
		// projection map: a SELECTED original point maps to its own subset
		// position (distance 0, exact); the remaining originals kNN-project to
		// the nearest selected point - same semantics as the voxel path
		std::vector<int64_t> positionOf(n, -1);
		for (size_t o = 0; o < m; ++o)
		{
			positionOf[static_cast<size_t>(subsetIdxs[o])] = static_cast<int64_t>(o);
		}
		pre.adaptor = std::make_unique<PointCloudAdaptor>(PointCloudAdaptor{&pre.subPoints});
		pre.tree    = std::make_unique<KDTree>(3, *pre.adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		for (size_t p = 0; p < n; ++p)
		{
			const int64_t own = positionOf[p];
			pre.projInds[p]   = own >= 0 ? own : knnQuery(*pre.tree, m, xyz.data() + p * 3, 1)[0];
		}
		pre.M = pre.subPoints.size() / 3;
		return pre;
	}

	std::vector<float> pts = xyz;
	if (shiftMin && voxelSize > 0.0)
	{
		float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
		for (size_t p = 0; p < n; p++)
		{
			for (int d = 0; d < 3; d++)
			{
				mn[d] = std::min(mn[d], pts[p * 3 + d]);
			}
		}
		for (size_t p = 0; p < n; p++)
		{
			for (int d = 0; d < 3; d++)
			{
				pts[p * 3 + d] -= mn[d];
			}
		}
	}

	if (voxelSize > 0.0)
	{
		// voxel hash: floor(p / voxelSize) -> barycenter (points + mean features + majority label)
		// The key MUST be computed in double precision: the reference
		// (open3d.ml.contrib.subsample) floors (double)point / (double)sampleDl,
		// and a single-precision division puts points sitting exactly on a
		// voxel boundary into the wrong voxel (changes the sub point count!)
		struct Acc
		{
			double                       sum[3]     = {0, 0, 0};
			double                       fsumRgb[3] = {0, 0, 0};
			double                       fsumI      = 0;
			int                          count      = 0;
			std::unordered_map<int, int> labelVotes;
		};
		std::unordered_map<VoxelKey, Acc, VoxelKeyHash> voxels;
		voxels.reserve(n / 2 + 1);
		for (size_t p = 0; p < n; p++)
		{
			VoxelKey k{static_cast<int64_t>(std::floor(static_cast<double>(pts[p * 3 + 0]) / voxelSize)),
			           static_cast<int64_t>(std::floor(static_cast<double>(pts[p * 3 + 1]) / voxelSize)),
			           static_cast<int64_t>(std::floor(static_cast<double>(pts[p * 3 + 2]) / voxelSize))};
			Acc&     a = voxels[k];
			for (int d = 0; d < 3; d++)
			{
				a.sum[d] += pts[p * 3 + d];
			}
			if (!rgb.empty())
			{
				for (int d = 0; d < 3; d++)
				{
					a.fsumRgb[d] += rgb[p * 3 + d];
				}
			}
			if (!intensity.empty())
			{
				a.fsumI += intensity[p];
			}
			if (!labels.empty())
			{
				a.labelVotes[labels[p]]++;
			}
			a.count++;
		}
		const size_t m = voxels.size();
		pre.subPoints.resize(m * 3);
		if (!rgb.empty())
		{
			pre.subRgb.resize(m * 3);
		}
		if (!intensity.empty())
		{
			pre.subIntensity.resize(m);
		}
		if (!labels.empty())
		{
			pre.subLabels.resize(m);
		}
		size_t out = 0;
		for (auto& kv : voxels)
		{
			const Acc& a = kv.second;
			for (int d = 0; d < 3; d++)
			{
				pre.subPoints[out * 3 + d] = static_cast<float>(a.sum[d] / a.count);
			}
			if (!rgb.empty())
			{
				for (int d = 0; d < 3; d++)
				{
					pre.subRgb[out * 3 + d] = static_cast<float>(a.fsumRgb[d] / a.count);
				}
			}
			if (!intensity.empty())
			{
				pre.subIntensity[out] = static_cast<float>(a.fsumI / a.count);
			}
			if (!labels.empty())
			{
				int best  = 0;
				int bestc = -1;
				for (auto& v : a.labelVotes)
				{
					if (v.second > bestc)
					{
						bestc = v.second;
						best  = v.first;
					}
				}
				pre.subLabels[out] = best;
			}
			out++;
		}
		// projection map: original (possibly shifted) points -> nearest sub point
		pre.adaptor = std::make_unique<PointCloudAdaptor>(PointCloudAdaptor{&pre.subPoints});
		pre.tree    = std::make_unique<KDTree>(3, *pre.adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		for (size_t p = 0; p < n; p++)
		{
			pre.projInds[p] = knnQuery(*pre.tree, m, pts.data() + p * 3, 1)[0];
		}
	}
	else
	{
		pre.subPoints    = pts;
		pre.subRgb       = rgb;
		pre.subIntensity = intensity;
		pre.subLabels    = labels;
		for (size_t p = 0; p < n; p++)
		{
			pre.projInds[p] = static_cast<int64_t>(p);
		}
		pre.adaptor = std::make_unique<PointCloudAdaptor>(PointCloudAdaptor{&pre.subPoints});
		pre.tree    = std::make_unique<KDTree>(3, *pre.adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	}
	pre.M = pre.subPoints.size() / 3;
	return pre;
}

namespace
{
	//! kNN of every point of b against a tree over a
	std::vector<int64_t> knnTreeOnAQueryB(const std::vector<float>& a, const std::vector<float>& b, int k)
	{
		PointCloudAdaptor ad{&a};
		KDTree            tree(3, ad, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		return knnQueryBatch(tree, a.size() / 3, b, k);
	}
} // namespace

Hierarchy computeHierarchy(const std::vector<float>& points, const std::vector<int>& stageNpoints, int numNeighbors, std::mt19937_64& rng, const std::vector<int64_t>* fpsStarts)
{
	const size_t n = points.size() / 3;
	const int    K = numNeighbors;
	Hierarchy    h;
	// idx0 and idx1 are the same tree/query pair - compute once, use twice
	const auto idx01 = knnTreeOnAQueryB(points, points, K);
	h.idx.push_back(idx01); // idx0
	h.idx.push_back(idx01); // idx1

	std::vector<float> current      = points;
	size_t             nextFpsStart = 0; // recorded starts are consumed in FPS-invocation order
	for (size_t s = 0; s < stageNpoints.size(); s++)
	{
		size_t spN = std::min<size_t>(static_cast<size_t>(stageNpoints[s]), current.size() / 3);
		// fpsStarts (optional) pins the FPS start of each FPS INVOCATION in
		// order (a stage where every point is kept runs no FPS and consumes
		// no start - see test/TestSemSeg.cpp)
		int64_t start = -1;
		if (fpsStarts && nextFpsStart < fpsStarts->size() && spN < current.size() / 3)
		{
			start = (*fpsStarts)[nextFpsStart++];
		}
		auto               cents = fpsIndices(current, spN, rng, start);
		std::vector<float> next(cents.size() * 3);
		for (size_t i = 0; i < cents.size(); i++)
		{
			std::memcpy(next.data() + i * 3, current.data() + cents[i] * 3, 3 * sizeof(float));
		}
		h.sp.push_back(std::move(next));
		h.idx.push_back(knnTreeOnAQueryB(current, h.sp.back(), K));     // down: (S,K)
		h.idx.push_back(knnTreeOnAQueryB(h.sp.back(), h.sp.back(), K)); // self: (S,K)
		current = h.sp.back();
	}
	// upsample nearest (K=1): tree on the coarse set, query the finer set
	h.idx.push_back(knnTreeOnAQueryB(h.sp[3], h.sp[2], 1)); // idx10: (S2,1)
	h.idx.push_back(knnTreeOnAQueryB(h.sp[2], h.sp[1], 1)); // idx11: (S1,1)
	h.idx.push_back(knnTreeOnAQueryB(h.sp[1], h.sp[0], 1)); // idx12: (S0,1)
	h.idx.push_back(knnTreeOnAQueryB(h.sp[0], points, 1));  // idx13: (N,1)
	return h;
}
