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

// Model adapters: the FKAConv branch of buildPatch/assembleBatchBuffers and
// the FKAConv-specific IO checks of validateSpecAgainstIO moved here verbatim
// (the parity suite is the safety net: wrappers keep the old signatures).

#include "ModelAdapter.h"

#include "SphereFormerAdapter.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace
{
	//! "the model requires the '<what>' feature, but the selected cloud does
	//! not provide it" (shared by the adapters' cloud-feature checks)
	std::string missingFeature(const char* what)
	{
		return std::string("the model requires the '") + what + "' feature, but the selected cloud does not provide it";
	}

	//! Resolves the order of the extra feature channels (after xyz / per feature mode)
	/** The reference pipeline takes the extra features in the order given at
	    training time (infer_onnx.py -f ...), recorded in the manifest as
	    preprocessing.feature_columns. Default: intensity first, then rgb. **/
	std::vector<std::string> resolveFeatureColumns(const ModelSpec& spec)
	{
		std::vector<std::string> raw;
		if (!spec.featureColumns.empty())
		{
			raw = spec.featureColumns;
		}
		else
		{
			raw = {"intensity", "rgb"};
		}
		// "rgb" expands to 3 scalar channels (r, g, b)
		std::vector<std::string> expanded;
		for (const auto& col : raw)
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
		return expanded;
	}

	//! Value of one extra feature channel of a subsampled point
	float extraChannelValue(const CloudPre& pre, int64_t gi, const std::string& column)
	{
		if (column == "intensity" || column == "i")
		{
			if (pre.subIntensity.empty())
			{
				throw std::runtime_error("feature 'intensity' is required by the model but the cloud provides no scalar field for it");
			}
			return pre.subIntensity[gi];
		}
		if (column == "r" || column == "g" || column == "b")
		{
			if (pre.subRgb.empty())
			{
				throw std::runtime_error("feature 'rgb' is required by the model but the cloud has no colors");
			}
			const int c = (column == "r") ? 0 : (column == "g") ? 1
			                                                    : 2;
			return pre.subRgb[gi * 3 + c];
		}
		if (column == "rgb")
		{
			throw std::runtime_error("feature column 'rgb' is not a single scalar - it expands to 3 channels; use 'r', 'g', 'b' or the feature_mode 'rgb'/'i_rgb' paths instead");
		}
		throw std::runtime_error("unsupported feature column '" + column + "' (supported: intensity, rgb)");
	}

	//! The extra feature channels (after xyz), k rows of E columns, row-major
	std::vector<float> buildExtraChannels(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, int extraCount)
	{
		const std::vector<std::string> columns = resolveFeatureColumns(spec);
		const size_t                   k       = idxs.size();
		std::vector<float>             extras(k * static_cast<size_t>(extraCount));
		for (size_t i = 0; i < k; i++)
		{
			for (int e = 0; e < extraCount; e++)
			{
				extras[i * extraCount + e] = extraChannelValue(pre, idxs[i], columns[e]);
			}
		}
		return extras;
	}

	//! PointNet feature construction (moved VERBATIM from the former
	//! modelName-keyed branch of buildPatchFeatures): the training feature
	//! order (feature_columns) applied on top of xyz
	/** recenteredPts: k*3, the patch points as they go into the model (the
	    GenericAdapter mid-range-recenter them afterwards). **/
	std::vector<float> buildPointNetFeatures(const ModelSpec& spec, const CloudPre& pre, const std::vector<float>& recenteredPts, const std::vector<int64_t>& idxs)
	{
		const size_t       k = idxs.size();
		const int          C = spec.inChannels;
		std::vector<float> feats(k * static_cast<size_t>(C));

		// PointNet.transform: feature vector = xyz + extra channels, in the
		// training feature order (feature_columns); xyz recentered later.
		if (C < 3)
		{
			throw std::runtime_error("PointNet models require in_channels >= 3 (xyz)");
		}
		const int          extras     = C - 3;
		std::vector<float> extrasData = buildExtraChannels(spec, pre, idxs, extras);
		// color-like channels are divided by 255 when their max > 1 (the
		// reference's PCD reader delivers rgb in [0,1]; raw intensities keep
		// their scale unless the patch mixes both scales)
		float extraMax = 0.f;
		bool  any      = false;
		for (float v : extrasData)
		{
			any      = true;
			extraMax = std::max(extraMax, v);
		}
		const bool divideBy255 = any && extraMax > 1.0f;
		for (size_t i = 0; i < k; i++)
		{
			for (int c = 0; c < 3; c++)
			{
				feats[i * C + c] = recenteredPts[i * 3 + c];
			}
			for (int e = 0; e < extras; e++)
			{
				feats[i * C + 3 + e] = divideBy255 ? extrasData[i * extras + e] / 255.f : extrasData[i * extras + e];
			}
		}
		return feats;
	}

	//! Builds the k*C patch feature matrix (row-major) for the FKAConv family,
	//! replicating FKAConv._build_features (feature_mode-driven)
	/** recenteredPts: k*3, the patch points AFTER the eval augmentation (the
	    same coordinates that go into the 'points' input). **/
	std::vector<float> buildPatchFeatures(const ModelSpec& spec, const CloudPre& pre, const std::vector<float>& recenteredPts, const std::vector<int64_t>& idxs)
	{
		const size_t       k    = idxs.size();
		const int          C    = spec.inChannels;
		const QString&     fm   = spec.featureMode;
		const std::string  mode = fm.toStdString();
		std::vector<float> feats(k * static_cast<size_t>(C));

		if (mode == "constant")
		{
			// FKAConv._build_features 'constant': a single channel of ones
			if (C != 1)
			{
				throw std::runtime_error("feature_mode 'constant' requires in_channels=1");
			}
			std::fill(feats.begin(), feats.end(), 1.f);
			return feats;
		}
		if (mode == "xyz")
		{
			// features == the (recentered) patch points, exactly as the
			// reference: _build_features(sampled_pc) runs on the AUGMENTED
			// points, so both 'points' and 'features' inputs carry the same
			// recentered coordinates
			if (C != 3)
			{
				throw std::runtime_error("feature_mode 'xyz' requires in_channels=3");
			}
			std::memcpy(feats.data(), recenteredPts.data(), k * 3 * sizeof(float));
			return feats;
		}
		if (mode == "i" || mode == "intensity")
		{
			if (C != 1)
			{
				throw std::runtime_error("feature_mode 'i' requires in_channels=1");
			}
			for (size_t i = 0; i < k; i++)
			{
				feats[i] = extraChannelValue(pre, idxs[i], "intensity");
			}
			return feats;
		}
		if (mode == "rgb")
		{
			if (C != 3)
			{
				throw std::runtime_error("feature_mode 'rgb' requires in_channels=3");
			}
			if (pre.subRgb.empty())
			{
				throw std::runtime_error("feature_mode 'rgb' requires colors, but the cloud has none");
			}
			for (size_t i = 0; i < k; i++)
			{
				const int64_t gi = idxs[i];
				for (int c = 0; c < 3; c++)
				{
					feats[i * 3 + c] = pre.subRgb[gi * 3 + c];
				}
			}
			return feats;
		}
		if (mode == "i_rgb" || mode == "irgb" || mode == "intensity_rgb" || mode == "rgb_i")
		{
			// order: intensity first, then rgb (reference feature_mode 'i_rgb')
			if (C != 4)
			{
				throw std::runtime_error("feature_mode 'i_rgb' requires in_channels=4");
			}
			for (size_t i = 0; i < k; i++)
			{
				const int64_t gi = idxs[i];
				feats[i * 4 + 0] = extraChannelValue(pre, gi, "intensity");
				for (int c = 0; c < 3; c++)
				{
					feats[i * 4 + 1 + c] = pre.subRgb.empty() ? 0.f : pre.subRgb[gi * 3 + c];
				}
			}
			return feats;
		}
		if (mode == "feat")
		{
			// all extra channels, in the manifest feature_columns order
			const std::vector<std::string> columns = resolveFeatureColumns(spec);
			if (C != static_cast<int>(columns.size()))
			{
				throw std::runtime_error("feature_mode 'feat' requires in_channels=" + std::to_string(columns.size()) + " (feature_columns)");
			}
			for (size_t i = 0; i < k; i++)
			{
				for (size_t c = 0; c < columns.size(); c++)
				{
					feats[i * C + c] = extraChannelValue(pre, idxs[i], columns[c]);
				}
			}
			return feats;
		}
		if (mode == "feat_xyz")
		{
			// xyz (recentered!) first, then the extra channels
			const std::vector<std::string> columns = resolveFeatureColumns(spec);
			if (C != 3 + static_cast<int>(columns.size()))
			{
				throw std::runtime_error("feature_mode 'feat_xyz' requires in_channels=" + std::to_string(3 + columns.size()) + " (xyz + feature_columns)");
			}
			std::vector<float> extrasData = buildExtraChannels(spec, pre, idxs, static_cast<int>(columns.size()));
			for (size_t i = 0; i < k; i++)
			{
				for (int c = 0; c < 3; c++)
				{
					feats[i * C + c] = recenteredPts[i * 3 + c];
				}
				for (size_t e = 0; e < columns.size(); e++)
				{
					feats[i * C + 3 + e] = extrasData[i * columns.size() + e];
				}
			}
			return feats;
		}
		throw std::runtime_error("unsupported feature_mode '" + mode + "' (supported: constant, xyz, i, rgb, i_rgb, feat, feat_xyz)");
	}

	//! Batched tensor assembly shared by the adapters (verbatim from the
	//! former assembleBatchBuffers body, with the hierarchy inputs gated)
	/** allowHierarchyInputs=false rejects the FKAConv-specific support-point
	    (sp) and index (idx) inputs with an explicit error instead of silently
	    trying to feed them. **/
	BatchBuffers assembleBatchBuffersImpl(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage, bool allowHierarchyInputs)
	{
		BatchBuffers buffers;
		if (patches.empty())
		{
			if (errorMessage)
			{
				*errorMessage = "no patches to assemble";
			}
			return buffers;
		}

		const int64_t B = static_cast<int64_t>(patches.size());
		const int64_t k = static_cast<int64_t>(patches[0].idxs.size());

		buffers.shapes.assign(spec.inputNames.size(), {});
		buffers.f32.assign(spec.inputNames.size(), {});
		buffers.i64.assign(spec.inputNames.size(), {});

		auto fail = [&](const std::string& message)
		{
			if (errorMessage)
			{
				*errorMessage = message;
			}
			return BatchBuffers{};
		};

		for (size_t in = 0; in < spec.inputNames.size(); in++)
		{
			const std::string& name = spec.inputNames[in];
			buffers.shapes[in]      = OrtSession::resolveShape(spec.inputShapes[in], B, k);
			size_t perPatch         = 1;
			for (size_t d = 1; d < buffers.shapes[in].size(); d++)
			{
				perPatch *= static_cast<size_t>(buffers.shapes[in][d]);
			}

			if (spec.inputDtypes[in] == "int64")
			{
				if (!allowHierarchyInputs)
				{
					return fail("unexpected int64 input '" + name + "' for a generic single-tensor model");
				}
				buffers.i64[in].resize(static_cast<size_t>(B) * perPatch);
				for (int64_t b = 0; b < B; b++)
				{
					const Patch&                p   = patches[b];
					const std::vector<int64_t>* src = nullptr;
					if (name.rfind("idx", 0) == 0)
					{
						int idxNo = std::stoi(name.substr(3));
						if (idxNo < 0 || idxNo >= static_cast<int>(p.hier.idx.size()))
						{
							return fail("unknown index input: " + name);
						}
						src = &p.hier.idx[idxNo];
					}
					else
					{
						return fail("unknown int64 input: " + name);
					}
					if (src->size() != perPatch)
					{
						return fail("input " + name + " size mismatch: " + std::to_string(src->size()) + " vs " + std::to_string(perPatch));
					}
					std::memcpy(buffers.i64[in].data() + b * perPatch, src->data(), perPatch * sizeof(int64_t));
				}
			}
			else if (spec.inputDtypes[in] == "float32")
			{
				buffers.f32[in].resize(static_cast<size_t>(B) * perPatch);
				std::vector<float> spCHW; // scratch for sp transpose
				for (int64_t b = 0; b < B; b++)
				{
					const Patch&              p   = patches[b];
					const std::vector<float>* src = nullptr;
					if (name == "points")
					{
						src = &p.pointsCHW;
					}
					else if (name == "features")
					{
						src = &p.featuresCHW;
					}
					else if (name.rfind("sp", 0) == 0)
					{
						if (!allowHierarchyInputs)
						{
							return fail("unexpected support-point input '" + name + "' for a generic single-tensor model");
						}
						// sp is stored point-major (S*3); the ONNX input is channel-major (B, 3, S) — transpose per patch
						int spNo = std::stoi(name.substr(2));
						if (spNo < 0 || spNo >= static_cast<int>(p.hier.sp.size()))
						{
							return fail("unknown support-point input: " + name);
						}
						const std::vector<float>& sp = p.hier.sp[spNo];
						const size_t              S  = sp.size() / 3;
						spCHW.resize(3 * S);
						for (size_t i = 0; i < S; i++)
						{
							for (int d = 0; d < 3; d++)
							{
								spCHW[d * S + i] = sp[i * 3 + d];
							}
						}
						src = &spCHW;
					}
					else
					{
						return fail("unknown float input: " + name);
					}
					if (src->size() != perPatch)
					{
						return fail("input " + name + " size mismatch: " + std::to_string(src->size()) + " vs " + std::to_string(perPatch));
					}
					std::memcpy(buffers.f32[in].data() + b * perPatch, src->data(), perPatch * sizeof(float));
				}
			}
			else
			{
				return fail("input '" + name + "' has unsupported dtype '" + spec.inputDtypes[in] + "'");
			}
		}
		return buffers;
	}

	//! FKAConv: 4-stage hierarchy + 14 index inputs + 'points'/'features' tensors
	class FKAConvAdapter : public ModelAdapter
	{
	  public:
		const char* id() const override
		{
			return "FKAConv";
		}

		void applyDefaults(ModelSpec& spec) const override
		{
			// the reference FKAConv preprocessing does not shift the cloud
			// minimum (that is the PointNet-style rule)
			spec.shiftMin = false;
		}

		std::string validate(const ModelSpec& spec, const IOContract& io) const override
		{
			// 'points' and 'features' inputs are required
			const bool hasFeaturesInput = std::find(io.inputNames.begin(), io.inputNames.end(), std::string("features")) != io.inputNames.end();
			const bool hasPointsInput   = std::find(io.inputNames.begin(), io.inputNames.end(), std::string("points")) != io.inputNames.end();
			if (!hasFeaturesInput || !hasPointsInput)
			{
				return "FKAConv models require 'points' and 'features' inputs";
			}

			// the manifest must declare the 4 hierarchy stages
			if (spec.stageNpoints.size() != 4)
			{
				return "FKAConv models require 4 hierarchy stages (adapter.stage_npoints in the manifest)";
			}

			// and the graph must expose the 14 int64 hierarchy index inputs
			int idxCount = 0;
			for (const std::string& name : io.inputNames)
			{
				if (name.rfind("idx", 0) == 0)
				{
					const size_t dtypeIdx = static_cast<size_t>(&name - io.inputNames.data());
					if (io.inputDtypes[dtypeIdx] != "int64")
					{
						return "hierarchy input '" + name + "' must be int64";
					}
					++idxCount;
				}
			}
			if (idxCount != 14)
			{
				return "FKAConv models require 14 hierarchy index inputs (idx0..idx13), the graph declares " + std::to_string(idxCount);
			}
			return {};
		}

		std::string validateCloudFeatures(const ModelSpec& spec, bool hasColors, bool hasIntensity) const override
		{
			// the FKAConv feature_mode switch (moved verbatim from the former
			// modelName-dispatched free function)
			const QString mode = spec.featureMode;

			if (mode == "xyz" || mode == "constant")
			{
				return {};
			}
			if (mode == "i" || mode == "intensity")
			{
				return hasIntensity ? std::string() : missingFeature("intensity");
			}
			if (mode == "rgb")
			{
				return hasColors ? std::string() : missingFeature("rgb");
			}
			if (mode == "i_rgb" || mode == "irgb" || mode == "intensity_rgb" || mode == "rgb_i")
			{
				if (!hasIntensity)
				{
					return missingFeature("intensity");
				}
				return hasColors ? std::string() : missingFeature("rgb");
			}
			if (mode == "feat" || mode == "feat_xyz")
			{
				std::vector<std::string> columns = spec.featureColumns;
				if (columns.empty())
				{
					columns.push_back("intensity");
					columns.push_back("rgb");
				}
				// expand "rgb" to 3 channels (matches resolveFeatureColumns)
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
				columns          = expanded;
				const int needed = mode == "feat" ? static_cast<int>(columns.size()) : 3 + static_cast<int>(columns.size());
				if (needed != spec.inChannels)
				{
					return "feature_mode '" + mode.toStdString() + "' with " + std::to_string(columns.size()) + " feature column(s) needs in_channels=" + std::to_string(needed) + ", but the manifest declares " + std::to_string(spec.inChannels);
				}
				for (const std::string& column : columns)
				{
					if ((column == "r" || column == "g" || column == "b") && !hasColors)
					{
						return missingFeature("rgb");
					}
					if ((column == "intensity" || column == "i") && !hasIntensity)
					{
						return missingFeature("intensity");
					}
				}
				return {};
			}
			return "unsupported feature_mode '" + mode.toStdString() + "' (supported: constant, xyz, i, rgb, i_rgb, feat, feat_xyz)";
		}

		Patch buildPatch(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, std::mt19937_64& rng, const std::vector<int64_t>* fpsStarts, const float* fixedMean) const override
		{
			if (spec.stageNpoints.size() != 4)
			{
				throw std::runtime_error("FKAConv models require 4 hierarchy stages (adapter.stage_npoints in the manifest)");
			}

			Patch patch;
			patch.idxs     = idxs;
			const size_t k = idxs.size();

			// gather sampled points
			std::vector<float> pts(k * 3);
			for (size_t i = 0; i < k; i++)
			{
				std::memcpy(pts.data() + i * 3, pre.subPoints.data() + idxs[i] * 3, 3 * sizeof(float));
			}

			// eval augmentation, in the reference's order:
			// 1) recenter: subtract the per-dim MEAN of the patch points
			//    (the reference computes it in numpy float32; production uses a
			//    double-precision mean rounded to float32, the parity test can pin
			//    the reference's exact mean via fixedMean)
			// 2) normalize (manifest-driven): subtract the mean again, divide by the
			//    largest per-dim range (the reference's normalize 'linear' method)
			if (spec.recenter)
			{
				float mean[3] = {0.f, 0.f, 0.f};
				if (fixedMean)
				{
					std::memcpy(mean, fixedMean, 3 * sizeof(float));
				}
				else
				{
					double sum[3] = {0, 0, 0};
					for (size_t i = 0; i < k; i++)
					{
						for (int d = 0; d < 3; d++)
						{
							sum[d] += pts[i * 3 + d];
						}
					}
					for (int d = 0; d < 3; d++)
					{
						mean[d] = static_cast<float>(sum[d] / static_cast<double>(k));
					}
				}
				for (size_t i = 0; i < k; i++)
				{
					for (int d = 0; d < 3; d++)
					{
						pts[i * 3 + d] -= mean[d];
					}
				}
			}
			if (spec.normalizePoints)
			{
				double mean[3] = {0, 0, 0};
				double mn[3]   = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
				double mx[3]   = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
				for (size_t i = 0; i < k; i++)
				{
					for (int d = 0; d < 3; d++)
					{
						const double v = pts[i * 3 + d];
						mean[d] += v / static_cast<double>(k);
						mn[d] = std::min(mn[d], v);
						mx[d] = std::max(mx[d], v);
					}
				}
				double range = 0.0;
				for (int d = 0; d < 3; d++)
				{
					range = std::max(range, mx[d] - mn[d]);
				}
				if (range > 0.0)
				{
					for (size_t i = 0; i < k; i++)
					{
						for (int d = 0; d < 3; d++)
						{
							pts[i * 3 + d] = static_cast<float>((pts[i * 3 + d] - mean[d]) / range);
						}
					}
				}
			}

			patch.pointsCHW.resize(3 * k);
			for (size_t i = 0; i < k; i++)
			{
				for (int d = 0; d < 3; d++)
				{
					patch.pointsCHW[d * k + i] = pts[i * 3 + d];
				}
			}

			// the feature tensor is built from the SAME augmented points the
			// 'points' input carries (feature modes containing xyz must not fall
			// back to the absolute cloud coordinates)
			std::vector<float> feats = buildPatchFeatures(spec, pre, pts, idxs); // k*C row-major
			if (feats.size() != k * static_cast<size_t>(spec.inChannels))
			{
				throw std::runtime_error("feature count mismatch: expected " + std::to_string(k * spec.inChannels) + " values for in_channels=" + std::to_string(spec.inChannels));
			}
			patch.featuresCHW.resize(static_cast<size_t>(spec.inChannels) * k);
			for (size_t i = 0; i < k; i++)
			{
				for (int c = 0; c < spec.inChannels; c++)
				{
					patch.featuresCHW[c * k + i] = feats[i * spec.inChannels + c];
				}
			}

			patch.hier = computeHierarchy(pts, spec.stageNpoints, spec.numNeighbors, rng, fpsStarts);
			return patch;
		}

		BatchBuffers assembleBatch(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage) const override
		{
			return assembleBatchBuffersImpl(spec, patches, errorMessage, /*allowHierarchyInputs=*/true);
		}
	};

	//! Generic single-tensor models (PointNet-style): 'points' [+ 'features']
	//! float tensors only, no hierarchy inputs
	class GenericAdapter : public ModelAdapter
	{
	  public:
		const char* id() const override
		{
			return "PointNet";
		}

		void applyDefaults(ModelSpec& spec) const override
		{
			// the PointNet-style preprocessing subtracts the cloud minimum
			// before subsampling (reference shift_min)
			spec.shiftMin = true;
		}

		std::string validate(const ModelSpec& spec, const IOContract& io) const override
		{
			// the generic single-tensor contract: every input is a float32
			// 'points' or 'features' tensor (CHW), no hierarchy inputs. The
			// batch assembly would reject anything else mid-run; failing here
			// surfaces the problem before any inference.
			bool hasPointsInput = false;
			for (size_t in = 0; in < io.inputNames.size(); in++)
			{
				const std::string& name = io.inputNames[in];
				if (name != "points" && name != "features")
				{
					return "this model architecture only feeds 'points' and 'features' float tensors, but the graph declares input '" + name + "'";
				}
				if (io.inputDtypes[in] != "float32")
				{
					return "input '" + name + "' must be float32 (the graph declares " + io.inputDtypes[in] + ")";
				}
				hasPointsInput = hasPointsInput || name == "points";
			}
			if (!hasPointsInput)
			{
				return "this model architecture requires a 'points' input";
			}
			// and the output is [batch, num_points, num_classes]
			if (io.outputShapes.empty() || io.outputShapes[0].size() != 3)
			{
				return "this model architecture expects a [batch, num_points, num_classes] output tensor";
			}
			return {};
		}

		std::string validateCloudFeatures(const ModelSpec& spec, bool hasColors, bool hasIntensity) const override
		{
			// xyz + extra channels; every required extra must exist (moved
			// verbatim from the former modelName-keyed branch)
			const int extras = spec.inChannels - 3;
			if (extras < 0)
			{
				return "PointNet models require in_channels >= 3 (xyz)";
			}
			std::vector<std::string> columns = spec.featureColumns;
			if (columns.empty())
			{
				// reference default: intensity first, then rgb
				if (extras >= 1)
					columns.push_back("intensity");
				if (extras >= 2)
					columns.push_back("rgb");
			}
			if (static_cast<int>(columns.size()) < extras)
			{
				return "the model needs " + std::to_string(extras) + " extra feature channel(s) (after xyz) but only " + std::to_string(columns.size()) + " are declared";
			}
			for (int e = 0; e < extras; e++)
			{
				if (columns[e] == "rgb" && !hasColors)
				{
					return missingFeature("rgb");
				}
				if ((columns[e] == "intensity" || columns[e] == "i") && !hasIntensity)
				{
					return missingFeature("intensity");
				}
			}
			return {};
		}

		Patch buildPatch(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, std::mt19937_64& /*rng*/, const std::vector<int64_t>* /*fpsStarts*/, const float* /*fixedMean*/) const override
		{
			Patch patch;
			patch.idxs     = idxs;
			const size_t k = idxs.size();

			// gather sampled points
			std::vector<float> pts(k * 3);
			for (size_t i = 0; i < k; i++)
			{
				std::memcpy(pts.data() + i * 3, pre.subPoints.data() + idxs[i] * 3, 3 * sizeof(float));
			}

			// PointNet: feature vector = xyz + extra channels, recentered to the
			// mid-range origin
			std::vector<float> feats = buildPointNetFeatures(spec, pre, pts, idxs); // k*C
			if (feats.size() != k * static_cast<size_t>(spec.inChannels))
			{
				throw std::runtime_error("feature count mismatch: expected " + std::to_string(k * spec.inChannels) + " values for in_channels=" + std::to_string(spec.inChannels));
			}
			float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
			float mx[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
			for (size_t i = 0; i < k; i++)
			{
				for (int d = 0; d < 3; d++)
				{
					mn[d] = std::min(mn[d], feats[i * spec.inChannels + d]);
					mx[d] = std::max(mx[d], feats[i * spec.inChannels + d]);
				}
			}
			for (size_t i = 0; i < k; i++)
			{
				for (int d = 0; d < 3; d++)
				{
					feats[i * spec.inChannels + d] -= (mn[d] + mx[d]) / 2.f;
				}
			}
			patch.pointsCHW.resize(static_cast<size_t>(spec.inChannels) * k);
			for (size_t i = 0; i < k; i++)
			{
				for (int c = 0; c < spec.inChannels; c++)
				{
					patch.pointsCHW[c * k + i] = feats[i * spec.inChannels + c];
				}
			}
			return patch;
		}

		BatchBuffers assembleBatch(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage) const override
		{
			return assembleBatchBuffersImpl(spec, patches, errorMessage, /*allowHierarchyInputs=*/false);
		}
	};
} // namespace

//! The adapter registry, populated on first use (function-local static:
//! construction is thread-safe and ordered before any use)
const std::unordered_map<std::string, std::unique_ptr<ModelAdapter>>& registry()
{
	static const std::unordered_map<std::string, std::unique_ptr<ModelAdapter>> r = []
	{
		std::unordered_map<std::string, std::unique_ptr<ModelAdapter>> map;
		map.emplace("FKAConv", std::make_unique<FKAConvAdapter>());
		map.emplace("PointNet", std::make_unique<GenericAdapter>());
		// SphereFormerSemSeg exports are keyed under both the adapter id and
		// the manifest model_name (adapters are stateless, two instances)
		map.emplace("SphereFormer", std::make_unique<SphereFormerAdapter>());
		map.emplace("SphereFormerSemSeg", std::make_unique<SphereFormerAdapter>());
		return map;
	}();
	return r;
}

const ModelAdapter* findModelAdapter(const QString& modelName)
{
	const auto& map = registry();
	const auto  it  = map.find(modelName.toStdString());
	return it == map.end() ? nullptr : it->second.get();
}

const ModelAdapter* modelAdapterForSpec(const ModelSpec& spec)
{
	// the manifest's adapter.id pins the family even when model_name is a
	// descriptive name with no registered adapter of its own
	if (!spec.adapterId.isEmpty())
	{
		return findModelAdapter(spec.adapterId);
	}
	return findModelAdapter(spec.modelName);
}

std::vector<std::string> modelAdapterNames()
{
	std::vector<std::string> names;
	for (const auto& entry : registry())
	{
		names.push_back(entry.first);
	}
	return names;
}

std::string unknownAdapterError(const ModelSpec& spec)
{
	// when adapter.id is set but unknown, THAT (not model_name) is the
	// failing key
	const bool     adapterIdUnknown = !spec.adapterId.isEmpty() && findModelAdapter(spec.adapterId) == nullptr;
	const QString& failingKey       = adapterIdUnknown ? spec.adapterId : spec.modelName;
	const char*    what             = adapterIdUnknown ? "adapter.id" : "model_name";
	std::string    supported;
	for (const std::string& name : modelAdapterNames())
	{
		supported += (supported.empty() ? "" : ", ") + name;
	}
	return std::string("unknown ") + what + " '" + failingKey.toStdString() + "' (supported: " + supported
	       + "); set it in the manifest to a supported model architecture";
}

int fixedBatchSize(const ModelSpec& spec)
{
	const ModelAdapter* adapter = modelAdapterForSpec(spec);
	return adapter ? adapter->fixedBatchSize(spec) : 0;
}
