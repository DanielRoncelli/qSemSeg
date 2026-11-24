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

#include "SlidingWindow.h"

#include "ModelAdapter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

std::vector<int64_t> samplePatch(const CloudPre& pre, int numPoints, float tolerance, const SWParams& params, const std::vector<float>& possibilities, std::mt19937_64& rng, float* centerOut)
{
	const size_t m = pre.M;
	const int    k = std::max(1, static_cast<int>(numPoints * tolerance));

	int64_t centerIdx;
	if (params.sampler == 0)
	{
		centerIdx = static_cast<int64_t>(std::min_element(possibilities.begin(), possibilities.end()) - possibilities.begin());
	}
	else
	{
		std::uniform_real_distribution<float> u(0.f, 1.f);
		if (!pre.subLabels.empty() && u(rng) < 0.5f)
		{
			std::vector<int64_t> minority;
			for (size_t i = 0; i < m; i++)
			{
				if (pre.subLabels[i] == 1)
				{
					minority.push_back(static_cast<int64_t>(i));
				}
			}
			if (!minority.empty())
			{
				std::uniform_int_distribution<size_t> d(0, minority.size() - 1);
				centerIdx = minority[d(rng)];
			}
			else
			{
				std::uniform_int_distribution<size_t> d(0, m - 1);
				centerIdx = static_cast<int64_t>(d(rng));
			}
		}
		else
		{
			std::uniform_int_distribution<size_t> d(0, m - 1);
			centerIdx = static_cast<int64_t>(d(rng));
		}
	}

	std::vector<float> centerPt(pre.subPoints.begin() + centerIdx * 3, pre.subPoints.begin() + centerIdx * 3 + 3);
	if (centerOut)
	{
		std::memcpy(centerOut, centerPt.data(), 3 * sizeof(float));
	}
	std::vector<int64_t> idxs;
	if (m < static_cast<size_t>(k))
	{
		idxs.resize(k);
		std::uniform_int_distribution<size_t> d(0, m - 1);
		for (size_t i = 0; i < m; i++)
		{
			idxs[i] = static_cast<int64_t>(i);
		}
		for (size_t i = m; i < static_cast<size_t>(k); i++)
		{
			idxs[i] = static_cast<int64_t>(d(rng));
		}
	}
	else
	{
		idxs = knnQuery(*pre.tree, m, centerPt.data(), k);
	}
	std::shuffle(idxs.begin(), idxs.end(), rng);
	return idxs;
}

Patch buildPatch(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, std::mt19937_64& rng, const std::vector<int64_t>* fpsStarts, const float* fixedMean)
{
	// dispatch to the model family's adapter (see ModelAdapter.h); the
	// sliding-window math around it stays model-independent. An unknown
	// model family is a loud error here too - validateSpecAgainstIO normally
	// catches it first.
	const ModelAdapter* adapter = modelAdapterForSpec(spec);
	if (!adapter)
	{
		throw std::runtime_error(unknownAdapterError(spec));
	}
	return adapter->buildPatch(spec, pre, idxs, rng, fpsStarts, fixedMean);
}

BatchBuffers assembleBatchBuffers(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage)
{
	const ModelAdapter* adapter = modelAdapterForSpec(spec);
	if (!adapter)
	{
		if (errorMessage)
		{
			*errorMessage = unknownAdapterError(spec);
		}
		return BatchBuffers{};
	}
	return adapter->assembleBatch(spec, patches, errorMessage);
}

std::vector<OrtPtr<OrtValue>> buildBatchInputs(const ModelSpec& spec, const OrtSession& session, const std::vector<Patch>& patches, std::string* errorMessage, std::vector<std::vector<float>>* outF32, std::vector<std::vector<int64_t>>* outI64, std::vector<std::vector<int32_t>>* outI32)
{
	// per-input data buffers, batch-concatenated. Stored in the caller-provided
	// containers (or local fallbacks) so they outlive the session run.
	std::vector<std::vector<float>>    localF32;
	std::vector<std::vector<int64_t>>  localI64;
	std::vector<std::vector<int32_t>>  localI32;
	std::vector<std::vector<float>>&   f32Keep = outF32 ? *outF32 : localF32;
	std::vector<std::vector<int64_t>>& i64Keep = outI64 ? *outI64 : localI64;
	std::vector<std::vector<int32_t>>& i32Keep = outI32 ? *outI32 : localI32;

	std::string  error;
	BatchBuffers buffers = assembleBatchBuffers(spec, patches, &error);
	if (buffers.shapes.empty())
	{
		if (errorMessage && !error.empty())
		{
			*errorMessage = error;
		}
		return {};
	}
	// keep the buffers alive across the session run (the tensors borrow them)
	f32Keep = std::move(buffers.f32);
	i64Keep = std::move(buffers.i64);
	i32Keep = std::move(buffers.i32);

	std::vector<OrtPtr<OrtValue>> inputs;
	for (size_t in = 0; in < spec.inputNames.size(); in++)
	{
		if (spec.inputDtypes[in] == "int64")
		{
			inputs.push_back(session.makeTensorI64(buffers.shapes[in], i64Keep[in], errorMessage));
		}
		else if (spec.inputDtypes[in] == "int32")
		{
			inputs.push_back(session.makeTensorI32(buffers.shapes[in], i32Keep[in], errorMessage));
		}
		else
		{
			inputs.push_back(session.makeTensorF32(buffers.shapes[in], f32Keep[in], errorMessage));
		}
		if (!inputs.back())
		{
			return {};
		}
	}
	return inputs;
}

int effectiveNumPoints(const ModelSpec& spec)
{
	return spec.staticNumPoints > 0 ? spec.staticNumPoints : spec.numPoints;
}

void accumulatePatchLogits(const float* logits, size_t rowCount, const std::vector<int64_t>& idxs, int numClasses, double smooth, std::vector<float>& testProbs, bool inputIsLogits, size_t rowsValid)
{
	if (numClasses <= 0 || idxs.size() != rowCount)
	{
		throw std::runtime_error("accumulatePatchLogits: logits rows (" + std::to_string(rowCount) + ") do not match the patch indices (" + std::to_string(idxs.size()) + ")");
	}
	// rowsValid = 0 means "all rows are real"; rows beyond rowsValid are
	// padding (e.g. a static num_points model fed its fixed row count) and
	// must never touch testProbs - their idxs may repeat or be garbage
	const size_t validRows  = (rowsValid == 0 || rowsValid > rowCount) ? rowCount : rowsValid;
	const double complement = 1.0 - smooth;

	// float64 softmax of every row (the reference softmaxes the whole patch in
	// float64), rounded to float32 exactly like the reference does. When the
	// model already outputs probabilities (output.kind = "probabilities") the
	// rows are used as-is instead - a second softmax would corrupt them.
	std::vector<float> probs(rowCount * static_cast<size_t>(numClasses));
	if (inputIsLogits)
	{
		for (size_t r = 0; r < validRows; r++)
		{
			const float* row = logits + r * numClasses;
			double       mx  = row[0];
			for (int c = 1; c < numClasses; c++)
			{
				mx = std::max(mx, static_cast<double>(row[c]));
			}
			double sum = 0.0;
			for (int c = 0; c < numClasses; c++)
			{
				sum += std::exp(static_cast<double>(row[c]) - mx);
			}
			for (int c = 0; c < numClasses; c++)
			{
				probs[r * numClasses + c] = static_cast<float>(std::exp(static_cast<double>(row[c]) - mx) / sum);
			}
		}
	}
	else
	{
		std::memcpy(probs.data(), logits, probs.size() * sizeof(float));
	}

	// testProbs[idxs] = smooth * testProbs[idxs] + (1 - smooth) * probs, in
	// float64 like the reference, stored back as float32. Duplicated indices
	// follow the numpy scatter order (later rows overwrite earlier ones).
	// Only the valid (non-padding) rows participate.
	for (size_t r = 0; r < validRows; r++)
	{
		const int64_t gi = idxs[r];
		if (gi < 0 || static_cast<size_t>(gi) * numClasses + numClasses > testProbs.size())
		{
			throw std::runtime_error("accumulatePatchLogits: patch index out of range");
		}
		for (int c = 0; c < numClasses; c++)
		{
			const double previous          = testProbs[gi * numClasses + c];
			testProbs[gi * numClasses + c] = static_cast<float>(smooth * previous + complement * probs[r * numClasses + c]);
		}
	}
}

std::vector<int> argmaxLabels(const std::vector<float>& testProbs, int numClasses)
{
	const size_t     m = numClasses > 0 ? testProbs.size() / numClasses : 0;
	std::vector<int> labels(m);
	for (size_t i = 0; i < m; i++)
	{
		int best = 0;
		for (int c = 1; c < numClasses; c++)
		{
			if (testProbs[i * numClasses + c] > testProbs[i * numClasses + best])
			{
				best = c;
			}
		}
		labels[i] = best;
	}
	return labels;
}

std::vector<float> maxProbabilities(const std::vector<float>& testProbs, int numClasses)
{
	const size_t       m = numClasses > 0 ? testProbs.size() / numClasses : 0;
	std::vector<float> maxProbs(m, 0.0f);
	for (size_t i = 0; i < m; i++)
	{
		float best = testProbs[i * numClasses];
		for (int c = 1; c < numClasses; c++)
		{
			best = std::max(best, testProbs[i * numClasses + c]);
		}
		maxProbs[i] = best;
	}
	return maxProbs;
}

double testProbsMemoryBytes(size_t m, int numClasses)
{
	return static_cast<double>(m) * static_cast<double>(numClasses) * static_cast<double>(sizeof(float));
}

bool checkTestProbsMemory(size_t m, int numClasses, double maxBytes, std::string* errorMessage)
{
	const double bytes = testProbsMemoryBytes(m, numClasses);
	if (bytes <= maxBytes)
	{
		return true;
	}
	if (errorMessage)
	{
		char buf[384];
		std::snprintf(buf,
		              sizeof(buf),
		              "the per-point probability buffer would need %.2f GB (%.1f million subsampled points x %d classes), above the %.1f GB limit. Raise the voxel size to subsample fewer points",
		              bytes / 1024.0 / 1024.0 / 1024.0,
		              static_cast<double>(m) / 1.0e6,
		              numClasses,
		              maxBytes / 1024.0 / 1024.0 / 1024.0);
		*errorMessage = buf;
	}
	return false;
}

bool allFinite(const float* data, size_t count)
{
	for (size_t i = 0; i < count; i++)
	{
		if (!std::isfinite(data[i]))
		{
			return false;
		}
	}
	return true;
}

namespace
{
	std::vector<float> possibilityDelta(const CloudPre& pre, const std::vector<int64_t>& idxs, const float* center)
	{
		std::vector<float> dists(idxs.size());
		float              mx = 0.f;
		for (size_t i = 0; i < idxs.size(); i++)
		{
			const float* p  = pre.subPoints.data() + idxs[i] * 3;
			float        dx = p[0] - center[0];
			float        dy = p[1] - center[1];
			float        dz = p[2] - center[2];
			dists[i]        = dx * dx + dy * dy + dz * dz;
			mx              = std::max(mx, dists[i]);
		}
		// (1 - dist/max_dist)^2; all ones when the patch is degenerate
		std::vector<float> delta(idxs.size(), 1.f);
		if (mx > 0.f)
		{
			for (size_t i = 0; i < idxs.size(); i++)
			{
				delta[i] = (1.f - dists[i] / mx) * (1.f - dists[i] / mx);
			}
		}
		return delta;
	}
} // namespace

void addPossibilityDelta(std::vector<float>& possibilities, const std::vector<int64_t>& idxs, const std::vector<float>& delta)
{
	std::vector<float> updated(idxs.size());
	for (size_t i = 0; i < idxs.size(); i++)
	{
		const int64_t gi = idxs[i];
		if (gi < 0 || static_cast<size_t>(gi) >= possibilities.size())
		{
			throw std::runtime_error("possibility update: patch index out of range");
		}
		updated[i] = possibilities[gi] + delta[i];
	}
	for (size_t i = 0; i < idxs.size(); i++)
	{
		possibilities[idxs[i]] = updated[i];
	}
}

SegmentationResult runSlidingWindow(OrtSession& session, const ModelSpec& spec, CloudPre& pre, const SWParams& params, std::mt19937_64& rng, const std::function<bool(int, float)>& progressCallback, bool* canceled)
{
	if (canceled)
	{
		*canceled = false;
	}

	const size_t m = pre.M;
	const int    C = spec.numClasses;
	if (m == 0 || C <= 0)
	{
		return {};
	}
	// pre-flight memory guard: testProbs is [m x C] floats and can dwarf every
	// other allocation when the subsampled cloud is huge
	std::string memoryError;
	if (!checkTestProbsMemory(m, C, 2.0 * 1024.0 * 1024.0 * 1024.0, &memoryError))
	{
		throw std::runtime_error(memoryError);
	}

	std::vector<float> possibilities(m);
	{
		std::uniform_real_distribution<float> u(0.f, 1.f);
		for (auto& v : possibilities)
		{
			v = u(rng) * 1e-3f;
		}
	}
	std::vector<float> testProbs(m * static_cast<size_t>(C), 0.f);

	// a STATIC num_points axis (recorded by validateSpecAgainstIO) imposes the
	// patch size: the graph rejects anything but exactly that many points per
	// patch, so the requested patch size and any tolerance < 1 are ignored
	const int   patchPoints    = effectiveNumPoints(spec);
	const float patchTolerance = spec.staticNumPoints > 0 ? 1.0f : params.tolerance;
	// a FIXED batch axis (e.g. SphereFormer exports bake B=1) cannot stack
	// patches: clamp the effective patch batch (the calibration is skipped
	// separately inside autoSelectBatchSize)
	const int fixedBatch = fixedBatchSize(spec);
	const int patchBatch = fixedBatch > 0 ? std::min(std::max(params.patchBatch, 1), fixedBatch) : std::max(params.patchBatch, 1);

	int                it = 0;
	std::vector<Patch> batch;
	while (*std::min_element(possibilities.begin(), possibilities.end()) <= 0.5f && it < params.maxIters)
	{
		float center[3];
		auto  idxs = samplePatch(pre, patchPoints, patchTolerance, params, possibilities, rng, center);
		if (idxs.size() < 2)
		{
			// degenerate patch: bump center, retry
			int64_t ci = static_cast<int64_t>(std::min_element(possibilities.begin(), possibilities.end()) - possibilities.begin());
			possibilities[ci] += 0.001f;
			continue;
		}
		Patch patch = buildPatch(spec, pre, idxs, rng);
		auto  delta = possibilityDelta(pre, patch.idxs, center);
		addPossibilityDelta(possibilities, patch.idxs, delta);
		batch.push_back(std::move(patch));

		bool done = *std::min_element(possibilities.begin(), possibilities.end()) > 0.5f;
		if (static_cast<int>(batch.size()) >= patchBatch || done)
		{
			std::string                       error;
			std::vector<std::vector<float>>   batchF32;
			std::vector<std::vector<int64_t>> batchI64;
			std::vector<std::vector<int32_t>> batchI32;
			auto                              inputs = buildBatchInputs(spec, session, batch, &error, &batchF32, &batchI64, &batchI32);
			if (inputs.empty())
			{
				throw std::runtime_error("could not build ONNX inputs: " + error);
			}
			auto outs = session.run(inputs, &error);
			// batchF32/batchI64/batchI32 kept alive until here (end of scope) — good
			(void)batchF32;
			(void)batchI64;
			(void)batchI32;
			if (outs.empty())
			{
				throw std::runtime_error("ONNX inference failed: " + error);
			}

			size_t       logitCount = 0;
			const float* logits     = OrtSession::tensorDataF32(session.api(), outs[0].get(), &logitCount);
			if (!logits)
			{
				throw std::runtime_error("ONNX output is not a float32 tensor");
			}
			if (logitCount != batch.size() * batch[0].idxs.size() * static_cast<size_t>(C))
			{
				throw std::runtime_error("ONNX output size " + std::to_string(logitCount) + " does not match [batch, num_points, num_classes] = "
				                         + std::to_string(batch.size()) + "x" + std::to_string(batch[0].idxs.size()) + "x" + std::to_string(C));
			}
			// NaN/Inf outputs mean a broken model or execution provider: fail
			// loudly instead of silently producing garbage labels
			if (!allFinite(logits, logitCount))
			{
				throw NonFiniteOutputError("ONNX inference produced NaN/Inf output values (batch of " + std::to_string(batch.size())
				                           + " patch(es)) - the model or the execution provider is malfunctioning");
			}
			// output.kind = "probabilities" models emit normalized rows: they
			// must not be softmaxed a second time (see accumulatePatchLogits)
			const bool inputIsLogits = spec.outputKind != ModelSpec::OutputKind::Probabilities;
			for (size_t b = 0; b < batch.size(); b++)
			{
				accumulatePatchLogits(logits + b * batch[b].idxs.size() * C, batch[b].idxs.size(), batch[b].idxs, C, params.smooth, testProbs, inputIsLogits);
			}
			it += static_cast<int>(batch.size());
			batch.clear();

			// progress metric: the covered fraction (points above the 0.5
			// threshold), like the reference pipeline's display. The MINIMUM
			// possibility is useless as a bar: each patch jumps its center to
			// ~1 but the next least-covered point has received no delta yet,
			// so the minimum stays at the initial jitter for most of the walk
			// and would keep a progress bar at 0% until the very end
			size_t covered = 0;
			for (float v : possibilities)
			{
				if (v > 0.5f)
				{
					covered++;
				}
			}
			if (progressCallback && !progressCallback(it, static_cast<float>(covered) / static_cast<float>(possibilities.size())))
			{
				if (canceled)
				{
					*canceled = true;
				}
				return {};
			}
		}
	}

	// final labels + argmax confidence on the subsampled cloud, projected to
	// the original cloud
	std::vector<int>   labels   = argmaxLabels(testProbs, C);
	std::vector<float> maxProbs = maxProbabilities(testProbs, C);
	SegmentationResult result;
	result.labels.resize(pre.projInds.size());
	result.maxProbs.resize(pre.projInds.size());
	for (size_t i = 0; i < pre.projInds.size(); i++)
	{
		const int64_t proj = pre.projInds[i];
		if (proj < 0 || static_cast<size_t>(proj) >= labels.size())
		{
			throw std::runtime_error("projection index out of range");
		}
		result.labels[i]   = labels[proj];
		result.maxProbs[i] = maxProbs[proj];
	}
	return result;
}

namespace
{
	//! Zero-filled inputs built straight from the session IO contract
	/** Used by the warm-up probe (batch 1) and the batch-size calibration.
	    The backing buffers are written into outF32/outI64 and MUST outlive the
	    session run (CreateTensorWithDataAsOrtValue borrows the data pointer).
	    Returns empty inputs on failure, with *errorMessage set. **/
	std::vector<OrtPtr<OrtValue>> buildZeroInputs(const OrtSession& session, const ModelSpec& spec, int64_t batch, std::vector<std::vector<float>>* outF32, std::vector<std::vector<int64_t>>* outI64, std::string* errorMessage, std::vector<std::vector<int32_t>>* outI32 = nullptr)
	{
		const int64_t numPoints = spec.numPoints > 0 ? static_cast<int64_t>(spec.numPoints) : 4096;

		std::vector<OrtPtr<OrtValue>> inputs;
		inputs.reserve(spec.inputNames.size());

		for (size_t in = 0; in < spec.inputNames.size(); in++)
		{
			const std::vector<int64_t> shape = OrtSession::resolveShape(spec.inputShapes[in], batch, numPoints);
			if (shape.empty())
			{
				if (errorMessage)
				{
					*errorMessage = "probe: empty shape for input '" + spec.inputNames[in] + "'";
				}
				return {};
			}
			size_t elementCount = 1;
			for (int64_t d : shape)
			{
				elementCount *= static_cast<size_t>(d);
			}

			OrtPtr<OrtValue>  tensor;
			const std::string dtype = in < spec.inputDtypes.size() ? spec.inputDtypes[in] : "float32";
			if (dtype == "int64")
			{
				outI64->emplace_back(elementCount, 0);
				tensor = session.makeTensorI64(shape, outI64->back(), errorMessage);
			}
			else if (dtype == "int32")
			{
				if (!outI32)
				{
					if (errorMessage)
					{
						*errorMessage = "probe: int32 input '" + spec.inputNames[in] + "' but no int32 buffer sink";
					}
					return {};
				}
				outI32->emplace_back(elementCount, 0);
				tensor = session.makeTensorI32(shape, outI32->back(), errorMessage);
			}
			else
			{
				outF32->emplace_back(elementCount, 0.f);
				tensor = session.makeTensorF32(shape, outF32->back(), errorMessage);
			}
			if (!tensor)
			{
				if (errorMessage && errorMessage->empty())
				{
					*errorMessage = "probe: could not create the input tensor for '" + spec.inputNames[in] + "'";
				}
				return {};
			}
			inputs.push_back(std::move(tensor));
		}
		return inputs;
	}
} // namespace

bool probeSession(const OrtSession& session, const ModelSpec& spec, std::string* errorMessage)
{
	std::vector<std::vector<float>>   f32Buffers;
	std::vector<std::vector<int64_t>> i64Buffers;
	std::vector<std::vector<int32_t>> i32Buffers;
	auto                              inputs = buildZeroInputs(session, spec, 1, &f32Buffers, &i64Buffers, errorMessage, &i32Buffers);
	if (inputs.empty())
	{
		if (errorMessage && errorMessage->empty())
		{
			*errorMessage = "warm-up probe: the session exposes no inputs";
		}
		return false;
	}

	std::string                   error;
	std::vector<OrtPtr<OrtValue>> outputs = session.run(inputs, &error);
	if (outputs.empty())
	{
		if (errorMessage)
		{
			*errorMessage = "warm-up inference failed: " + error;
		}
		return false;
	}
	// outputs are discarded: the probe only verifies that the graph runs
	return true;
}

//! Readable note for batch-size calibration candidates that could not run
std::string calibrationFailureNote(const std::vector<int>& failedSizes, int chosenBatch, const std::string& rawError)
{
	if (failedSizes.empty())
	{
		return {};
	}
	std::string sizes;
	for (size_t i = 0; i < failedSizes.size(); ++i)
	{
		sizes += (i > 0 ? ", " : "") + std::to_string(failedSizes[i]);
	}
	return "patch batch size(s) " + sizes + " could not run during calibration"
	                                        " (the [E:onnxruntime] log line above is from the failed calibration"
	                                        " attempt); continuing with batch size "
	       + std::to_string(chosenBatch)
	       + " instead. Original error: " + rawError;
}

int autoSelectBatchSize(const OrtSession& session, const ModelSpec& spec, std::string* errorMessage)
{
	// A FIXED batch axis (e.g. SphereFormer exports bake B=1) leaves nothing
	// to calibrate: the run loop clamps the batch to the fixed size anyway.
	if (fixedBatchSize(spec) > 0)
	{
		return 1;
	}

	// Calibrate on zero-filled inputs of the session's real shapes: time
	// forward passes for increasing batch sizes and keep the size with the
	// best per-patch time. The batch size only groups forward passes into one
	// session.run - the patch sequence and the labels are unaffected (the
	// coverage updates are applied before the forward and never depend on its
	// output), so this is safe to auto-tune.
	const bool             isCuda     = session.activeProvider().find("CUDA") != std::string::npos;
	const std::vector<int> candidates = isCuda ? std::vector<int>{1, 4, 8, 16, 32} : std::vector<int>{1, 2, 4, 8};

	int              best         = 1;
	double           bestPerPatch = std::numeric_limits<double>::infinity();
	std::vector<int> failedSizes;
	std::string      rawFailure;
	for (int b : candidates)
	{
		std::vector<std::vector<float>>   f32Buffers;
		std::vector<std::vector<int64_t>> i64Buffers;
		std::vector<std::vector<int32_t>> i32Buffers;
		auto                              inputs = buildZeroInputs(session, spec, b, &f32Buffers, &i64Buffers, errorMessage, &i32Buffers);
		if (inputs.empty())
		{
			break; // keep the best batch size found so far
		}

		// one untimed run (allocations, EP plan setup), then timed runs
		const int runs   = isCuda ? 3 : 2;
		double    bestMs = std::numeric_limits<double>::infinity();
		bool      failed = false;
		for (int r = 0; r < runs; r++)
		{
			const auto t0   = std::chrono::steady_clock::now();
			auto       outs = session.run(inputs, errorMessage);
			const auto t1   = std::chrono::steady_clock::now();
			if (outs.empty())
			{
				failed = true;
				break;
			}
			const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
			bestMs          = std::min(bestMs, ms);
		}
		if (failed)
		{
			// this batch size does not work (e.g. memory): remember it and keep
			// the best so far - the plugin reports the failure as a note
			rawFailure = errorMessage ? *errorMessage : "";
			failedSizes.push_back(b);
			break;
		}

		const double perPatch = bestMs / static_cast<double>(b);
		// grow the batch only when it is clearly better (>= 10%)
		if (perPatch < bestPerPatch * 0.9)
		{
			bestPerPatch = perPatch;
			best         = b;
		}
	}
	if (errorMessage && !failedSizes.empty())
	{
		*errorMessage = calibrationFailureNote(failedSizes, best, rawFailure);
	}
	return best;
}
