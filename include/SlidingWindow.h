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
#include "OnnxRuntime.h"
#include "Preprocess.h"

#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

//! Sliding-window inference parameters
struct SWParams
{
	int      numPoints  = 25536;
	float    tolerance  = 1.0f;  // regular sampler: full numPoints (1.0)
	int      sampler    = 0;     // 0 = regular (argmin possibilities), 1 = class-balanced
	float    smooth     = 0.95f; // test_probs accumulation: 0.95*old + 0.05*softmax(logits)
	int      maxIters   = 100000;
	int      patchBatch = 1; // patches per ONNX forward pass (dynamic batch axis)
	uint64_t seed       = 42;
};

//! One sampled patch ready to be turned into ONNX inputs
struct Patch
{
	std::vector<int64_t> idxs;        // indices into the subsampled cloud
	std::vector<float>   pointsCHW;   // 3*k (FKAConv) or C*k (PointNet), channels-first
	std::vector<float>   featuresCHW; // C*k (FKAConv only)
	Hierarchy            hier;        // FKAConv only
	//! Adapter-owned raw tensor payloads, PARALLEL to spec.inputNames (empty
	/** entry = the input belongs to another family). Used by adapters whose
	    per-patch inputs are neither 'points'/'features' nor hierarchy tables
	    (e.g. SphereFormer's 66 int32 index tables + feat). **/
	std::vector<std::vector<float>>   adapterF32;
	std::vector<std::vector<int32_t>> adapterI32;
};

//! Samples one patch (regular sampler: center = argmin(possibilities); k-NN patch, shuffled)
/** centerOut (3 floats, optional) receives the patch center point. **/
std::vector<int64_t> samplePatch(const CloudPre& pre, int numPoints, float tolerance, const SWParams& params, const std::vector<float>& possibilities, std::mt19937_64& rng, float* centerOut = nullptr);

//! Builds the per-patch model inputs (eval augmentation + features + hierarchy)
/** fpsStarts (optional) pins the FPS start index of each hierarchy stage, and
    fixedMean (optional, 3 floats) pins the recenter mean - both for exact
    parity with the Python reference (see test/TestSemSeg.cpp). In production
    both are null and everything is computed. **/
Patch buildPatch(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, std::mt19937_64& rng, const std::vector<int64_t>* fpsStarts = nullptr, const float* fixedMean = nullptr);

//! Per-input data buffers for a batch of patches, in ONNX input order (pure data assembly)
/** shape[b] holds the resolved per-input tensor shape; the float/int64 buffers
    hold the batch-concatenated tensor payload. Kept separate from OrtValue
    creation so it is testable without a live ONNX Runtime session. **/
struct BatchBuffers
{
	std::vector<std::vector<int64_t>> shapes;
	std::vector<std::vector<float>>   f32;
	std::vector<std::vector<int64_t>> i64;
	std::vector<std::vector<int32_t>> i32; // SphereFormer-style int32 index tables
};

//! Assembles the batched tensor payloads for a set of patches (no ONNX Runtime involved)
BatchBuffers assembleBatchBuffers(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage = nullptr);

//! Builds the batched ONNX input values for a set of patches (batch axis = patches).
/** The backing data buffers are written into outF32/outI64/outI32 and MUST
    outlive the session run (CreateTensorWithDataAsOrtValue does not copy the
    data). **/
std::vector<OrtPtr<OrtValue>> buildBatchInputs(const ModelSpec& spec, const OrtSession& session, const std::vector<Patch>& patches, std::string* errorMessage = nullptr, std::vector<std::vector<float>>* outF32 = nullptr, std::vector<std::vector<int64_t>>* outI64 = nullptr, std::vector<std::vector<int32_t>>* outI32 = nullptr);

//! Warm-up probe: runs ONE forward pass on synthetic zero-filled inputs
/** The inputs are built directly from the session IO contract (batch 1,
    num_points = spec.numPoints or a fallback), so no patch building or model
    adapter is involved and this works for any model. It exercises the real
    graph end-to-end (driver DLLs, CUDA/cuDNN init, EP/graph compatibility) so
    ORT-level failures surface as catchable errors BEFORE the real run. A
    native crash inside a broken execution provider cannot be caught here -
    that is what the out-of-process qSemSegProbe is for.
    \return false with *errorMessage set when any input cannot be built or the run fails **/
bool probeSession(const OrtSession& session, const ModelSpec& spec, std::string* errorMessage = nullptr);

//! Calibrates a patch batch size on the live session
/** Times zero-filled forward passes for increasing batch sizes and returns
    the size with the best per-patch time (a larger size only wins when it is
    >= 10% faster per patch). The batch size only groups forward passes into
    one session.run - the patch sequence and the labels are unaffected, so
    auto-tuning is result-safe. \\return >= 1 (1 when calibration is not
    possible). **/
int autoSelectBatchSize(const OrtSession& session, const ModelSpec& spec, std::string* errorMessage = nullptr);

//! Readable note for batch-size calibration candidates that could not run
/** \param failedSizes batch sizes whose calibration forward failed (e.g. out
    of memory); empty -> empty note
    \param chosenBatch the batch size autoSelectBatchSize fell back to
    \param rawError the raw ORT error of the failed attempt (kept for the log) **/
std::string calibrationFailureNote(const std::vector<int>& failedSizes, int chosenBatch, const std::string& rawError);

//! The patch point count the run must use for this model
/** A model with a static num_points axis (fixed at export time, recorded in
    spec.staticNumPoints by validateSpecAgainstIO) only accepts exactly that
    many points per patch - the IO contract wins over the manifest/dialog
    value. 0 (dynamic axis) -> spec.numPoints. **/
int effectiveNumPoints(const ModelSpec& spec);

//! Interprets the model output and accumulates it into testProbs with
//! testProbs[idx] = smooth * testProbs[idx] + (1 - smooth) * probs
/** logits: rowCount * numClasses (row-major, one row per sampled point);
    idxs: subsampled-cloud index per row. Duplicated indices follow the numpy
    scatter semantics of the reference (later rows overwrite earlier ones).
    \param inputIsLogits when true (default), each row is softmaxed in float64
    (matching the reference's numpy softmax) and rounded to float32 first;
    when false the rows are already probabilities (manifest output.kind =
    "probabilities") and are used as-is - softmaxing them a second time would
    distort every value toward the class ranking and away from the true
    probabilities.
    \param rowsValid how many of the leading rows are real points; 0 (the
    default) means all rowCount rows are real. A model with a STATIC
    num_points axis must be fed exactly its fixed row count; when a patch is
    padded up to that size, the padded tail rows must not contribute (their
    idxs may repeat or be garbage): pass the real count here. **/
void accumulatePatchLogits(const float* logits, size_t rowCount, const std::vector<int64_t>& idxs, int numClasses, double smooth, std::vector<float>& testProbs, bool inputIsLogits = true, size_t rowsValid = 0);

//! Argmax over the class axis -> one label per subsampled point
std::vector<int> argmaxLabels(const std::vector<float>& testProbs, int numClasses);

//! Per-point argmax confidence: the accumulated probability of the argmax class
/** Row max of the [m x numClasses] testProbs buffer. Same accumulation
    semantics as the labels (0.95*old + 0.05*softmax + initial jitter), so the
    values converge toward 1 as coverage grows; a point covered only once
    stays near its first softmax value. **/
std::vector<float> maxProbabilities(const std::vector<float>& testProbs, int numClasses);

//! Estimated memory footprint (bytes) of the [m x numClasses] test_probs accumulation buffer
double testProbsMemoryBytes(size_t m, int numClasses);

//! Pre-flight guard: false (with an actionable errorMessage) when the estimated
//! test_probs buffer would exceed maxBytes (default 2 GB)
bool checkTestProbsMemory(size_t m, int numClasses, double maxBytes = 2.0 * 1024.0 * 1024.0 * 1024.0, std::string* errorMessage = nullptr);

//! True when every value is finite (neither NaN nor Inf); an empty range is finite
bool allFinite(const float* data, size_t count);

//! Thrown when a forward pass returns NaN/Inf values
/** A distinct type lets the plugin tell a PROVIDER problem (e.g. the CUDA
    provider producing NaNs for this graph) from generic errors, so Auto
    mode can retry the run on the CPU instead of aborting the cloud. **/
struct NonFiniteOutputError : std::runtime_error
{
	explicit NonFiniteOutputError(const std::string& message)
	    : std::runtime_error(message)
	{
	}
};

//! possibilities[idxs] += delta with the reference's numpy scatter semantics
/** The values are computed from the gathered (pre-update) possibilities and
    scattered in index order: duplicated indices end at the value of their LAST
    occurrence (they do not accumulate). **/
void addPossibilityDelta(std::vector<float>& possibilities, const std::vector<int64_t>& idxs, const std::vector<float>& delta);

//! One inference result: the predicted class per point and its confidence
/** labels / maxProbs are indexed like the ORIGINAL point cloud (the
    subsampled->original projection is already applied). maxProbs is the
    argmax class's accumulated probability (see maxProbabilities). **/
struct SegmentationResult
{
	std::vector<int>   labels;
	std::vector<float> maxProbs;
};

//! Full sliding-window inference. Returns one label + confidence per original point.
/** \param progressCallback optional, called after each batch with
    (patch index, covered fraction of the subsampled cloud: points with
    possibility > 0.5; rises steadily and reaches 1.0 at the stop threshold);
    return false from it to cancel the inference (then canceled is set and an
    empty result is returned)
**/
SegmentationResult runSlidingWindow(OrtSession& session, const ModelSpec& spec, CloudPre& pre, const SWParams& params, std::mt19937_64& rng, const std::function<bool(int patchIndex, float coverageFraction)>& progressCallback = {}, bool* canceled = nullptr);
