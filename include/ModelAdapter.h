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
#include "SlidingWindow.h" // Patch, CloudPre, BatchBuffers

#include <random>
#include <string>
#include <vector>

//! Adapter that teaches the plugin how one model family is fed
/** The sliding-window math (patch sampling, coverage walk, accumulation,
    projection) is model-independent. An adapter owns everything the MODEL
    decides:
    - defaults for fields no manifest source provided (applyDefaults),
    - the model-specific IO contract requirements (validate),
    - the per-patch tensor construction (buildPatch),
    - the batched tensor assembly (assembleBatch).
    Adapters are registered statically in ModelAdapter.cpp, keyed by the
    manifest's model_name (see findModelAdapter). A model whose model_name
    has no adapter is a loud validation error - never a silent fallback. **/
class ModelAdapter
{
  public:
	virtual ~ModelAdapter() = default;

	//! Registry key (e.g. "FKAConv") - matches the manifest model_name
	virtual const char* id() const = 0;

	//! Fills the adapter's defaults for fields no manifest source provided
	/** Called on the FINAL spec by loadModelSpec, only for fields that no
	    source (sidecar manifest / embedded metadata) provided. Must not
	    touch fields whose ManifestField bit is set. **/
	virtual void applyDefaults(ModelSpec& spec) const = 0;

	//! Model-specific IO validation; an empty string means compatible
	/** The generic checks (IO non-empty, features-in_channels match, static
	    num_points axis, output class count, positive parameters) stay in
	    validateSpecAgainstIO; this adds the adapter's own requirements. **/
	virtual std::string validate(const ModelSpec& spec, const IOContract& io) const = 0;

	//! Cloud-feature availability check for this family; empty string = OK
	/** The free validateCloudFeatures() wrapper in ModelSpec dispatches here.
	    FKAConv checks the declared feature_mode; generic models check the
	    extra channels of feature_columns (xyz + extras). **/
	virtual std::string validateCloudFeatures(const ModelSpec& spec, bool hasColors, bool hasIntensity) const = 0;

	//! Builds one patch's model inputs from the sampled indices
	/** Same contract as the free buildPatch() wrapper (fpsStarts/fixedMean
	    pin the FPS start / recenter mean for exact reference parity; both
	    null in production). **/
	virtual Patch buildPatch(const ModelSpec& spec, const CloudPre& pre, const std::vector<int64_t>& idxs, std::mt19937_64& rng, const std::vector<int64_t>* fpsStarts, const float* fixedMean) const = 0;

	//! Assembles the batched tensor payloads for a set of patches
	/** Same contract as the free assembleBatchBuffers() wrapper: empty
	    BatchBuffers with *errorMessage set on failure. **/
	virtual BatchBuffers assembleBatch(const ModelSpec& spec, const std::vector<Patch>& patches, std::string* errorMessage) const = 0;

	//! Points-per-patch value imposed by a static input axis (0 = none)
	/** The generic T13 detection only knows the FKAConv/PointNet-style
	    'points'/'features' channel-first inputs; adapters with a different
	    layout (e.g. SphereFormer's [num_points, channels] 'feat') report their
	    static axis here. Called by validateSpecAgainstIO when the generic
	    detection found nothing. **/
	virtual int64_t staticNumPointsFromIO(const IOContract& io) const
	{
		(void)io;
		return 0;
	}

	//! Batch size the exported graph is fixed to (0 = dynamic batch axis)
	/** A graph exported with NO dynamic batch axis (every input dim concrete,
	    batch baked to 1) cannot run more than one patch per forward: the run
	    must clamp the patch batch to this value and skip the calibration. **/
	virtual int fixedBatchSize(const ModelSpec& spec) const
	{
		(void)spec;
		return 0;
	}
};

//! Looks up the adapter registered for a manifest model_name (nullptr = unknown)
const ModelAdapter* findModelAdapter(const QString& modelName);

//! Resolves the adapter for a spec: the manifest's adapter.id wins over
//! model_name (both keyed against the same registry; nullptr = unknown)
const ModelAdapter* modelAdapterForSpec(const ModelSpec& spec);

//! Names of every registered adapter (for "supported: ..." error messages)
std::vector<std::string> modelAdapterNames();

//! Shared error text for a spec whose adapter could not be resolved
/** Names the failing key (adapter.id first, model_name otherwise) and the
    supported families; used by the IO validation and the build/assemble
    wrappers so the message is composed in exactly one place. **/
std::string unknownAdapterError(const ModelSpec& spec);

//! Batch size the spec's adapter fixes the graph to (0 = dynamic batch axis)
/** Free dispatch to fixedBatchSize() of the resolved adapter; 0 when no
    adapter is registered (unknown families fail loudly elsewhere) or the
    graph exports a dynamic batch axis. **/
int fixedBatchSize(const ModelSpec& spec);
