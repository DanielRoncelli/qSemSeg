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

#include <QString>
#include <QStringList>
#include <cstdint>
#include <string>
#include <vector>

class OrtSession;

//! The input/output tensor contract of an ONNX model (as read from a session)
/** Kept independent of OrtSession so validation is testable without a live
    ONNX Runtime (see IOContractFromSession). Shapes use the same template
    encoding as OrtSession: -1 = dynamic batch, -2 = dynamic num_points. **/
struct IOContract
{
	std::vector<std::string>          inputNames;
	std::vector<std::string>          inputDtypes; // "float32" | "int32" | "int64"
	std::vector<std::vector<int64_t>> inputShapes;
	std::vector<std::vector<int64_t>> outputShapes;
};

//! Everything the plugin needs to know about an ONNX segmentation model
/** Filled from the "<model>.onnx.meta.json" manifest written next to the
    exported model (num_points, voxel size, eval augmentation, class labels,
    FKAConv stages, feature mode, ...) and from the ONNX session itself
    (input/output contract, which is authoritative over the manifest). **/
struct ModelSpec
{
	//! How the model's per-point output must be interpreted before accumulation
	/** "logits" (default) are softmaxed (float64) during accumulation;
	    "probabilities" are already normalized and are used as-is. A wrong
	    interpretation silently produces wrong labels, so an unknown manifest
	    value is a hard error (see parseErrors), never a guess. **/
	enum class OutputKind
	{
		Logits        = 0,
		Probabilities = 1
	};

	QString modelName = "FKAConv";
	//! Registry key overriding model_name for the adapter lookup (manifest
	/** adapter.id lets a contributed model keep a descriptive model_name
	    (e.g. "Open3D-ML-Small") while pinning the code family that feeds it
	    ("PointNet" = the generic single-tensor path). Empty = model_name. **/
	QString adapterId;
	int     inChannels = 3;
	int     numClasses = 2;
	int     numPoints  = 2500;
	//! voxel-grid edge length in world units; <= 0 disables subsampling.
	/** double (not float) on purpose: the reference grid subsampling computes
	    the voxel keys as floor((double)point / (double)sampleDl) - a float
	    0.08f is a different number and can flip boundary points into the wrong
	    voxel (changing the subsampled point count). **/
	double           voxelSize   = 0.0;
	QString          featureMode = "xyz";
	std::vector<int> stageNpoints;         // FKAConv support points per stage
	int              numNeighbors = 16;    // FKAConv K
	bool             shiftMin     = false; // subtract the cloud minimum before subsampling (PointNet-style)

	// eval-time per-patch augmentation (manifest preprocessing.augment).
	// The reference applies recenter (subtract the per-dim patch mean) and
	// optionally normalize (subtract the mean again, divide by the largest
	// per-dim range) BEFORE building the features and the hierarchy.
	bool   recenter        = true;  // FKAConv/PointNet reference behavior
	bool   hasAugmentSpec  = false; // manifest carried an explicit augment spec
	bool   normalizePoints = false; // manifest augment.normalize.points
	bool   normalizeFeat   = false; // manifest augment.normalize.feat
	double featBias        = 0.0;   // normalize.feat bias
	double featScale       = 1.0;   // normalize.feat scale

	// order of the extra feature channels (after xyz), as used at training
	// time (manifest preprocessing.feature_columns); tokens: intensity, rgb
	std::vector<std::string> featureColumns;

	// dataset label value per model class index (manifest labels.values /
	// labels.names, or legacy top-level label_values). Empty = raw class
	// indices (with a warning at run time).
	std::vector<int>         labelValues;
	std::vector<std::string> labelNames;

	// Input/output contract (filled from the ORT session, which is authoritative)
	std::vector<std::string>          inputNames;
	std::vector<std::string>          inputDtypes; // "float32" | "int32" | "int64"
	std::vector<std::vector<int64_t>> inputShapes; // template: -1 = batch, -2 = num_points
	std::vector<std::vector<int64_t>> outputShapes;

	//! How to interpret the model output (manifest output.kind; default logits)
	OutputKind outputKind = OutputKind::Logits;

	//! Where the model prefers to run (manifest execution.provider)
	/** "auto" (default) leaves the decision to the plugin; "cpu" means the
	    export is known to be unreliable or much slower on the GPU (e.g.
	    graphs whose CUDA kernels produce NaN/Inf or take minutes per
	    forward), so Auto uses the CPU and skips the GPU probes entirely;
	    "cuda" / "dml" name one GPU family and fall back to the CPU when this
	    ONNX Runtime build has no such provider; "gpu" (legacy) prefers CUDA,
	    then DirectML, and still falls back to the CPU in Auto mode. An
	    explicit dialog choice (CPU / CUDA / DirectML) always wins. **/
	enum class ExecutionProvider
	{
		Auto = 0,
		Cpu  = 1,
		Gpu  = 2, //!< legacy "prefer a GPU": CUDA, then DirectML
		Cuda = 3,
		Dml  = 4
	};

	ExecutionProvider executionProvider = ExecutionProvider::Auto;

	//! num_points value imposed by a STATIC input axis (0 = dynamic axis)
	/** A model exported with a fixed num_points axis only accepts exactly
	    that many points per patch: the IO contract is authoritative, so
	    validateSpecAgainstIO records the value here and the run uses it
	    (effectiveNumPoints) instead of the manifest/dialog num_points when
	    they disagree (with a warning - never silently). **/
	int staticNumPoints = 0;

	//! How the cloud is subsampled before patch extraction (manifest preprocessing.subsampling)
	/** voxel: voxel-grid barycenters (the default, uses voxel_size);
	    none: no subsampling; random: rng choice without replacement;
	    fps: cloud-level farthest point sampling. random/fps need
	    subsampling.target. The projection back to the original cloud stays
	    exact for the subset types (the sub points ARE original points). **/
	enum class SubsamplingType
	{
		Voxel  = 0,
		None   = 1,
		Random = 2,
		Fps    = 3
	};

	SubsamplingType subsamplingType   = SubsamplingType::Voxel;
	int             subsamplingTarget = 0; // random/fps subset size (manifest subsampling.target)

	// SphereFormer per-level window/quant constants (manifest adapter.window)
	// Flat lists of 15 floats each (5 levels x 3 axes, level-major): the cart
	// branch works on xyz, the sphere branch on cart2sphere(xyz). The
	// SphereFormerAdapter requires these when the model is a SphereFormer.
	std::vector<float> sphereCarWindow;
	std::vector<float> sphereCarQuant;
	std::vector<float> sphereSphWindow;
	std::vector<float> sphereSphQuant;

	// SphereFormer structural capacities (manifest adapter.caps): feature-row
	// caps for levels 1..4 (level 0 = the patch point count, fixed by the IO),
	// plus per-branch/level window-count caps and attention-pair caps (flat
	// 10-entry lists, order car0..car4 then sph0..sph4). The exported graph
	// cannot grow its buffers: exceeding a cap is a loud runtime error
	// ("re-export with larger caps"), never silent truncation.
	std::vector<int64_t> sphereCapsC; // 4 entries (levels 1..4)
	std::vector<int64_t> sphereCapsW; // 10 entries
	std::vector<int64_t> sphereCapsM; // 10 entries

	//! HARD problems found while reading the manifest / embedded metadata
	/** Unlike warnings (recoverable assumptions), these make a run UNSAFE:
	    e.g. an unsupported manifest schema version, or an unknown output.kind.
	    Every consumer (plugin run, dialog, validation) must refuse to run when
	    this is non-empty and show the messages. **/
	QStringList parseErrors;

	// ---- knowledge-source bookkeeping (model card + console) ----
	//! Manifest keys that were EXPLICITLY provided by the loaded sources
	/** Precedence: sidecar manifest > embedded ONNX metadata > defaults.
	    Embedded metadata only fills the gaps (fields whose bit is unset). **/
	enum ManifestField : uint32_t
	{
		mbModelName      = 1u << 0,
		mbInChannels     = 1u << 1,
		mbNumClasses     = 1u << 2,
		mbNumPoints      = 1u << 3,
		mbVoxelSize      = 1u << 4,
		mbShiftMin       = 1u << 5,
		mbRecenter       = 1u << 6,
		mbNormalize      = 1u << 7, // augment.normalize (points and/or feat)
		mbFeatureColumns = 1u << 8,
		mbLabelValues    = 1u << 9,
		mbLabelNames     = 1u << 10,
		mbStageNpoints   = 1u << 11,
		mbNumNeighbors   = 1u << 12,
		mbFeatureMode    = 1u << 13,
		mbOutputKind     = 1u << 14,
		mbSubsampling    = 1u << 15,
		mbAdapterId      = 1u << 16,
		mbSphereWindow   = 1u << 17, // adapter.window (SphereFormer per-level constants)
		mbSphereCaps     = 1u << 18, // adapter.caps (SphereFormer structural capacities)
		mbExecution      = 1u << 19, // execution.provider (auto | cpu | cuda | dml | gpu)
	};
	bool     hasSidecarManifest  = false;
	bool     hasEmbeddedMetadata = false;
	uint32_t manifestFields      = 0; // ManifestField bits actually provided
};

//! Loads the model spec for the given .onnx file
/** The manifest is looked up as "<modelPath>.meta.json"; additionally the
    model file's own embedded metadata (ONNX key "semseg.manifest", stamped by
    Open3D-ML's convert_onnx.py) fills every field the sidecar did not set
    (precedence: sidecar manifest > embedded metadata > defaults). Missing
    sources are reported via \p warning.
    By default \p warning also contains the assumption warnings
    (assumptionWarnings) - the single console line covers everything that was
    assumed. When \p assumptionsOut is given, the assumptions are returned
    there INSTEAD (so a caller that renders them separately does not show
    them twice). **/
ModelSpec loadModelSpec(const QString& modelPath, QString* warning = nullptr, QStringList* assumptionsOut = nullptr);

//! Merges one embedded-manifest JSON string into the spec (gap-filling only)
/** Used with the metadata read from the ONNX file (readEmbeddedManifest)
    or from a live session (applyEmbeddedMetadata). Only fields that no
    source provided yet (unset ManifestField bits) are applied; the bits
    are set for what is applied, making repeated merges idempotent.
    Payload problems that make a run UNSAFE (unsupported schema version,
    unknown output.kind) are recorded in spec.parseErrors (hard errors);
    an unsupported schema version additionally prevents the payload from
    being applied at all.
    \return false when \p json is not a valid manifest JSON object, or when
    nothing could be applied (e.g. an unsupported schema version). **/
bool applyEmbeddedJsonToSpec(ModelSpec& spec, const std::string& json, QString* info = nullptr);

//! Applies the embedded ONNX metadata of a live session to the spec
/** Reads the custom metadata key "semseg.manifest" (API >= 14) and merges it
    with sidecar-manifest precedence (gap-filling only); sets
    spec.hasEmbeddedMetadata on success.
    \return true when embedded metadata was found and applied. **/
bool applyEmbeddedMetadata(ModelSpec& spec, const OrtSession& session, QString* info = nullptr);

//! Reads the embedded manifest JSON ("semseg.manifest") directly from an .onnx file
/** Parses the protobuf wire format of the ModelProto metadata_props without
    loading the graph (so the dialog/model card can use it before a session
    exists, and the tests can run without ONNX Runtime).
    \return true when the key was found (*jsonOut filled). **/
bool readEmbeddedManifest(const QString& onnxPath, std::string* jsonOut, QString* error = nullptr);

//! Names every spec field that no source (sidecar/embedded) provided
/** The guiding rule: anything that is guessed instead of known must warn,
    naming what was assumed. Called after all manifest sources are merged. **/
QStringList assumptionWarnings(const ModelSpec& spec);

//! GPU execution providers Auto mode considers for this spec, in priority order
/** ORT provider ids (strings, so no ONNX Runtime types leak here). Empty when
    the model declares CPU execution (execution.provider = "cpu"); "cuda" or
    "dml" restrict the list to that one GPU family (the other family is never
    tried with such a model - an unvalidated path is not used); "auto" / the
    legacy "gpu" / no hint return CUDA first, then DirectML. The dialog's
    pinned CUDA/DirectML choices bypass this function entirely. **/
std::vector<std::string> autoGpuCandidates(const ModelSpec& spec);

//! Fills spec.inputNames / inputDtypes / inputShapes / outputShapes from the session
void fillSpecFromSession(ModelSpec& spec, const OrtSession& session);

//! Extracts the IO contract from a session (for validateSpecAgainstIO)
IOContract readIOContract(const OrtSession& session);

//! Validates the manifest against the actual model IO contract
/** Returns an empty string when compatible, otherwise a human-readable
    description of the first incompatibility found (the session contract is
    authoritative: a manifest that disagrees with the graph is an error).
    Also records the model's STATIC num_points axis (if any) in
    spec.staticNumPoints: a graph with a fixed points-per-patch axis imposes
    that patch size on the whole pipeline. **/
std::string validateSpecAgainstIO(ModelSpec& spec, const IOContract& io);

//! Validates that the cloud provides the features the model requires
/** hasIntensity: an intensity-like scalar field is available.
    Returns an empty string when compatible, otherwise an actionable message. **/
std::string validateCloudFeatures(const ModelSpec& spec, bool hasColors, bool hasIntensity);

//! Runs every compatibility check for a model and formats them as rich text
/** The contributor on-ramp behind the "Validate model" dialog action: loads
    the model knowledge (sidecar manifest + embedded metadata), resolves the
    model adapter, checks the cloud feature requirements (needs no runtime)
    and - when ONNX Runtime is loadable - creates a CPU session and validates
    the manifest against the graph IO contract. The report starts with a
    PASS/FAIL headline followed by one rich-text line per check (check mark,
    cross, warning sign); no inference is run. When the runtime is NOT
    loadable the report says so and skips the session checks instead of
    failing (the manifest/family/feature checks still ran).
    \return true when no check failed (the model is usable). **/
bool buildModelValidationReport(const QString& modelPath, bool hasColors, bool hasIntensity, QString& report);
