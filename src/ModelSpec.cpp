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

#include "ModelSpec.h"

#include "ModelAdapter.h"
#include "OnnxRuntime.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>
#include <algorithm>
#include <cstring>

namespace
{
	QString joinList(const QStringList& items)
	{
		return items.join(QStringLiteral(", "));
	}

	//! Protobuf varint reader (wire-format base-128); false on truncation/malformed
	bool pbVarint(const unsigned char*& p, const unsigned char* end, uint64_t& out)
	{
		out            = 0;
		unsigned shift = 0;
		while (p < end)
		{
			const unsigned char byte = *p++;
			out |= static_cast<uint64_t>(byte & 0x7F) << shift;
			if (!(byte & 0x80))
			{
				return true;
			}
			shift += 7;
			if (shift >= 64)
			{
				return false;
			}
		}
		return false;
	}

	//! Skips one field of the given wire type; false when the buffer ends mid-field
	bool pbSkip(const unsigned char*& p, const unsigned char* end, uint64_t wireType)
	{
		switch (wireType)
		{
		case 0: // varint
		{
			uint64_t v;
			return pbVarint(p, end, v);
		}
		case 1: // 64-bit fixed
			if (end - p < 8)
			{
				return false;
			}
			p += 8;
			return true;
		case 2: // length-delimited
		{
			uint64_t len;
			if (!pbVarint(p, end, len) || static_cast<uint64_t>(end - p) < len)
			{
				return false;
			}
			p += len;
			return true;
		}
		case 5: // 32-bit fixed
			if (end - p < 4)
			{
				return false;
			}
			p += 4;
			return true;
		default: // groups (3,4) are deprecated and never appear in ONNX files
			return false;
		}
	}
} // namespace

//! Fills spec from one manifest JSON object
/** Shared by the sidecar manifest and the embedded ONNX metadata (they carry
    the same schema). Records which keys were explicitly present in
    spec.manifestFields so a lower-precedence source can fill only the gaps.
    Only STRUCTURAL problems (inconsistent label tables) warn here - fields
    that are simply ABSENT are reported once, after all sources are merged,
    by assumptionWarnings(). **/
void specFromManifest(const QJsonObject& meta, ModelSpec& spec, QStringList& warnings)
{
	// schema_version: absent = 1.x (the current schema). A manifest that
	// declares an unknown MAJOR version may interpret its fields differently,
	// so its contents are NOT applied at all and a hard parse error is
	// recorded (the consumer must refuse to run).
	if (meta.contains("schema_version"))
	{
		const QString     v     = meta.value("schema_version").toString();
		const QStringList parts = v.split(QLatin1Char('.'));
		bool              ok    = !parts.isEmpty() && parts.size() <= 2;
		int               major = 0;
		int               minor = 0;
		if (ok)
		{
			bool okMajor = false;
			major        = parts[0].toInt(&okMajor);
			ok           = okMajor && major >= 0;
			if (ok && parts.size() == 2)
			{
				bool okMinor = false;
				minor        = parts[1].toInt(&okMinor);
				ok           = okMinor && minor >= 0;
			}
		}
		if (!ok)
		{
			spec.parseErrors << QStringLiteral("schema_version \"%1\" is not a valid version string (expected MAJOR or MAJOR.MINOR, e.g. \"1.0\")").arg(v);
			return;
		}
		if (major != 1)
		{
			spec.parseErrors << QStringLiteral("schema version %1.%2 is not supported (this plugin understands 1.x); the manifest's contents were ignored").arg(major).arg(minor);
			return;
		}
	}

	if (meta.contains("model_name"))
	{
		spec.modelName = meta.value("model_name").toString(spec.modelName);
		spec.manifestFields |= ModelSpec::mbModelName;
	}
	if (meta.contains("in_channels"))
	{
		spec.inChannels = meta.value("in_channels").toInt(spec.inChannels);
		spec.manifestFields |= ModelSpec::mbInChannels;
	}
	if (meta.contains("num_classes"))
	{
		spec.numClasses = meta.value("num_classes").toInt(spec.numClasses);
		spec.manifestFields |= ModelSpec::mbNumClasses;
	}
	if (meta.contains("num_points"))
	{
		spec.numPoints = meta.value("num_points").toInt(spec.numPoints);
		spec.manifestFields |= ModelSpec::mbNumPoints;
	}

	const QJsonObject preprocessing = meta.value("preprocessing").toObject();
	if (preprocessing.contains("voxel_size") && !preprocessing.value("voxel_size").isNull())
	{
		spec.voxelSize = preprocessing.value("voxel_size").toDouble();
		spec.manifestFields |= ModelSpec::mbVoxelSize;
	}

	// subsampling menu: voxel (default), none, random (rng choice without
	// replacement) or fps (cloud-level farthest point sampling). Recoverable
	// problems (unknown type, missing target) warn and fall back - they do
	// not make the model un-runnable, they change the point counts.
	const QJsonObject subsampling = preprocessing.value("subsampling").toObject();
	if (!subsampling.isEmpty())
	{
		const QString type = subsampling.value("type").toString();
		if (type == QLatin1String("voxel"))
		{
			spec.subsamplingType = ModelSpec::SubsamplingType::Voxel;
			spec.manifestFields |= ModelSpec::mbSubsampling;
		}
		else if (type == QLatin1String("none"))
		{
			spec.subsamplingType = ModelSpec::SubsamplingType::None;
			spec.manifestFields |= ModelSpec::mbSubsampling;
		}
		else if (type == QLatin1String("random") || type == QLatin1String("fps"))
		{
			spec.subsamplingTarget = subsampling.value("target").toInt(0);
			if (spec.subsamplingTarget <= 0)
			{
				warnings << QStringLiteral("subsampling.type '%1' needs a positive subsampling.target - falling back to no subsampling").arg(type);
				spec.subsamplingType = ModelSpec::SubsamplingType::None;
			}
			else
			{
				spec.subsamplingType = (type == QLatin1String("random")) ? ModelSpec::SubsamplingType::Random : ModelSpec::SubsamplingType::Fps;
			}
			spec.manifestFields |= ModelSpec::mbSubsampling;
		}
		else
		{
			warnings << QStringLiteral("unknown subsampling.type \"%1\" (supported: voxel, none, random, fps) - keeping voxel subsampling").arg(type);
		}
	}

	// shift_min is explicit in the manifest/metadata (PointNet-style min-shift
	// before subsampling); when no source provides it, the model-name rule
	// decides after the merge (assumptionWarnings reports it)
	if (preprocessing.contains("shift_min"))
	{
		spec.shiftMin = preprocessing.value("shift_min").toBool(false);
		spec.manifestFields |= ModelSpec::mbShiftMin;
	}

	// eval-time augmentation: the reference applies recenter (subtract the
	// per-dim patch mean) before building features/hierarchy, and optionally
	// normalize. This is semantic preprocessing - it must come from the
	// manifest/metadata, never be inferred silently.
	const QJsonObject augment = preprocessing.value("augment").toObject();
	if (!augment.isEmpty())
	{
		spec.hasAugmentSpec = true;
		if (augment.contains("recenter"))
		{
			const QJsonValue rc = augment.value("recenter");
			spec.recenter       = rc.isBool()     ? rc.toBool()
			                      : rc.isDouble() ? (rc.toDouble() != 0.0)
			                                      : !rc.toObject().isEmpty();
			spec.manifestFields |= ModelSpec::mbRecenter;
		}
		const QJsonObject normalize = augment.value("normalize").toObject();
		if (!normalize.isEmpty())
		{
			const QJsonObject normPoints = normalize.value("points").toObject();
			spec.normalizePoints         = !normPoints.isEmpty();
			const QJsonObject normFeat   = normalize.value("feat").toObject();
			if (!normFeat.isEmpty())
			{
				spec.normalizeFeat = true;
				spec.featBias      = normFeat.value("bias").toDouble(0.0);
				spec.featScale     = normFeat.value("scale").toDouble(1.0);
			}
			spec.manifestFields |= ModelSpec::mbNormalize;
		}
	}

	// order of the extra feature channels (as at training time)
	const QJsonArray featureColumns = preprocessing.value("feature_columns").toArray();
	if (!featureColumns.isEmpty())
	{
		for (const QJsonValue& v : featureColumns)
		{
			spec.featureColumns.push_back(v.toString().toStdString());
		}
		spec.manifestFields |= ModelSpec::mbFeatureColumns;
	}

	// class index -> dataset label value (the models are trained on the
	// dataset's non-ignored labels in sorted order, e.g. ParisLille3D classes
	// 0..8 are dataset labels 1..9; writing raw class indices would shift every
	// label and collide with the ignored 'unclassified' label 0)
	QJsonObject      labels      = meta.value("labels").toObject();
	const QJsonArray labelValues = labels.isEmpty() ? meta.value("label_values").toArray() : labels.value("values").toArray();
	if (!labelValues.isEmpty())
	{
		for (const QJsonValue& v : labelValues)
		{
			spec.labelValues.push_back(v.toInt());
		}
		spec.manifestFields |= ModelSpec::mbLabelValues;
	}
	const QJsonArray labelNames = labels.isEmpty() ? QJsonArray() : labels.value("names").toArray();
	if (!labelNames.isEmpty())
	{
		for (const QJsonValue& v : labelNames)
		{
			spec.labelNames.push_back(v.toString().toStdString());
		}
		spec.manifestFields |= ModelSpec::mbLabelNames;
	}
	if (!spec.labelValues.empty() && spec.labelValues.size() != static_cast<size_t>(spec.numClasses))
	{
		warnings << QStringLiteral("labels.values has %1 entries but the model has %2 classes - ignoring the labels")
		                .arg(spec.labelValues.size())
		                .arg(spec.numClasses);
		spec.labelValues.clear();
		spec.labelNames.clear();
		spec.manifestFields &= ~(ModelSpec::mbLabelValues | ModelSpec::mbLabelNames);
	}
	if (!spec.labelNames.empty() && spec.labelNames.size() != static_cast<size_t>(spec.numClasses))
	{
		warnings << QStringLiteral("labels.names has %1 entries but the model has %2 classes - ignoring the names")
		                .arg(spec.labelNames.size())
		                .arg(spec.numClasses);
		spec.labelNames.clear();
		spec.manifestFields &= ~ModelSpec::mbLabelNames;
	}

	// how the model output must be interpreted: logits (default, softmaxed
	// here during accumulation) or already-normalized probabilities. An
	// unknown value is fatal (recorded as a parse error): guessing would
	// silently misinterpret every model output. The other fields still parse.
	const QJsonObject outputObj = meta.value("output").toObject();
	if (outputObj.contains("kind"))
	{
		const QString kind = outputObj.value("kind").toString();
		if (kind == QLatin1String("logits"))
		{
			spec.outputKind = ModelSpec::OutputKind::Logits;
			spec.manifestFields |= ModelSpec::mbOutputKind;
		}
		else if (kind == QLatin1String("probabilities"))
		{
			spec.outputKind = ModelSpec::OutputKind::Probabilities;
			spec.manifestFields |= ModelSpec::mbOutputKind;
		}
		else
		{
			spec.parseErrors << QStringLiteral("output.kind \"%1\" is not one of \"logits\" or \"probabilities\"").arg(kind);
		}
	}

	// where the model prefers to run: CPU-only exports (GPU kernels that
	// produce NaN/Inf or take minutes per forward) can declare it here so
	// Auto mode skips the GPU probe instead of wasting a forward pass. An
	// unknown value is fatal (a typo must not silently change the policy).
	const QJsonObject executionObj = meta.value("execution").toObject();
	if (executionObj.contains("provider"))
	{
		const QString provider = executionObj.value("provider").toString();
		if (provider == QLatin1String("auto"))
		{
			spec.executionProvider = ModelSpec::ExecutionProvider::Auto;
			spec.manifestFields |= ModelSpec::mbExecution;
		}
		else if (provider == QLatin1String("cpu"))
		{
			spec.executionProvider = ModelSpec::ExecutionProvider::Cpu;
			spec.manifestFields |= ModelSpec::mbExecution;
		}
		else if (provider == QLatin1String("gpu"))
		{
			spec.executionProvider = ModelSpec::ExecutionProvider::Gpu;
			spec.manifestFields |= ModelSpec::mbExecution;
		}
		else if (provider == QLatin1String("cuda"))
		{
			spec.executionProvider = ModelSpec::ExecutionProvider::Cuda;
			spec.manifestFields |= ModelSpec::mbExecution;
		}
		else if (provider == QLatin1String("dml"))
		{
			spec.executionProvider = ModelSpec::ExecutionProvider::Dml;
			spec.manifestFields |= ModelSpec::mbExecution;
		}
		else
		{
			spec.parseErrors << QStringLiteral("execution.provider \"%1\" is not one of \"auto\", \"cpu\", \"cuda\", \"dml\" or \"gpu\"").arg(provider);
		}
	}

	const QJsonObject adapter      = meta.value("adapter").toObject();
	const QJsonArray  stageNpoints = adapter.value("stage_npoints").toArray();
	if (!stageNpoints.isEmpty())
	{
		for (const QJsonValue& v : stageNpoints)
		{
			spec.stageNpoints.push_back(v.toInt());
		}
		spec.manifestFields |= ModelSpec::mbStageNpoints;
	}
	if (adapter.contains("num_neighbors"))
	{
		spec.numNeighbors = adapter.value("num_neighbors").toInt(spec.numNeighbors);
		spec.manifestFields |= ModelSpec::mbNumNeighbors;
	}
	if (adapter.contains("feature_mode"))
	{
		spec.featureMode = adapter.value("feature_mode").toString(spec.featureMode);
		spec.manifestFields |= ModelSpec::mbFeatureMode;
	}
	// adapter.id pins the code family feeding the model, independent of a
	// descriptive model_name (registry lookup order: adapter.id -> model_name)
	if (adapter.contains("id") && !adapter.value("id").toString().isEmpty())
	{
		spec.adapterId = adapter.value("id").toString();
		spec.manifestFields |= ModelSpec::mbAdapterId;
	}

	// SphereFormer per-level window/quant constants (adapter.window). A wrong
	// shape is a hard parse error: guessing window sizes would feed the
	// attention a partition the exported graph cannot represent.
	const QJsonObject window = adapter.value("window").toObject();
	if (!window.isEmpty())
	{
		struct WindowList
		{
			const char*         key;
			std::vector<float>* out;
		};
		const WindowList lists[] = {
		    {"car_window", &spec.sphereCarWindow},
		    {"car_quant", &spec.sphereCarQuant},
		    {"sph_window", &spec.sphereSphWindow},
		    {"sph_quant", &spec.sphereSphQuant},
		};
		bool ok = true;
		for (const WindowList& entry : lists)
		{
			const QJsonArray arr = window.value(entry.key).toArray();
			if (arr.size() != 15)
			{
				spec.parseErrors << QStringLiteral("adapter.window.%1 must be a flat list of 5 levels x 3 axes = 15 numbers (got %2)").arg(entry.key).arg(arr.size());
				ok = false;
				continue;
			}
			entry.out->clear();
			for (const QJsonValue& v : arr)
			{
				entry.out->push_back(static_cast<float>(v.toDouble()));
			}
		}
		if (ok)
		{
			spec.manifestFields |= ModelSpec::mbSphereWindow;
		}
	}

	// SphereFormer structural capacities (adapter.caps): C = row caps for
	// levels 1..4 (level 0 is null - it is the patch point count fixed by the
	// IO), W/M = per-branch/level window-count / attention-pair caps as
	// objects with the car0..car4 / sph0..sph4 keys.
	const QJsonObject caps = adapter.value("caps").toObject();
	if (!caps.isEmpty())
	{
		const QJsonArray cArr = caps.value("C").toArray();
		bool             cOk  = cArr.size() == 5 && cArr.at(0).isNull();
		if (cOk)
		{
			spec.sphereCapsC.clear();
			for (int l = 1; l < 5; l++)
			{
				const QJsonValue v = cArr.at(l);
				if (!v.isDouble())
				{
					spec.parseErrors << QStringLiteral("adapter.caps.C[%1] must be an integer row capacity (levels 1..4); level 0 must be null (the patch point count)").arg(l);
					cOk = false;
					break;
				}
				spec.sphereCapsC.push_back(static_cast<int64_t>(v.toDouble()));
			}
		}
		else
		{
			spec.parseErrors << QStringLiteral("adapter.caps.C must be a list of 5 entries with null at level 0 and the row capacities of levels 1..4");
			cOk = false;
		}

		// W/M: flat 10-entry lists in car0..car4, sph0..sph4 order
		const auto readCapMap = [&spec, &caps](const char* key, std::vector<int64_t>* out) -> bool
		{
			const QJsonObject obj = caps.value(key).toObject();
			if (obj.isEmpty())
			{
				spec.parseErrors << QStringLiteral("adapter.caps.%1 must be an object with the keys car0..car4 and sph0..sph4").arg(key);
				return false;
			}
			out->clear();
			out->reserve(10);
			bool ok = true;
			for (int b = 0; b < 2 && ok; b++)
			{
				const char* prefix = b == 0 ? "car" : "sph";
				for (int l = 0; l < 5 && ok; l++)
				{
					const QString    k = QStringLiteral("%1%2").arg(prefix).arg(l);
					const QJsonValue v = obj.value(k);
					if (!v.isDouble())
					{
						spec.parseErrors << QStringLiteral("adapter.caps.%1 is missing the numeric key \"%2\"").arg(key).arg(k);
						ok = false;
						break;
					}
					out->push_back(static_cast<int64_t>(v.toDouble()));
				}
			}
			return ok;
		};
		const bool wOk = readCapMap("W", &spec.sphereCapsW);
		const bool mOk = readCapMap("M", &spec.sphereCapsM);
		if (cOk && wOk && mOk)
		{
			spec.manifestFields |= ModelSpec::mbSphereCaps;
		}
	}
}

QStringList assumptionWarnings(const ModelSpec& spec)
{
	QStringList warnings;

	// preprocessing assumptions: ONE bundled line (same category - values the
	// model metadata does not state, so the plugin decides)
	QStringList preprocessing;
	if (!(spec.manifestFields & ModelSpec::mbShiftMin))
	{
		preprocessing << QStringLiteral("shift_min = %1 (points are shifted by the cloud minimum before subsampling)")
		                     .arg(spec.shiftMin ? "true" : "false");
	}
	if (!(spec.manifestFields & ModelSpec::mbRecenter))
	{
		preprocessing << QStringLiteral("recenter = %1 (each patch is recentered on its mean)")
		                     .arg(spec.recenter ? "true" : "false");
	}
	if (!preprocessing.isEmpty())
	{
		warnings << QStringLiteral("the model does not specify all preprocessing steps - assuming defaults: %1")
		                .arg(preprocessing.join(QStringLiteral(", ")));
	}
	if (!(spec.manifestFields & ModelSpec::mbLabelValues))
	{
		warnings << QStringLiteral("the model does not provide labels.values - the output scalar field will contain the model's class indices (0..%1), not dataset labels")
		                .arg(spec.numClasses - 1);
	}
	return warnings;
}

std::vector<std::string> autoGpuCandidates(const ModelSpec& spec)
{
	// the model declares CPU execution: never probe a GPU for it
	if (spec.executionProvider == ModelSpec::ExecutionProvider::Cpu)
	{
		return {};
	}
	// a named GPU family is the only GPU family Auto may try: the export was
	// validated there, the other family is not a supported path for it
	if (spec.executionProvider == ModelSpec::ExecutionProvider::Cuda)
	{
		return {"CUDAExecutionProvider"};
	}
	if (spec.executionProvider == ModelSpec::ExecutionProvider::Dml)
	{
		return {"DmlExecutionProvider"};
	}
	// "auto" / legacy "gpu" / no hint: CUDA first (longest-supported in this
	// plugin), then DirectML (any DX12 GPU)
	return {"CUDAExecutionProvider", "DmlExecutionProvider"};
}

ModelSpec loadModelSpec(const QString& modelPath, QString* warning, QStringList* assumptionsOut)
{
	ModelSpec   spec;
	QStringList warnings;

	const QString metaPath = modelPath + ".meta.json";

	// ---- source 1: sidecar manifest "<model>.onnx.meta.json" (highest precedence) ----
	{
		QFile metaFile(metaPath);
		if (metaFile.open(QIODevice::ReadOnly))
		{
			const QJsonDocument doc = QJsonDocument::fromJson(metaFile.readAll());
			if (doc.isObject())
			{
				const qsizetype before       = spec.parseErrors.size();
				const uint32_t  fieldsBefore = spec.manifestFields;
				specFromManifest(doc.object(), spec, warnings);
				// hard errors are prefixed with the file they came from
				for (qsizetype i = before; i < spec.parseErrors.size(); ++i)
				{
					spec.parseErrors[i] = QStringLiteral("manifest '%1': %2").arg(metaPath, spec.parseErrors[i]);
				}
				// "has a sidecar" means it contributed knowledge (an
				// unsupported-schema or empty manifest did not)
				spec.hasSidecarManifest = spec.manifestFields != fieldsBefore;
			}
			else
			{
				warnings << QStringLiteral("Manifest '%1' is not a valid JSON object - ignoring it").arg(metaPath);
			}
		}
	}

	// ---- source 2: the model's embedded ONNX metadata (fills the gaps) ----
	{
		std::string embeddedJson;
		QString     embeddedError;
		if (readEmbeddedManifest(modelPath, &embeddedJson, &embeddedError))
		{
			QString info;
			if (applyEmbeddedJsonToSpec(spec, embeddedJson, &info))
			{
				spec.hasEmbeddedMetadata = true;
			}
			if (!info.isEmpty())
			{
				warnings << info;
			}
		}
		else if (!embeddedError.isEmpty())
		{
			warnings << embeddedError;
		}
	}

	// ---- default: the model family's own defaults (assumptionWarnings reports
	// them). Applied to the FINAL spec after all merges: copying a source's
	// rule-derived shiftMin would be wrong when e.g. the sidecar knows
	// model_name but no source declares shift_min explicitly.
	if (!(spec.manifestFields & ModelSpec::mbShiftMin))
	{
		if (const ModelAdapter* adapter = modelAdapterForSpec(spec))
		{
			adapter->applyDefaults(spec);
		}
	}

	const QStringList assumed = assumptionWarnings(spec);
	QStringList       all     = warnings;
	if (assumptionsOut)
	{
		// the caller renders the assumptions itself (e.g. one line each);
		// keep them out of the composed warning so they are not shown twice
		*assumptionsOut = assumed;
	}
	else
	{
		all << assumed;
	}

	if (warning)
	{
		if (!spec.hasSidecarManifest && !spec.hasEmbeddedMetadata)
		{
			const QString strong = QStringLiteral("No manifest '%1' found and the model carries no embedded "
			                                      "metadata, using default model parameters (voxel size, eval "
			                                      "augmentation and class labels are NOT known without them "
			                                      "- predictions may be wrong)")
			                           .arg(metaPath);
			*warning = all.isEmpty() ? strong : strong + QLatin1String(" | ") + joinList(all);
		}
		else if (!all.isEmpty())
		{
			*warning = joinList(all);
		}
	}
	return spec;
}

bool applyEmbeddedJsonToSpec(ModelSpec& spec, const std::string& json, QString* info)
{
	const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(json));
	if (!doc.isObject())
	{
		if (info)
		{
			*info = QStringLiteral("the embedded model metadata is not a valid JSON object - ignored");
		}
		return false;
	}

	// Merge into a scratch spec first, then copy only the fields no other
	// source provided yet (precedence: sidecar manifest > embedded metadata
	// > defaults). The ManifestField bits make the merge idempotent: applying
	// the same metadata twice changes nothing the second time.
	// The scratch numClasses is seeded from the target so the embedded label
	// tables are checked against the best class count known so far.
	ModelSpec   scratch;
	QStringList structural;
	scratch.modelName  = spec.modelName;
	scratch.numClasses = spec.numClasses;
	specFromManifest(doc.object(), scratch, structural);

	// parse errors are hard problems of the payload itself: they always
	// propagate (they are not a field the precedence rule could veto), and
	// they are prefixed so they cannot be mistaken for sidecar problems
	for (const QString& e : scratch.parseErrors)
	{
		spec.parseErrors << QStringLiteral("embedded metadata: %1").arg(e);
	}

	const auto copyField = [&spec, &scratch](ModelSpec::ManifestField bit, auto&& apply)
	{
		if (!(spec.manifestFields & bit) && (scratch.manifestFields & bit))
		{
			apply();
			spec.manifestFields |= bit;
		}
	};

	copyField(ModelSpec::mbModelName,
	          [&]
	          {
		          spec.modelName = scratch.modelName;
	          });
	copyField(ModelSpec::mbInChannels,
	          [&]
	          {
		          spec.inChannels = scratch.inChannels;
	          });
	copyField(ModelSpec::mbNumClasses,
	          [&]
	          {
		          spec.numClasses = scratch.numClasses;
	          });
	copyField(ModelSpec::mbNumPoints,
	          [&]
	          {
		          spec.numPoints = scratch.numPoints;
	          });
	copyField(ModelSpec::mbVoxelSize,
	          [&]
	          {
		          spec.voxelSize = scratch.voxelSize;
	          });
	copyField(ModelSpec::mbShiftMin,
	          [&]
	          {
		          spec.shiftMin = scratch.shiftMin;
	          });
	copyField(ModelSpec::mbRecenter,
	          [&]
	          {
		          spec.recenter = scratch.recenter;
	          });
	copyField(ModelSpec::mbNormalize,
	          [&]
	          {
		          spec.normalizePoints = scratch.normalizePoints;
		          spec.normalizeFeat   = scratch.normalizeFeat;
		          spec.featBias        = scratch.featBias;
		          spec.featScale       = scratch.featScale;
	          });
	copyField(ModelSpec::mbFeatureColumns,
	          [&]
	          {
		          spec.featureColumns = scratch.featureColumns;
	          });
	// label tables: additionally consistent with the FINAL class count (the
	// scratch validated them against its own numClasses, but a sidecar may
	// have won a different one) - a mismatching table is reported, not applied
	QStringList extraWarnings;
	const auto  copyLabelTables = [&]
	{
		const auto count = static_cast<size_t>(spec.numClasses);
		if (!(spec.manifestFields & ModelSpec::mbLabelValues) && (scratch.manifestFields & ModelSpec::mbLabelValues))
		{
			if (scratch.labelValues.size() == count)
			{
				spec.labelValues = scratch.labelValues;
				spec.manifestFields |= ModelSpec::mbLabelValues;
			}
			else
			{
				extraWarnings << QStringLiteral("labels.values has %1 entries but the model has %2 classes - not applied")
				                     .arg(scratch.labelValues.size())
				                     .arg(spec.numClasses);
			}
		}
		if (!(spec.manifestFields & ModelSpec::mbLabelNames) && (scratch.manifestFields & ModelSpec::mbLabelNames))
		{
			if (scratch.labelNames.size() == count)
			{
				spec.labelNames = scratch.labelNames;
				spec.manifestFields |= ModelSpec::mbLabelNames;
			}
			else
			{
				extraWarnings << QStringLiteral("labels.names has %1 entries but the model has %2 classes - not applied")
				                     .arg(scratch.labelNames.size())
				                     .arg(spec.numClasses);
			}
		}
	};
	copyLabelTables();
	copyField(ModelSpec::mbStageNpoints,
	          [&]
	          {
		          spec.stageNpoints = scratch.stageNpoints;
	          });
	copyField(ModelSpec::mbNumNeighbors,
	          [&]
	          {
		          spec.numNeighbors = scratch.numNeighbors;
	          });
	copyField(ModelSpec::mbFeatureMode,
	          [&]
	          {
		          spec.featureMode = scratch.featureMode;
	          });
	copyField(ModelSpec::mbOutputKind,
	          [&]
	          {
		          spec.outputKind = scratch.outputKind;
	          });
	copyField(ModelSpec::mbExecution,
	          [&]
	          {
		          spec.executionProvider = scratch.executionProvider;
	          });
	copyField(ModelSpec::mbSubsampling,
	          [&]
	          {
		          spec.subsamplingType   = scratch.subsamplingType;
		          spec.subsamplingTarget = scratch.subsamplingTarget;
	          });
	copyField(ModelSpec::mbAdapterId,
	          [&]
	          {
		          spec.adapterId = scratch.adapterId;
	          });
	copyField(ModelSpec::mbSphereWindow,
	          [&]
	          {
		          spec.sphereCarWindow = scratch.sphereCarWindow;
		          spec.sphereCarQuant  = scratch.sphereCarQuant;
		          spec.sphereSphWindow = scratch.sphereSphWindow;
		          spec.sphereSphQuant  = scratch.sphereSphQuant;
	          });
	copyField(ModelSpec::mbSphereCaps,
	          [&]
	          {
		          spec.sphereCapsC = scratch.sphereCapsC;
		          spec.sphereCapsW = scratch.sphereCapsW;
		          spec.sphereCapsM = scratch.sphereCapsM;
	          });
	spec.hasAugmentSpec = spec.hasAugmentSpec || scratch.hasAugmentSpec;

	// structural problems of the embedded payload itself (e.g. an
	// inconsistent label table) are reported prefixed, so they cannot be
	// mistaken for problems of the winning source
	if (info)
	{
		QStringList all = structural;
		all << extraWarnings;
		QStringList prefixed;
		for (const QString& w : all)
		{
			prefixed << QStringLiteral("embedded metadata: %1").arg(w);
		}
		if (!prefixed.isEmpty())
		{
			*info = joinList(prefixed);
		}
	}

	// nothing applied at all (e.g. an unsupported schema version): report
	// false so the caller does not count the payload as a knowledge source
	return scratch.manifestFields != 0;
}

bool applyEmbeddedMetadata(ModelSpec& spec, const OrtSession& session, QString* info)
{
	std::string json;
	std::string lookupError;
	if (!session.lookupModelMetadata("semseg.manifest", &json, &lookupError))
	{
		if (info && !lookupError.empty())
		{
			*info = QString::fromStdString(lookupError);
		}
		return false;
	}
	const bool applied = applyEmbeddedJsonToSpec(spec, json, info);
	if (applied)
	{
		spec.hasEmbeddedMetadata = true;
	}
	return applied;
}

bool readEmbeddedManifest(const QString& onnxPath, std::string* jsonOut, QString* error)
{
	if (jsonOut)
	{
		jsonOut->clear();
	}

	QFile file(onnxPath);
	if (!file.open(QIODevice::ReadOnly))
	{
		// a missing/unreadable model file is reported where it matters (the
		// session creation); stay quiet here so manifest-only flows (dialog,
		// tests) stay clean
		return false;
	}
	// scanning needs the bytes in memory: guard against pathological files
	constexpr qint64 kMaxScanBytes = 1536LL * 1024 * 1024;
	if (file.size() > kMaxScanBytes)
	{
		if (error)
		{
			*error = QStringLiteral("model file is too large to scan for embedded metadata");
		}
		return false;
	}
	const QByteArray     bytes = file.readAll();
	const unsigned char* p     = reinterpret_cast<const unsigned char*>(bytes.constData());
	const unsigned char* end   = p + bytes.size();

	auto fail = [&error](const char* why)
	{
		if (error)
		{
			*error = QString::fromUtf8(why);
		}
		return false;
	};
	const char* notParsable = "the model file is not a parsable ONNX protobuf";

	// ModelProto walk: metadata_props is field 14, a repeated
	// StringStringEntryProto (key = field 1, value = field 2)
	while (p < end)
	{
		uint64_t tag;
		if (!pbVarint(p, end, tag) || (tag >> 3) == 0)
		{
			return fail(notParsable);
		}
		const uint64_t fieldNumber = tag >> 3;
		const uint64_t wireType    = tag & 7;
		if (wireType != 2)
		{
			if (!pbSkip(p, end, wireType))
			{
				return fail(notParsable);
			}
			continue;
		}
		uint64_t len;
		if (!pbVarint(p, end, len) || static_cast<uint64_t>(end - p) < len)
		{
			return fail(notParsable);
		}
		const unsigned char* data = p;
		p += len;
		if (fieldNumber != 14)
		{
			continue;
		}
		const unsigned char* ep     = data;
		const unsigned char* eend   = data + len;
		const char*          key    = nullptr;
		size_t               keyLen = 0;
		const char*          value  = nullptr;
		size_t               valLen = 0;
		while (ep < eend)
		{
			uint64_t etag;
			if (!pbVarint(ep, eend, etag) || (etag >> 3) == 0)
			{
				return fail(notParsable);
			}
			const uint64_t efield = etag >> 3;
			const uint64_t ewire  = etag & 7;
			if (ewire == 2 && (efield == 1 || efield == 2))
			{
				uint64_t elen;
				if (!pbVarint(ep, eend, elen) || static_cast<uint64_t>(eend - ep) < elen)
				{
					return fail(notParsable);
				}
				if (efield == 1)
				{
					key    = reinterpret_cast<const char*>(ep);
					keyLen = elen;
				}
				else
				{
					value  = reinterpret_cast<const char*>(ep);
					valLen = elen;
				}
				ep += elen;
			}
			else if (!pbSkip(ep, eend, ewire))
			{
				return fail(notParsable);
			}
		}
		if (key && value && keyLen == 15 && std::memcmp(key, "semseg.manifest", 15) == 0)
		{
			if (jsonOut)
			{
				jsonOut->assign(value, valLen);
			}
			return true;
		}
	}
	return false; // valid protobuf, but no embedded manifest
}

void fillSpecFromSession(ModelSpec& spec, const OrtSession& session)
{
	IOContract io     = readIOContract(session);
	spec.inputNames   = io.inputNames;
	spec.inputDtypes  = io.inputDtypes;
	spec.inputShapes  = io.inputShapes;
	spec.outputShapes = io.outputShapes;
}

IOContract readIOContract(const OrtSession& session)
{
	IOContract io;
	io.inputNames = session.inputNames;
	io.inputDtypes.clear();
	for (ONNXTensorElementDataType type : session.inputTypes)
	{
		// int32 must be mapped distinctly (SphereFormer-style fixed-shape models
		// carry 60+ int32 index-table inputs); anything else but float/int64
		// falls back to float32 (its tensors are created as float32 and the
		// validation rejects them as soon as a family check looks at them)
		switch (type)
		{
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			io.inputDtypes.push_back("int64");
			break;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			io.inputDtypes.push_back("int32");
			break;
		default:
			io.inputDtypes.push_back("float32");
			break;
		}
	}
	io.inputShapes  = session.inputShapes;
	io.outputShapes = session.outputShapes;
	return io;
}

std::string validateSpecAgainstIO(ModelSpec& spec, const IOContract& io)
{
	if (io.inputNames.empty())
	{
		return "the model has no inputs";
	}
	if (io.outputShapes.empty())
	{
		return "the model has no outputs";
	}

	// the model family owns its adapter-specific requirements; an unknown
	// model family is a loud error, never a silent fallback to some family
	const ModelAdapter* adapter = modelAdapterForSpec(spec);
	if (!adapter)
	{
		return unknownAdapterError(spec);
	}
	if (std::string adapterError = adapter->validate(spec, io); !adapterError.empty())
	{
		return adapterError;
	}

	// the 'features' input must match the manifest in_channels (when present)
	auto       featuresIt       = std::find(io.inputNames.begin(), io.inputNames.end(), std::string("features"));
	const bool hasFeaturesInput = featuresIt != io.inputNames.end();
	if (hasFeaturesInput)
	{
		const size_t                featuresIdx = static_cast<size_t>(featuresIt - io.inputNames.begin());
		const std::vector<int64_t>& shape       = io.inputShapes[featuresIdx];
		if (shape.size() < 2)
		{
			return "the 'features' input must be at least 2-dimensional";
		}
		if (shape[1] > 0 && shape[1] != spec.inChannels)
		{
			return "the graph's 'features' input expects " + std::to_string(shape[1]) + " channels, but the manifest declares in_channels=" + std::to_string(spec.inChannels);
		}
	}

	// the output's last axis is the class axis: it must match numClasses
	const std::vector<int64_t>& outputShape = io.outputShapes[0];
	if (outputShape.size() < 2)
	{
		return "the model output must be at least 2-dimensional [.., num_classes]";
	}
	const int64_t outputClasses = outputShape.back();
	if (outputClasses > 0 && outputClasses != spec.numClasses)
	{
		return "the graph output has " + std::to_string(outputClasses) + " classes, but the manifest declares num_classes=" + std::to_string(spec.numClasses);
	}

	if (spec.numPoints <= 0)
	{
		return "num_points must be positive (manifest num_points / dialog value)";
	}
	if (spec.voxelSize < 0.0)
	{
		return "voxel_size must not be negative";
	}

	// ---- static num_points axis (T13) ----
	// A model exported with a FIXED points-per-patch axis carries a concrete
	// value where a dynamic model has the -2 placeholder. The patch inputs
	// are channel-first ([B, C, N]), so the num_points axis is the LAST axis
	// of each float patch input. A static axis imposes the patch size on the
	// whole pipeline (the graph rejects anything else); the IO contract is
	// authoritative over the manifest/dialog num_points.
	const auto staticPointsAxis = [&io](const std::string& name, int64_t* valueOut)
	{
		*valueOut = 0;
		for (size_t in = 0; in < io.inputNames.size(); in++)
		{
			// only FLOAT patch inputs carry the points axis (int64/int32 inputs
			// are index tables whose last axis is a tap count, not points)
			if (io.inputNames[in] != name || io.inputDtypes[in] != "float32")
			{
				continue;
			}
			const std::vector<int64_t>& shape = io.inputShapes[in];
			if (shape.size() >= 2 && shape.back() > 0)
			{
				*valueOut = shape.back();
				return;
			}
		}
	};
	int64_t staticPoints   = 0;
	int64_t staticFeatures = 0;
	staticPointsAxis("points", &staticPoints);
	staticPointsAxis("features", &staticFeatures);
	if (staticPoints > 0 && staticFeatures > 0 && staticPoints != staticFeatures)
	{
		return "the graph's 'points' input fixes " + std::to_string(staticPoints) + " points per patch but its 'features' input fixes "
		       + std::to_string(staticFeatures) + " - the graph is inconsistent and cannot be fed";
	}
	// any static axis wins (a dynamic sibling just follows the imposed size)
	spec.staticNumPoints = static_cast<int>(staticPoints > 0 ? staticPoints : staticFeatures);

	// adapters with a different input layout (e.g. SphereFormer's
	// [num_points, channels] 'feat' tensor) report their own static axis when
	// the generic detection found nothing
	if (spec.staticNumPoints == 0)
	{
		const int64_t adapterStatic = adapter->staticNumPointsFromIO(io);
		if (adapterStatic > 0)
		{
			spec.staticNumPoints = static_cast<int>(adapterStatic);
		}
	}

	return {};
}

std::string validateCloudFeatures(const ModelSpec& spec, bool hasColors, bool hasIntensity)
{
	// the family decides how feature columns are required (the checks live
	// in the adapters since the PointNet bits moved there; an unknown model
	// family is a loud error)
	const ModelAdapter* adapter = modelAdapterForSpec(spec);
	if (!adapter)
	{
		return unknownAdapterError(spec);
	}
	return adapter->validateCloudFeatures(spec, hasColors, hasIntensity);
}

bool buildModelValidationReport(const QString& modelPath, bool hasColors, bool hasIntensity, QString& report)
{
	const QString okMark   = QStringLiteral("<span style=\"color: #2e7d32; font-weight: bold;\">&#10003;</span>");
	const QString failMark = QStringLiteral("<span style=\"color: #b00020; font-weight: bold;\">&#10007;</span>");
	const QString warnMark = QStringLiteral("<span style=\"color: #b36b00; font-weight: bold;\">&#9888;</span>");

	QStringList lines;
	bool        ok = true;

	if (modelPath.trimmed().isEmpty())
	{
		report = QStringLiteral("<b><span style=\"color: #b00020;\">&#10007; Model validation FAILED</span></b><br>") + failMark
		         + QStringLiteral(" no model file given");
		return false;
	}

	// ---- model knowledge: sidecar manifest + embedded metadata ----
	// the assumptions are split out of the composed warning so they can be
	// rendered as their own lines (loadModelSpec would otherwise include
	// them there AND the section below would repeat them)
	QString     manifestWarning;
	QStringList assumedWarnings;
	ModelSpec   spec = loadModelSpec(modelPath, &manifestWarning, &assumedWarnings);
	if (!manifestWarning.isEmpty())
	{
		lines << warnMark + QLatin1Char(' ') + manifestWarning.toHtmlEscaped();
	}
	if (!spec.parseErrors.isEmpty())
	{
		// hard manifest problems (unsupported schema version, unknown
		// output.kind): the model would be misinterpreted - fail loudly
		ok = false;
		for (const QString& error : spec.parseErrors)
		{
			lines << failMark + QLatin1Char(' ') + error.toHtmlEscaped();
		}
	}
	else
	{
		QString knowledge;
		if (spec.hasSidecarManifest && spec.hasEmbeddedMetadata)
		{
			knowledge = QStringLiteral("sidecar manifest + embedded model metadata");
		}
		else if (spec.hasSidecarManifest)
		{
			knowledge = QStringLiteral("sidecar manifest (<model>.onnx.meta.json)");
		}
		else if (spec.hasEmbeddedMetadata)
		{
			knowledge = QStringLiteral("embedded model metadata");
		}
		// with NO source at all, the strong "no manifest" warning above covers it
		if (!knowledge.isEmpty())
		{
			lines << okMark + QStringLiteral(" model knowledge: ") + knowledge;
		}
		if (spec.executionProvider == ModelSpec::ExecutionProvider::Cpu)
		{
			lines << okMark + QStringLiteral(" execution: CPU only (declared by the model)");
		}
		else if (spec.executionProvider == ModelSpec::ExecutionProvider::Cuda)
		{
			lines << okMark + QStringLiteral(" execution: CUDA family only (declared by the model), CPU fallback");
		}
		else if (spec.executionProvider == ModelSpec::ExecutionProvider::Dml)
		{
			lines << okMark + QStringLiteral(" execution: DirectML family only (declared by the model), CPU fallback");
		}
		else if (spec.executionProvider == ModelSpec::ExecutionProvider::Gpu)
		{
			lines << okMark + QStringLiteral(" execution: any GPU (CUDA, then DirectML), CPU fallback");
		}
	}

	// ---- model family (registry lookup: adapter.id, then model_name) ----
	const ModelAdapter* adapter = modelAdapterForSpec(spec);
	if (!adapter)
	{
		ok = false;
		lines << failMark + QLatin1Char(' ') + QString::fromStdString(unknownAdapterError(spec));
	}
	else
	{
		lines << okMark + QStringLiteral(" model architecture \"%1\"").arg(QLatin1String(adapter->id()));
	}

	// ---- feature requirements vs the cloud (needs no ONNX Runtime) ----
	if (adapter)
	{
		const std::string featureError = validateCloudFeatures(spec, hasColors, hasIntensity);
		if (featureError.empty())
		{
			lines << okMark + QStringLiteral(" the cloud provides the features the model needs");
		}
		else
		{
			ok = false;
			lines << failMark + QLatin1Char(' ') + QString::fromStdString(featureError);
		}
	}

	// ---- everything that is only ASSUMED (never guessed silently) ----
	// (assumptionWarnings ran inside loadModelSpec; the split-out list)
	for (const QString& assumed : assumedWarnings)
	{
		lines << warnMark + QLatin1Char(' ') + assumed.toHtmlEscaped();
	}

	// ---- runtime: session creation + the graph IO contract ----
	// The runtime is loadable when the deployment puts it next to the
	// application; when it is NOT, the report warns and skips the session
	// checks instead of failing (the manifest/family/feature checks ran).
	if (!OrtLibrary::instance().isLoaded())
	{
		std::string loadError;
		OrtLibrary::instance().load(defaultOrtLibraryCandidates(), &loadError);
	}
	if (OrtLibrary::instance().isLoaded())
	{
		OrtSession  session;
		std::string createError;
		if (!session.create(modelPath.toStdString(), {"CPUExecutionProvider"}, &createError))
		{
			ok = false;
			lines << failMark + QStringLiteral(" ONNX Runtime %1 could not load the model: %2").arg(QString::fromStdString(OrtLibrary::instance().version()), QString::fromStdString(createError).toHtmlEscaped());
		}
		else
		{
			lines << okMark + QStringLiteral(" ONNX Runtime %1 loads the model (provider: %2)").arg(QString::fromStdString(OrtLibrary::instance().version()), QString::fromStdString(session.activeProvider()));
			const IOContract io = readIOContract(session);
			fillSpecFromSession(spec, session);
			const std::string ioError = validateSpecAgainstIO(spec, io);
			if (!ioError.empty())
			{
				ok = false;
				lines << failMark + QLatin1Char(' ') + QString::fromStdString(ioError).toHtmlEscaped();
			}
			else
			{
				lines << okMark + QStringLiteral(" the graph IO contract matches the manifest");
				if (spec.staticNumPoints > 0)
				{
					lines << okMark + QStringLiteral(" the graph fixes %1 points per patch (static input axis)").arg(spec.staticNumPoints);
				}
			}
		}

		// GPU status is informational: the GPU is opportunistic and Auto mode
		// falls back to the CPU when the out-of-process probe fails
		const std::vector<std::string> runtimeProviders = availableExecutionProviders();
		QStringList                    providedNames;
		for (const std::string& provided : runtimeProviders)
		{
			providedNames << QString::fromStdString(providerDisplayName(provided));
		}
		if (!providedNames.isEmpty())
		{
			lines << okMark + QStringLiteral(" this ONNX Runtime provides: ") + providedNames.join(QStringLiteral(", "));
		}

		// the exact Auto resolution, capability only - the run-time probe can
		// still downgrade a GPU choice to the CPU (see the plugin README)
		const std::vector<std::string> prefer = autoGpuCandidates(spec);
		QString                        autoChoice;
		if (spec.executionProvider == ModelSpec::ExecutionProvider::Cpu)
		{
			autoChoice = QStringLiteral("CPU (the model declares CPU execution)");
		}
		else
		{
			for (const std::string& candidate : prefer)
			{
				if (std::find(runtimeProviders.begin(), runtimeProviders.end(), candidate) != runtimeProviders.end())
				{
					autoChoice = QString::fromStdString(providerDisplayName(candidate)) + QStringLiteral(" (probe-gated at run time; the CPU is the fallback)");
					break;
				}
			}
			if (autoChoice.isEmpty())
			{
				autoChoice = QStringLiteral("CPU (no matching GPU provider in this ONNX Runtime build)");
			}
		}
		lines << okMark + QStringLiteral(" Auto would select: ") + autoChoice;

		// explain a CUDA preference that the deployed runtime cannot satisfy -
		// but only when there is no GPU provider at all (a DirectML build
		// legitimately ships without the CUDA provider DLLs) and the model did
		// not declare CPU execution in the first place
		const bool cudaPreferred = spec.executionProvider != ModelSpec::ExecutionProvider::Cpu
		                           && std::find(prefer.begin(), prefer.end(), "CUDAExecutionProvider") != prefer.end();
		const bool cudaMissing = std::find(runtimeProviders.begin(), runtimeProviders.end(), "CUDAExecutionProvider") == runtimeProviders.end();
		const bool dmlProvided = std::find(runtimeProviders.begin(), runtimeProviders.end(), "DmlExecutionProvider") != runtimeProviders.end();
		if (cudaPreferred && cudaMissing && !dmlProvided)
		{
			const std::string cudaHint = OrtLibrary::instance().cudaProviderHint();
			if (!cudaHint.empty())
			{
				lines << warnMark + QStringLiteral(" GPU: %1").arg(QString::fromStdString(cudaHint).toHtmlEscaped());
			}
		}
	}
	else
	{
		lines << warnMark + QStringLiteral(" ONNX Runtime is not available here - the graph IO contract was NOT checked (manifest, model architecture and cloud features were checked)");
	}

	const QString headline = ok ? QStringLiteral("<b><span style=\"color: #2e7d32;\">&#10003; Model validation PASSED</span></b>")
	                            : QStringLiteral("<b><span style=\"color: #b00020;\">&#10007; Model validation FAILED</span></b>");
	report                 = headline;
	if (!lines.isEmpty())
	{
		report += QStringLiteral("<br>") + lines.join(QStringLiteral("<br>"));
	}
	return ok;
}
