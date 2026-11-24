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

#include "qSemSeg.h"

#include "ModelAdapter.h"
#include "ModelSpec.h"
#include "OnnxRuntime.h"
#include "OutputMapper.h"
#include "Preprocess.h"
#include "SemSegDialog.h"
#include "SlidingWindow.h"

// CCPluginAPI
#include <ccMainAppInterface.h>

// qCC_db
#include <ccColorScale.h>
#include <ccColorTypes.h>
#include <ccHObjectCaster.h>
#include <ccPointCloud.h>
#include <ccScalarField.h>

// Qt
#include <QAction>
#include <QColor>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QMainWindow>
#include <QProcess>
#include <QProgressDialog>
#include <QRegularExpression>
#include <QSettings>
#include <QStringList>
#include <QTimer>
#include <QtConcurrent/QtConcurrentRun>
#include <algorithm>
#include <atomic>
#include <limits>
#include <random>
#include <string>
#include <vector>

qSemSeg::qSemSeg(QObject* parent)
    : QObject(parent)
    , ccStdPluginInterface(":/CC/plugin/qSemSeg/info.json")
    , m_action(nullptr)
{
}

void qSemSeg::onNewSelection(const ccHObject::Container& selectedEntities)
{
	if (!m_action)
	{
		return;
	}

	bool hasCloud = false;
	for (ccHObject* entity : selectedEntities)
	{
		if (entity && entity->isA(CC_TYPES::POINT_CLOUD))
		{
			hasCloud = true;
			break;
		}
	}
	m_action->setEnabled(hasCloud);
}

QList<QAction*> qSemSeg::getActions()
{
	if (!m_action)
	{
		m_action = new QAction(QStringLiteral("Apply Semantic Segmentation Model"), this);
		m_action->setToolTip(QStringLiteral("Apply a pretrained ONNX semantic segmentation model (e.g. FKAConv) to the selected point cloud.\n"
		                                    "The predicted labels are added to the cloud as a new scalar field."));
		m_action->setIcon(getIcon());

		connect(m_action, &QAction::triggered, this, &qSemSeg::doAction);
	}

	return {m_action};
}

// ---------------------------------------------------------------------------
// out-of-process GPU probe (crash isolation)
// ---------------------------------------------------------------------------

namespace
{
	//! Looks for the qSemSegProbe executable next to the application / plugin
	/** Returns an empty string when not deployed (the plugin then cannot verify
	    the GPU out-of-process and will not use it). **/
	QString findProbeExecutable()
	{
		const QString appDir = QCoreApplication::applicationDirPath();
#ifdef _WIN32
		const QLatin1String suffix(".exe");
#else
		const QLatin1String suffix("");
#endif
		const QStringList candidates = {
		    appDir + QStringLiteral("/qSemSegProbe") + suffix,                                 // next to the application (dev + install tree)
		    appDir + QStringLiteral("/plugins/qSemSegProbe") + suffix,                         // next to the plugin DLL (install tree)
		    appDir + QStringLiteral("/../plugins/core/Standard/qSemSeg/qSemSegProbe") + suffix // dev build tree
		};
		for (const QString& candidate : candidates)
		{
			const QString cleanPath = QDir::cleanPath(candidate);
			if (QFileInfo::exists(cleanPath))
			{
				return cleanPath;
			}
		}
		return {};
	}

	//! Identity of the loaded runtime (version + on-disk path)
	/** Probe verdicts and the per-model GPU-failure cache are keyed by this:
	    swapping to another ONNX Runtime build re-checks the GPU even when the
	    version string is identical (e.g. the CUDA and DirectML packages of the
	    same release). **/
	QString runtimeKey()
	{
		return QString::fromStdString(OrtLibrary::instance().version()) + QLatin1Char('|')
		       + QString::fromStdString(OrtLibrary::instance().loadedPath());
	}

	//! QSettings key of one provider's cached probe verdict
	QString gpuVerdictKey(const std::string& provider)
	{
		return QStringLiteral("qSemSeg/gpuProbe_%1").arg(QString::fromStdString(provider));
	}

	//! Reads the cached probe verdict for the given provider + runtime identity
	/** Verdicts are per provider AND per ONNX Runtime identity (version +
	    loaded library path): a runtime update or a swap between the CUDA and
	    the DirectML package of the same release must re-probe, because the
	    provider DLLs are part of that deployment. **/
	bool readGpuVerdict(const std::string& provider, const QString& ortVersion, bool& ok)
	{
		QSettings     settings;
		const QString cached = settings.value(gpuVerdictKey(provider)).toString();
		const int     sep    = cached.indexOf(QLatin1Char('|'));
		if (sep <= 0)
		{
			return false;
		}
		if (cached.mid(sep + 1) != ortVersion)
		{
			return false; // the runtime changed since the last probe: verify again
		}
		ok = cached.left(sep) == QStringLiteral("ok");
		return true;
	}

	//! Stores one provider's probe verdict, keyed by the ONNX Runtime version
	void writeGpuVerdict(const std::string& provider, const QString& ortVersion, bool ok)
	{
		const QString verdict = ok ? QStringLiteral("ok|") : QStringLiteral("fail|");
		const QString value   = verdict + ortVersion;
		QSettings     settings;
		settings.setValue(gpuVerdictKey(provider), value);
	}

	//! Stable fingerprint of a model file (absolute path + size + mtime)
	/** Runtime GPU failures are cached per model file: replacing the model
	    (new size/mtime) invalidates the cached verdict automatically. **/
	QString modelFingerprint(const QString& modelPath)
	{
		const QFileInfo info(modelPath);
		const QString   key = QDir::cleanPath(info.absoluteFilePath()) + QLatin1Char('|') + QString::number(info.size())
		                    + QLatin1Char('|') + QString::number(info.lastModified().toMSecsSinceEpoch());
		return QString::fromLatin1(QCryptographicHash::hash(key.toUtf8(), QCryptographicHash::Sha1).toHex().left(16));
	}

	//! True when this provider produced NaN/Inf for this model on this runtime
	/** Keyed by provider, model fingerprint AND the loaded runtime (version +
	    path): a runtime update re-checks the GPU, a replaced model file too. A
	    driver update is NOT detected - the console names the QSettings value
	    to delete. **/
	bool readModelGpuFail(const QString& modelPath, const std::string& provider)
	{
		QSettings settings;
		return settings.value(QStringLiteral("qSemSeg/gpuModelFail_%1_%2")
		                          .arg(QString::fromStdString(provider), modelFingerprint(modelPath)))
		           .toString()
		       == runtimeKey();
	}

	//! Records that this provider produced NaN/Inf for this model
	void writeModelGpuFail(const QString& modelPath, const std::string& provider)
	{
		QSettings settings;
		settings.setValue(QStringLiteral("qSemSeg/gpuModelFail_%1_%2")
		                      .arg(QString::fromStdString(provider), modelFingerprint(modelPath)),
		                  runtimeKey());
	}

	//! Runs the qSemSegProbe executable once; sets \p retryable for failures
	//! that may be transient (a nonzero exit, but no crash and no timeout)
	bool runGpuProbeOnce(const QString& probeExe, const QString& modelPath, const std::string& provider, QString& reason, bool& retryable)
	{
		retryable = false;
		QStringList args;
		args << QString::fromStdString(OrtLibrary::instance().loadedPath())
		     << modelPath
		     << QString::fromStdString(provider);

		reason.clear();
		QProcess probe;
		probe.setProcessChannelMode(QProcess::MergedChannels);
		probe.start(probeExe, args);
		if (!probe.waitForStarted(10000))
		{
			reason = QStringLiteral("could not start %1 (%2)").arg(probeExe, probe.errorString());
			return false;
		}
		if (!probe.waitForFinished(60000))
		{
			probe.kill();
			probe.waitForFinished(5000);
			reason = QStringLiteral("the GPU probe did not finish within 60 s");
			return false;
		}

		QString output = QString::fromUtf8(probe.readAllStandardOutput());
		// keep the last few lines only (the actionable error is usually at the end)
		const QStringList lines = output.trimmed().split(QRegularExpression(QStringLiteral("[\\r\\n]+")), Qt::SkipEmptyParts);
		QString           tail;
		for (qsizetype i = std::max<qsizetype>(0, lines.size() - 4); i < lines.size(); ++i)
		{
			tail += lines[i].left(160) + QStringLiteral(" | ");
		}

		const QString display = QString::fromStdString(providerDisplayName(provider));
		if (probe.exitStatus() == QProcess::CrashExit)
		{
			reason = QStringLiteral("the %1 probe crashed (exit code %2): the GPU driver stack is unusable on this machine [%3]")
			             .arg(display)
			             .arg(probe.exitCode())
			             .arg(tail);
			return false;
		}
		if (probe.exitCode() != 0)
		{
			// a clean nonzero exit is NOT cached-proof: GPU memory pressure,
			// a busy device or a driver timeout can fail one forward pass
			retryable = true;
			reason    = QStringLiteral("the %1 probe failed (exit code %2): %3").arg(display).arg(probe.exitCode()).arg(tail);
			return false;
		}
		return true;
	}

	//! Runs the qSemSegProbe executable to verify one execution provider
	/** The probe runs in its own process: a native crash inside a broken driver
	    (which cannot be caught in-process) kills the probe only, and is reported
	    here as a failed verdict. A clean nonzero exit is retried ONCE before the
	    verdict is cached: a transient failure (GPU memory pressure, busy device)
	    must not disable the provider until the runtime changes.
	    \param provider ORT provider id to verify, e.g. "DmlExecutionProvider"
	    \param reason set to a human-readable failure reason (empty on success)
	    \return true when the provider passed the probe **/
	bool runGpuProbe(const QString& modelPath, const std::string& provider, QString& reason)
	{
		const QString probeExe = findProbeExecutable();
		if (probeExe.isEmpty())
		{
			reason = QStringLiteral("the qSemSegProbe executable could not be found next to the application or the plugin (it verifies the GPU out-of-process)");
			return false;
		}

		bool retryable = false;
		if (runGpuProbeOnce(probeExe, modelPath, provider, reason, retryable))
		{
			return true;
		}
		if (!retryable)
		{
			return false;
		}
		const QString firstReason = reason;
		if (runGpuProbeOnce(probeExe, modelPath, provider, reason, retryable))
		{
			return true;
		}
		reason = firstReason + QStringLiteral(" | retried once: ") + reason;
		return false;
	}
} // namespace

namespace
{
	//! Scalar-field name that does not collide with an existing one
	/** "SemSeg.label" -> "SemSeg.label (2)" -> ... so rerunning the plugin
	    keeps the previous result instead of silently overwriting it. **/
	std::string uniqueScalarFieldName(ccPointCloud* cloud, const std::string& baseName)
	{
		std::string name    = baseName;
		int         version = 2;
		while (cloud->getScalarFieldIndexByName(name.c_str()) >= 0)
		{
			name = baseName + " (" + std::to_string(version++) + ")";
		}
		return name;
	}
} // namespace

void qSemSeg::doAction()
{
	// m_app should have already been initialized by CC when the plugin is loaded
	if (!m_app)
	{
		assert(false);
		return;
	}

	try
	{
		// every selected point cloud (processed one after another: per-cloud
		// failures are reported and named, the remaining clouds keep running)
		std::vector<ccPointCloud*> clouds;
		for (ccHObject* entity : m_app->getSelectedEntities())
		{
			if (entity && entity->isA(CC_TYPES::POINT_CLOUD))
			{
				clouds.push_back(ccHObjectCaster::ToPointCloud(entity));
			}
		}
		if (clouds.empty())
		{
			dispToConsole(QStringLiteral("Select at least one point cloud"), ccMainAppInterface::ERR_CONSOLE_MESSAGE);
			return;
		}

		// ---- load ONNX Runtime dynamically (no build-time dependency) ----
		// full paths first (application dir, plugins dir, deployment layouts);
		// a stale library on the OS search path (e.g. an old DLL in the
		// Windows system directory) is only used as a last resort and is
		// rejected when too old for the vendored API header. The discovery
		// is shared with the "Validate model" dialog action. Loading it BEFORE
		// the dialog lets the execution choices reflect the deployed runtime.
		{
			std::string ortError;
			if (!OrtLibrary::instance().load(defaultOrtLibraryCandidates(), &ortError))
			{
				dispToConsole(QStringLiteral("ONNX Runtime could not be loaded:\n%1\n"
				                             "Deploy the ONNX Runtime shared libraries next to CloudCompare.exe (onnxruntime.dll on Windows; for a GPU also onnxruntime_providers_cuda.dll + the NVIDIA runtime DLLs, or the DirectML build's DirectML.dll).")
				                  .arg(QString::fromStdString(ortError)),
				              ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}
		}
		// the loaded runtime is the authority on which providers exist (one
		// package carries the CUDA or the DirectML family, never both)
		const std::vector<std::string> provided = availableExecutionProviders();

		// dialog parameters (semi-persistent)
		static QString s_modelPath;
		{
			// the model card describes the FIRST selected cloud's feature
			// sources (the per-cloud feature checks happen later, in the loop)
			SemSegDialog dlg(s_modelPath, clouds.front(), m_app->getMainWindow(), provided);
			if (!dlg.exec())
			{
				return;
			}
			s_modelPath = dlg.modelPath();

			if (s_modelPath.isEmpty())
			{
				dispToConsole(QStringLiteral("No ONNX model file selected"), ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}

			QString   metaWarning;
			ModelSpec spec = loadModelSpec(s_modelPath, &metaWarning);
			if (!metaWarning.isEmpty())
			{
				dispToConsole(metaWarning, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
			}
			// hard manifest problems (unsupported schema version, unknown
			// output.kind): running would silently misinterpret the model
			if (!spec.parseErrors.isEmpty())
			{
				dispToConsole(QStringLiteral("The model manifest has errors that make a run unsafe - not starting:\n- %1")
				                  .arg(spec.parseErrors.join(QStringLiteral("\n- "))),
				              ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}

			// user choices override the manifest/metadata values (and count as
			// explicit knowledge: the embedded-metadata merge must not clobber
			// them, and no "assumed" warning is due for a user decision)
			spec.voxelSize = dlg.voxelSize();
			spec.numPoints = dlg.numPoints();
			spec.manifestFields |= ModelSpec::mbVoxelSize | ModelSpec::mbNumPoints;

			SWParams params;
			params.numPoints  = spec.numPoints;
			params.patchBatch = dlg.patchBatch();
			params.seed       = dlg.seed();

			// ---- execution plan: where should the model run? ----
			// Rules (documented in the plugin README, "Execution providers"):
			// the dialog either pins one provider (CPU / CUDA / DirectML) or
			// leaves it to Auto. Auto takes the model's manifest hint
			// (execution.provider) when it names a GPU family, otherwise it
			// prefers CUDA (longest-supported) and then DirectML; a GPU
			// candidate is used only when the deployed ONNX Runtime provides
			// it AND it passes the out-of-process probe, and the CPU is the
			// final fallback.
			const SemSegDialog::ExecMode execMode   = dlg.executionMode();
			auto                         isProvided = [&provided](const std::string& provider)
			{
				return std::find(provided.begin(), provided.end(), provider) != provided.end();
			};
			QStringList executionNotes; // one line per GPU candidate that was skipped

			std::string pinnedProvider; // non-empty = the dialog pinned one provider
			if (execMode == SemSegDialog::ExecMode::CudaOnly)
			{
				pinnedProvider = "CUDAExecutionProvider";
			}
			else if (execMode == SemSegDialog::ExecMode::DmlOnly)
			{
				pinnedProvider = "DmlExecutionProvider";
			}

			std::vector<std::string> gpuCandidates;
			if (!pinnedProvider.empty())
			{
				gpuCandidates = {pinnedProvider};
			}
			else if (execMode == SemSegDialog::ExecMode::Auto)
			{
				// exact policy: ModelSpec::autoGpuCandidates (manifest hint ->
				// model family restriction, else CUDA then DirectML)
				gpuCandidates = autoGpuCandidates(spec);
			}

			std::vector<std::string> providers;
			std::string              plannedGpu;
			if (execMode == SemSegDialog::ExecMode::CpuOnly)
			{
				providers.push_back("CPUExecutionProvider");
			}
			else if (!pinnedProvider.empty() && !isProvided(pinnedProvider))
			{
				QStringList providedNames;
				for (const std::string& id : provided)
				{
					providedNames << QString::fromStdString(providerDisplayName(id));
				}
				dispToConsole(QStringLiteral("%1 only was requested, but this ONNX Runtime build does not provide it (%2). Provided: %3.\n"
				                             "Deploy an ONNX Runtime package that provides it next to CloudCompare.exe - the plugin README's 'Runtime deployment' section lists the files of the CUDA and the DirectML packages.")
				                  .arg(QString::fromStdString(providerDisplayName(pinnedProvider)),
				                       QString::fromStdString(OrtLibrary::instance().loadedPath()),
				                       providedNames.isEmpty() ? QStringLiteral("none") : providedNames.join(QStringLiteral(", "))),
				              ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}
			else if (execMode == SemSegDialog::ExecMode::Auto && spec.executionProvider == ModelSpec::ExecutionProvider::Cpu)
			{
				// the model declares CPU execution (execution.provider): an
				// export whose GPU kernels produce NaN/Inf or take minutes per
				// forward must not waste a probe + a real forward pass
				executionNotes << QStringLiteral("the model declares CPU execution (execution.provider = \"cpu\" in its manifest) - the GPU is not probed");
				providers.push_back("CPUExecutionProvider");
			}
			else
			{
				// the GPU is only used when an out-of-process probe proves that
				// it works on this machine (a broken driver would otherwise
				// crash the process - such a crash cannot be caught in-process)
				const QString ortKey = runtimeKey();
				for (const std::string& candidate : gpuCandidates)
				{
					const QString display = QString::fromStdString(providerDisplayName(candidate));
					if (!isProvided(candidate))
					{
						executionNotes << QStringLiteral("%1 skipped: this ONNX Runtime build does not provide it (%2)")
						                      .arg(display, QString::fromStdString(OrtLibrary::instance().loadedPath()));
						continue;
					}
					// the per-model NaN/Inf cache only steers AUTO: an explicit
					// "CUDA only" / "DirectML only" choice must try the provider
					// the user asked for (a failing run then errors out loudly)
					if (pinnedProvider.empty() && readModelGpuFail(s_modelPath, candidate))
					{
						// a previous run already proved this provider produces
						// NaN/Inf for this exact model file on this runtime
						executionNotes << QStringLiteral("%1 skipped: it produced NaN/Inf outputs for this model on this machine last time (cached; re-checked after an ONNX Runtime update or when the model file changes - delete the qSemSeg/gpuModelFail_* settings to force a re-check)")
						                      .arg(display);
						continue;
					}
					bool probeOk = false;
					if (readGpuVerdict(candidate, ortKey, probeOk))
					{
						if (!probeOk)
						{
							executionNotes << QStringLiteral("%1 skipped: its probe failed previously on this machine with ONNX Runtime %2 (cached; the verdict is re-checked after a runtime update)")
							                      .arg(display, QString::fromStdString(OrtLibrary::instance().version()));
							continue;
						}
					}
					else
					{
						QString probeReason;
						probeOk = runGpuProbe(s_modelPath, candidate, probeReason);
						writeGpuVerdict(candidate, ortKey, probeOk);
						if (!probeOk)
						{
							executionNotes << QStringLiteral("%1 skipped: %2").arg(display, probeReason);
							continue;
						}
					}
					plannedGpu = candidate;
					break;
				}

				if (!plannedGpu.empty())
				{
					providers.push_back(plannedGpu);
					providers.push_back("CPUExecutionProvider");
				}
				else if (!pinnedProvider.empty())
				{
					dispToConsole(QStringLiteral("%1 only was requested, but it cannot be used: %2")
					                  .arg(QString::fromStdString(providerDisplayName(pinnedProvider)), executionNotes.join(QStringLiteral("; "))),
					              ccMainAppInterface::ERR_CONSOLE_MESSAGE);
					return;
				}
				else
				{
					providers.push_back("CPUExecutionProvider");
				}
			}

			// ---- ONNX session (providers verified beforehand, CPU fallback last resort) ----
			OrtSession  session;
			std::string sessionError;
			if (!session.create(s_modelPath.toStdString(), providers, &sessionError))
			{
				dispToConsole(QStringLiteral("Could not create the ONNX session: %1").arg(QString::fromStdString(sessionError)), ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}
			if (!pinnedProvider.empty() && session.activeProvider() != pinnedProvider)
			{
				dispToConsole(QStringLiteral("%1 only was requested, but the session runs on %2: %3")
				                  .arg(QString::fromStdString(providerDisplayName(pinnedProvider)),
				                       QString::fromStdString(providerDisplayName(session.activeProvider())),
				                       QString::fromStdString(session.providerNotes())),
				              ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}
			if (!plannedGpu.empty() && session.activeProvider() != plannedGpu)
			{
				// the probe passed but ORT still could not initialize the
				// provider for this graph: say so, the run continues on the CPU
				executionNotes << QStringLiteral("the session fell back to the CPU at creation: %1")
				                      .arg(QString::fromStdString(session.providerNotes()));
			}
			fillSpecFromSession(spec, session);

			// embedded-metadata verification pass: the file-level scan (before
			// the dialog) already applied it, so this merge is idempotent and
			// only fills anything the file scan could not see
			QString embeddedInfo;
			applyEmbeddedMetadata(spec, session, &embeddedInfo); // sets spec.hasEmbeddedMetadata on success
			if (!embeddedInfo.isEmpty())
			{
				dispToConsole(embeddedInfo, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
			}
			QString knowledgeSource;
			if (spec.hasSidecarManifest && spec.hasEmbeddedMetadata)
			{
				knowledgeSource = QStringLiteral("sidecar manifest + embedded metadata");
			}
			else if (spec.hasSidecarManifest)
			{
				knowledgeSource = QStringLiteral("sidecar manifest");
			}
			else if (spec.hasEmbeddedMetadata)
			{
				knowledgeSource = QStringLiteral("embedded ONNX metadata");
			}
			else
			{
				knowledgeSource = QStringLiteral("defaults only");
			}
			dispToConsole(QStringLiteral("Model knowledge: %1").arg(knowledgeSource));

			// the session contract is authoritative over the manifest: a
			// mismatch (channels, class count, missing hierarchy inputs) is an
			// error, not a silent fallback
			std::string ioError = validateSpecAgainstIO(spec, readIOContract(session));
			if (!ioError.empty())
			{
				dispToConsole(QStringLiteral("Model/manifest mismatch: %1").arg(QString::fromStdString(ioError)), ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}
			// a static num_points axis imposes the patch size (validateSpecAgainstIO
			// recorded it in spec.staticNumPoints): a diverging requested value is
			// meaningless for this graph - say so instead of silently ignoring it
			if (spec.staticNumPoints > 0 && spec.staticNumPoints != spec.numPoints)
			{
				dispToConsole(QStringLiteral("The model's input shape imposes exactly %1 points per patch (static num_points axis) - using %1 instead of the requested %2")
				                  .arg(spec.staticNumPoints)
				                  .arg(spec.numPoints),
				              ccMainAppInterface::WRN_CONSOLE_MESSAGE);
			}

			// warm-up probe: one forward pass on synthetic zero-filled inputs
			// exercises the real graph end-to-end so ORT-level failures (driver
			// DLLs present but broken, graph/EP incompatibilities) surface as
			// catchable errors BEFORE the real run. (Native crashes cannot be
			// caught in-process; they are handled by the out-of-process probe.)
			std::string warmupError;
			if (!probeSession(session, spec, &warmupError))
			{
				dispToConsole(QStringLiteral("Warm-up inference failed: %1").arg(QString::fromStdString(warmupError)), ccMainAppInterface::ERR_CONSOLE_MESSAGE);
				return;
			}

			// auto batch size: calibrates INSIDE the worker (below) when the
			// dialog left the patch batch at "auto" (0) - on a CPU-only
			// machine the calibration forwards take seconds, and the progress
			// dialog must already be visible while they run

			dispToConsole(QStringLiteral("ONNX Runtime %1 (%2), execution provider: %3, %4 input(s), %5 classes")
			                  .arg(QString::fromStdString(OrtLibrary::instance().version()),
			                       QString::fromStdString(OrtLibrary::instance().loadedPath()),
			                       QString::fromStdString(providerDisplayName(session.activeProvider())))
			                  .arg(spec.inputNames.size())
			                  .arg(spec.numClasses));
			for (const QString& note : executionNotes)
			{
				dispToConsole(QStringLiteral("Execution: %1").arg(note), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
			}

			// ---- sliding-window inference (worker thread; UI stays responsive) ----
			// re-entrancy guard: a second run must not start while the event
			// loop below processes UI events
			static bool s_inferenceRunning = false;
			if (s_inferenceRunning)
			{
				dispToConsole(QStringLiteral("A segmentation run is already in progress"), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
				return;
			}

			// class indices (argmax) -> dataset labels via the manifest's
			// labels.values (e.g. ParisLille3D classes 0..8 are labels 1..9;
			// raw indices would collide with the ignored 'unclassified' label 0).
			// Built once per run: it depends only on the spec, not on the cloud.
			std::string labelMapWarning;
			LabelMap    labelMap = buildLabelMap(spec, &labelMapWarning);
			if (!labelMapWarning.empty())
			{
				dispToConsole(QString::fromStdString(labelMapWarning), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
			}

			QProgressDialog* progress = new QProgressDialog(QStringLiteral("Running semantic segmentation..."), QStringLiteral("Cancel"), 0, 1000, m_app->getMainWindow());
			progress->setWindowModality(Qt::WindowModal);
			progress->setMinimumDuration(500);

			// cancellation: the dialog's Cancel button flips this flag, which
			// the worker's per-batch callback consumes (the flag is touched
			// from both threads, hence atomic). Cancel stops the whole run;
			// the clouds processed before it keep their results.
			std::atomic<bool> cancelFlag{false};
			QObject::connect(progress, &QProgressDialog::canceled, [&cancelFlag]()
			                 { cancelFlag.store(true, std::memory_order_relaxed); });

			// live progress: the coverage bar can sit near 0% for a long time
			// on huge clouds, and a single forward pass on a slow provider can
			// take minutes - so the label carries a ticking elapsed clock and
			// the patch counter. The worker only publishes lock-free counters;
			// a UI-thread timer repaints the dialog once per second (the
			// dialog must only be touched from the UI thread).
			struct LiveProgress
			{
				std::atomic<int>        stage{0}; // 0 preparing, 1 starting, 2 inferring, 3 calibrating
				std::atomic<int>        patch{0};
				std::atomic<int>        value{0}; // 0..1000 (the dialog's range)
				std::atomic<qulonglong> pointCount{0};
				QElapsedTimer           clock;
				QString                 prefix; // "Cloud 1/2: " - UI thread only
			} live;
			QTimer liveTimer(progress); // parented: destroyed with the dialog
			liveTimer.setInterval(1000);
			QObject::connect(&liveTimer, &QTimer::timeout, progress, [&live, progress]()
			                 {
				const qint64  secs  = live.clock.isValid() ? live.clock.elapsed() / 1000 : 0;
				const QString clock = secs >= 3600
				                          ? QStringLiteral("%1:%2:%3").arg(secs / 3600).arg((secs / 60) % 60, 2, 10, QLatin1Char('0')).arg(secs % 60, 2, 10, QLatin1Char('0'))
				                          : QStringLiteral("%1:%2").arg(secs / 60).arg(secs % 60, 2, 10, QLatin1Char('0'));
				const int stage = live.stage.load(std::memory_order_relaxed);
				const int value = live.value.load(std::memory_order_relaxed);
				QString   text;
				if (stage == 0)
				{
					text = QStringLiteral("%1preparing %2 points... (%3 elapsed)").arg(live.prefix).arg(live.pointCount.load(std::memory_order_relaxed)).arg(clock);
				}
				else if (stage == 3)
				{
					text = QStringLiteral("%1calibrating the patch batch size... (%2 elapsed)").arg(live.prefix, clock);
				}
				else if (stage == 2)
				{
					text = QStringLiteral("%1patch %2 - coverage %3% - %4 elapsed")
					           .arg(live.prefix)
					           .arg(live.patch.load(std::memory_order_relaxed))
					           .arg(value / 10.0, 0, 'f', 1)
					           .arg(clock);
				}
				else
				{
					text = QStringLiteral("%1starting inference... (%2 elapsed)").arg(live.prefix, clock);
				}
				progress->setValue(value);
				progress->setLabelText(text); });
			liveTimer.start();

			s_inferenceRunning = true;
			struct InferenceGuard
			{
				bool& flag;
				~InferenceGuard()
				{
					flag = false;
				}
			} inferenceGuard{s_inferenceRunning};

			// every selected cloud runs through the same pipeline; the progress
			// bar spans ALL clouds (cloud i owns [i*span, (i+1)*span)), and a
			// failing cloud is reported and skipped instead of aborting the rest
			const size_t cloudCount = clouds.size();
			const double cloudSpan  = 1000.0 / static_cast<double>(cloudCount);
			size_t       okClouds   = 0;
			bool         canceled   = false;

			// per-cloud failures are NOT reported at ERR level mid-run: an ERR
			// console message pops up a MODAL QMessageBox right away
			// (ccConsole::logMessage), which fights with the modal progress
			// dialog for input and traps the user between two dialogs. Each
			// failure gets a non-blocking console warning here (the run keeps
			// going without interaction); the full error report is emitted
			// after the progress dialog is closed (end-of-run reporting below).
			QStringList cloudFailures;

			// the progress dialog is torn down at the end of this scope, even
			// when a processing step throws: a lingering modal dialog would
			// block the final error report (same trap as mid-run popups)
			{
				struct ProgressTeardown
				{
					QProgressDialog* dialog;
					QTimer*          timer;
					~ProgressTeardown()
					{
						timer->stop();
						dialog->reset();
						// the queued cancel connection references the local
						// cancelFlag: detach it before the dialog (and the
						// flag) go away
						QObject::disconnect(dialog, &QProgressDialog::canceled, nullptr, nullptr);
						dialog->deleteLater();
					}
				} progressTeardown{progress, &liveTimer};

				for (size_t cloudIndex = 0; cloudIndex < cloudCount; ++cloudIndex)
				{
					ccPointCloud* cloud         = clouds[cloudIndex];
					const QString cloudLabel    = cloud->getName().isEmpty() ? QStringLiteral("cloud #%1").arg(cloudIndex + 1) : QStringLiteral("'%1'").arg(cloud->getName());
					const QString cloudProgress = cloudCount > 1 ? QStringLiteral("Cloud %1/%2: ").arg(cloudIndex + 1).arg(cloudCount) : QString();

					// the live label names the cloud being processed (it would
					// otherwise show the stale initial text during the long
					// preprocessing steps)
					live.prefix = cloudProgress;
					live.stage.store(0, std::memory_order_relaxed);
					live.patch.store(0, std::memory_order_relaxed);
					live.value.store(0, std::memory_order_relaxed);
					live.pointCount.store(static_cast<qulonglong>(cloud->size()), std::memory_order_relaxed);
					live.clock.start();

					// ---- extract the cloud data (per cloud) ----
					const unsigned count = cloud->size();
					if (count == 0)
					{
						const QString failure = QStringLiteral("Cloud %1 is empty - skipped").arg(cloudLabel);
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}

					// the intensity source: an explicitly named 'intensity' scalar field
					// if present, else the currently displayed one
					ccScalarField::Shared intensitySF;
					{
						int intensityIdx = cloud->getScalarFieldIndexByName("intensity");
						intensitySF      = intensityIdx >= 0 ? cloud->getCCScalarField(intensityIdx) : nullptr;
						if (!intensitySF || intensitySF->size() != count)
						{
							intensitySF = cloud->getCurrentDisplayedScalarField();
							if (intensitySF && intensitySF->size() != count)
							{
								intensitySF = nullptr;
							}
						}
					}
					const bool hasColors    = cloud->hasColors();
					const bool hasIntensity = intensitySF != nullptr;

					// the model's feature requirements are semantic: fail loudly
					// instead of silently feeding zeros (per cloud: clouds differ)
					std::string featureError = validateCloudFeatures(spec, hasColors, hasIntensity);
					if (!featureError.empty())
					{
						const QString failure = QStringLiteral("Cloud %1 skipped - model/cloud mismatch: %2")
						                            .arg(cloudLabel, QString::fromStdString(featureError));
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}

					std::vector<float> xyz(static_cast<size_t>(count) * 3);
					std::vector<float> rgb;
					if (hasColors)
					{
						rgb.resize(static_cast<size_t>(count) * 3);
					}
					std::vector<float> intensity;
					if (hasIntensity)
					{
						intensity.resize(count);
					}

					for (unsigned i = 0; i < count; ++i)
					{
						const CCVector3* p = cloud->getPoint(i);
						xyz[i * 3 + 0]     = p->x;
						xyz[i * 3 + 1]     = p->y;
						xyz[i * 3 + 2]     = p->z;
						if (hasColors)
						{
							const ccColor::Rgba& c = cloud->getPointColor(i);
							rgb[i * 3 + 0]         = c.r / 255.f;
							rgb[i * 3 + 1]         = c.g / 255.f;
							rgb[i * 3 + 2]         = c.b / 255.f;
						}
						if (hasIntensity)
						{
							intensity[i] = static_cast<float>(intensitySF->getValue(i));
						}
					}

					// manifest-driven feature normalization (augment.normalize.feat:
					// feat = (feat - bias) / scale, linear so it commutes with the voxel
					// averaging of the subsampling)
					if (spec.normalizeFeat)
					{
						const double scale = spec.featScale != 0.0 ? spec.featScale : 1.0;
						for (float& v : rgb)
						{
							v = static_cast<float>((v - spec.featBias) / scale);
						}
						for (float& v : intensity)
						{
							v = static_cast<float>((v - spec.featBias) / scale);
						}
					}

					dispToConsole(QStringLiteral("%1Running semantic segmentation on %2 points (model: %3)...").arg(cloudProgress).arg(count).arg(s_modelPath));

					// ---- preprocessing (subsampling menu + projection map) ----
					// random/fps need a seeded rng: the run's seed drives them so a
					// rerun with the same seed reproduces the same subset
					SubsamplingParams subsampling;
					subsampling.type   = spec.subsamplingType;
					subsampling.target = spec.subsamplingTarget;
					std::mt19937_64 preRng(params.seed);
					subsampling.rng = &preRng;
					CloudPre pre    = preprocessCloud(xyz, rgb, intensity, {}, spec.voxelSize, spec.shiftMin, subsampling);
					if (spec.subsamplingType == ModelSpec::SubsamplingType::Voxel)
					{
						dispToConsole(QStringLiteral("Voxel subsampling: %1 -> %2 points (voxel size: %3)").arg(count).arg(pre.M).arg(spec.voxelSize));
					}
					else
					{
						QString modeName;
						switch (spec.subsamplingType)
						{
						case ModelSpec::SubsamplingType::None:
							modeName = QStringLiteral("none");
							break;
						case ModelSpec::SubsamplingType::Random:
							modeName = QStringLiteral("random (target %1)").arg(spec.subsamplingTarget);
							break;
						case ModelSpec::SubsamplingType::Fps:
							modeName = QStringLiteral("farthest point sampling (target %1)").arg(spec.subsamplingTarget);
							break;
						default:
							break;
						}
						dispToConsole(QStringLiteral("Subsampling (%1): %2 -> %3 points").arg(modeName).arg(count).arg(pre.M));
						if (spec.subsamplingType != ModelSpec::SubsamplingType::None)
						{
							dispToConsole(QStringLiteral("The model declares subset subsampling - the voxel size parameter is not used"), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						}
					}

					SegmentationResult result;
					std::string        workerError; // written by the worker only, read after it finished
					bool               nonFinite           = false;
					const int          requestedPatchBatch = params.patchBatch; // 0 = auto (a CPU fallback recalibrates)

					auto progressCallback = [&cancelFlag, &live, cloudIndex, cloudSpan](int patchIndex, float coverageFraction) -> bool
					{
						if (cancelFlag.load(std::memory_order_relaxed))
						{
							return false;
						}
						// coverage fraction (points above the 0.5 threshold) rises
						// steadily across the walk - the minimum possibility would
						// sit at the initial jitter until the very end; the value is
						// mapped into this cloud's share of the overall bar. The
						// UI-thread timer turns these counters into the visible
						// patch counter, coverage and elapsed clock.
						live.stage.store(2, std::memory_order_relaxed);
						live.patch.store(patchIndex, std::memory_order_relaxed);
						live.value.store(std::min(1000, static_cast<int>((static_cast<double>(cloudIndex) + static_cast<double>(coverageFraction)) * cloudSpan)),
						                 std::memory_order_relaxed);
						return true;
					};

					// inference with an automatic CPU retry: a GPU provider can
					// produce NaN/Inf on real data even though the all-zero warm-up
					// passed (the SphereFormer export is such a case: ~11 min per
					// forward AND NaN outputs). Auto mode therefore reruns the cloud
					// on the CPU and remembers the failing model.
					for (int attempt = 0;; ++attempt)
					{
						workerError.clear();
						nonFinite = false;
						result    = {};
						live.stage.store(1, std::memory_order_relaxed);
						live.patch.store(0, std::memory_order_relaxed);
						live.value.store(0, std::memory_order_relaxed);

						{
							QEventLoop                         loop;
							QFutureWatcher<SegmentationResult> watcher;
							QObject::connect(&watcher, &QFutureWatcher<SegmentationResult>::finished, &loop, &QEventLoop::quit);

							auto future = QtConcurrent::run([&]() -> SegmentationResult
							                                {
							try
							{
								// auto batch size: calibrate on the live session
								// when the dialog left the patch batch at "auto"
								// (0). Batch grouping never changes the labels,
								// so this is safe to auto-tune. Runs here so the
								// progress dialog stays visible and responsive.
								if (params.patchBatch == 0)
								{
									// a FIXED batch axis (e.g. SphereFormer exports
									// bake B=1) leaves nothing to calibrate: the
									// run loop clamps to the fixed size anyway
									const int fixedBatch = fixedBatchSize(spec);
									if (fixedBatch > 0)
									{
										params.patchBatch = fixedBatch;
										QMetaObject::invokeMethod(progress, [this, fixedBatch]()
										                          {
											dispToConsole(QStringLiteral("Model is exported with a fixed batch axis (%1) - one patch per forward").arg(fixedBatch)); }, Qt::QueuedConnection);
									}
									else
									{
										live.stage.store(3, std::memory_order_relaxed);
										std::string batchError;
										const int   autoBatch = autoSelectBatchSize(session, spec, &batchError);
										params.patchBatch     = autoBatch;
										live.stage.store(1, std::memory_order_relaxed);
										QMetaObject::invokeMethod(progress, [this, autoBatch, provider = session.activeProvider(), batchError]()
										                          {
											dispToConsole(QStringLiteral("Patch batch size auto-selected: %1 (calibrated on the %2 provider)").arg(autoBatch).arg(QString::fromStdString(provider)));
											if (!batchError.empty())
											{
												dispToConsole(QStringLiteral("Batch-size calibration note: %1").arg(QString::fromStdString(batchError)), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
											} }, Qt::QueuedConnection);
									}
									if (cancelFlag.load(std::memory_order_relaxed))
									{
										canceled = true;
										return {};
									}
								}
								std::mt19937_64 rng(params.seed);
								return runSlidingWindow(session, spec, pre, params, rng, progressCallback, &canceled);
							}
							catch (const NonFiniteOutputError& e)
							{
								// distinct type: Auto mode may retry on the CPU
								workerError = e.what();
								nonFinite   = true;
								return {};
							}
							catch (const std::exception& e)
							{
								workerError = e.what();
								return {};
							}
							catch (...)
							{
								workerError = "unknown non-standard exception";
								return {};
						} });
							watcher.setFuture(future);
							// UI thread: processes dialog repaints + the Cancel button
							// while the worker infers (no processEvents re-entrancy)
							loop.exec();
							result = future.result();
						}

						if (canceled || workerError.empty())
						{
							break;
						}
						if (!nonFinite)
						{
							break; // a generic failure: retrying would not help
						}
						// which provider produced the NaN/Inf? (only a GPU session
						// can be retried on the CPU; a CPU failure is final)
						const std::string failedProvider = session.activeProvider();
						if (!isGpuExecutionProvider(failedProvider))
						{
							if (attempt > 0)
							{
								workerError = "the model produces NaN/Inf outputs on both the GPU and the CPU: " + workerError;
							}
							break;
						}
						if (execMode != SemSegDialog::ExecMode::Auto)
						{
							workerError += " (" + std::string(providerDisplayName(failedProvider)) + " only was requested - switch to Auto to retry on the CPU)";
							break;
						}
						if (attempt > 0)
						{
							break;
						}

						// Auto mode: the GPU produced NaN/Inf for real data. Recreate
						// the session CPU-only and rerun this cloud; remember the
						// failing model so later runs skip the GPU attempt entirely.
						dispToConsole(QStringLiteral("Cloud %1: the %2 provider produced NaN/Inf outputs - retrying on the CPU (the GPU stays unused for this model)")
						                  .arg(cloudLabel, QString::fromStdString(providerDisplayName(failedProvider))),
						              ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						std::string cpuCreateError;
						if (!session.create(s_modelPath.toStdString(), {"CPUExecutionProvider"}, &cpuCreateError))
						{
							workerError = "the GPU produced NaN/Inf and the CPU fallback session could not be created: " + cpuCreateError;
							break;
						}
						fillSpecFromSession(spec, session); // same IO contract, provider-independent
						std::string cpuWarmupError;
						if (!probeSession(session, spec, &cpuWarmupError))
						{
							workerError = "the GPU produced NaN/Inf and the CPU fallback warm-up failed: " + cpuWarmupError;
							break;
						}
						writeModelGpuFail(s_modelPath, failedProvider);
						params.patchBatch = requestedPatchBatch;
						dispToConsole(QStringLiteral("Cloud %1: CPU fallback active - rerunning the cloud").arg(cloudLabel), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
					}

					if (canceled)
					{
						break;
					}
					if (!workerError.empty())
					{
						const QString failure = QStringLiteral("Cloud %1: inference failed (%2) - continuing with the next cloud")
						                            .arg(cloudLabel, QString::fromStdString(workerError));
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}
					if (result.labels.size() != count)
					{
						const QString failure = QStringLiteral("Cloud %1: internal error - got %2 labels for %3 points - skipped")
						                            .arg(cloudLabel)
						                            .arg(result.labels.size())
						                            .arg(count);
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}
					if (result.maxProbs.size() != count)
					{
						const QString failure = QStringLiteral("Cloud %1: internal error - got %2 confidences for %3 points - skipped")
						                            .arg(cloudLabel)
						                            .arg(result.maxProbs.size())
						                            .arg(count);
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}
					const std::vector<int>& labels = result.labels;

					std::vector<int> mappedLabels;
					try
					{
						mappedLabels = mapClassIndicesToLabels(labels, labelMap);
					}
					catch (const std::exception& e)
					{
						const QString failure = QStringLiteral("Cloud %1: label mapping failed (%2) - skipped")
						                            .arg(cloudLabel, QString::fromUtf8(e.what()));
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}

					// ---- store the predicted labels + confidence as new scalar fields ----
					// versioned names: rerunning keeps the previous result instead of
					// silently overwriting it ("SemSeg.label (2)", ...)
					const std::string sfName = uniqueScalarFieldName(cloud, "SemSeg.label");
					const int         sfIdx  = cloud->addScalarField(sfName);
					if (sfIdx < 0)
					{
						const QString failure = QStringLiteral("Cloud %1: failed to create the '%2' scalar field - skipped").arg(cloudLabel, QString::fromStdString(sfName));
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}
					ccScalarField::Shared sf = cloud->getCCScalarField(sfIdx);
					if (!sf || !sf->resizeSafe(count))
					{
						const QString failure = QStringLiteral("Cloud %1: failed to resize the '%2' scalar field - skipped").arg(cloudLabel, QString::fromStdString(sfName));
						cloudFailures << failure;
						dispToConsole(failure, ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						continue;
					}
					for (unsigned i = 0; i < count; ++i)
					{
						sf->setValue(i, static_cast<ScalarType>(mappedLabels[i]));
					}
					sf->computeMinAndMax();

					// argmax confidence: the accumulated probability of the winning
					// class (converges toward 1 as the sliding-window coverage grows)
					const std::string confName = uniqueScalarFieldName(cloud, "SemSeg.confidence");
					const int         confIdx  = cloud->addScalarField(confName);
					if (confIdx >= 0)
					{
						ccScalarField::Shared confSF = cloud->getCCScalarField(confIdx);
						if (confSF && confSF->resizeSafe(count))
						{
							for (unsigned i = 0; i < count; ++i)
							{
								confSF->setValue(i, static_cast<ScalarType>(result.maxProbs[i]));
							}
							confSF->computeMinAndMax();
						}
						else
						{
							dispToConsole(QStringLiteral("Cloud %1: could not fill the '%2' scalar field").arg(cloudLabel, QString::fromStdString(confName)), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
						}
					}
					else
					{
						dispToConsole(QStringLiteral("Cloud %1: could not create the '%2' scalar field").arg(cloudLabel, QString::fromStdString(confName)), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
					}

					cloud->setCurrentDisplayedScalarField(sfIdx);
					cloud->showSF(true);

					// per-class palette on a colorless cloud when the model provides
					// class names (deterministic colors; the original colors of a
					// colored cloud are never overwritten, CC's color toggle stays
					// available to show the palette instead of the SF)
					if (!cloud->hasColors() && !labelMap.names.empty())
					{
						cloud->resizeTheRGBTable(false);
						for (unsigned i = 0; i < count; ++i)
						{
							cloud->setPointColor(i, paletteColor(labels[i]));
						}
						dispToConsole(QStringLiteral("Colored the cloud with the per-class palette (toggle the display colors to see it)"));
					}

					// named color scale on the label scalar field when the model
					// provides class names: the ramp shows one solid band per
					// class, ticked with the dataset label + class name.
					// showNaNValuesInGrey(false) matters for the colorbar
					// legend: with the default ON, ccScalarField::getColor
					// answers a LIGHT GREY (not null) for values outside the
					// SF's display range - e.g. the tick value of a class the
					// cloud has no point of - and DrawColorRamp's
					// custom-labels fallback (which maps such a row through
					// the scale itself) never fires: those rows render grey
					// and the first classes' colors go missing from the bar.
					// The label SF carries no NaN values, so disabling the
					// grey has no other effect.
					if (ccColorScale::Shared scale = classColorScale(labelMap, spec.numClasses))
					{
						sf->showNaNValuesInGrey(false);
						sf->setColorScale(scale);
					}

					// mark the cloud's display as dirty: refreshAll() only redraws
					// windows whose refresh flag is on, and that flag is set by
					// prepareDisplayForRefresh (without it the 3D view keeps showing
					// the previously displayed scalar field)
					cloud->prepareDisplayForRefresh();

					m_app->refreshAll();
					m_app->updateUI();

					// per-class distribution table (ascending class index, named when
					// the manifest provides the names)
					{
						std::vector<size_t> classCounts(static_cast<size_t>(spec.numClasses), 0);
						for (int c : labels)
						{
							if (c >= 0 && c < spec.numClasses)
							{
								++classCounts[c];
							}
						}
						dispToConsole(QStringLiteral("Class distribution (see '%1'):").arg(QString::fromStdString(sfName)));
						for (int c = 0; c < spec.numClasses; ++c)
						{
							if (classCounts[c] == 0)
							{
								continue;
							}
							const int datasetLabel = c < static_cast<int>(labelMap.values.size()) ? labelMap.values[c] : c;
							QString   className;
							if (c < static_cast<int>(labelMap.names.size()))
							{
								className = QStringLiteral(" '%1'").arg(QString::fromStdString(labelMap.names[c]));
							}
							dispToConsole(QStringLiteral("  %1%2: %3 points (%4%)")
							                  .arg(datasetLabel)
							                  .arg(className)
							                  .arg(classCounts[c])
							                  .arg(100.0 * classCounts[c] / count, 0, 'f', 1));
						}
					}

					dispToConsole(QStringLiteral("%1Semantic segmentation done: %2 points -> '%3' (confidence: '%4')")
					                  .arg(cloudProgress)
					                  .arg(count)
					                  .arg(QString::fromStdString(sfName))
					                  .arg(QString::fromStdString(confName)));
					++okClouds;
				}
			}

			// the deferred per-cloud failure report: ERR console messages pop
			// up a modal QMessageBox each - safe only now that the progress
			// dialog is gone (nothing modal is left; every popup is closable
			// and nothing blocks the rest of the run)
			for (const QString& failure : cloudFailures)
			{
				dispToConsole(failure, ccMainAppInterface::ERR_CONSOLE_MESSAGE);
			}

			if (canceled)
			{
				dispToConsole(QStringLiteral("Semantic segmentation canceled by user - %1 of %2 cloud(s) were processed").arg(okClouds).arg(cloudCount), ccMainAppInterface::WRN_CONSOLE_MESSAGE);
				return;
			}
			if (cloudCount > 1)
			{
				dispToConsole(QStringLiteral("Semantic segmentation finished: %1 of %2 clouds processed (per-cloud failures are listed above)").arg(okClouds).arg(cloudCount));
			}
		}
	}
	catch (const std::exception& e)
	{
		dispToConsole(QStringLiteral("Semantic segmentation failed: %1").arg(QString::fromStdString(e.what())), ccMainAppInterface::ERR_CONSOLE_MESSAGE);
	}
}
