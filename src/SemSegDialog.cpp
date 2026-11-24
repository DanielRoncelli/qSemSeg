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

#include "SemSegDialog.h"

#include "ModelSpec.h"
#include "OnnxRuntime.h"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStringList>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <ccPointCloud.h>
#include <ccScalarField.h>

SemSegDialog::SemSegDialog(const QString& defaultModelPath, const ccPointCloud* cloud, QWidget* parent, const std::vector<std::string>& availableProviders)
    : QDialog(parent)
    , m_modelEdit(new QLineEdit(defaultModelPath, this))
    , m_voxelSizeSpin(new QDoubleSpinBox(this))
    , m_numPointsSpin(new QSpinBox(this))
    , m_patchBatchSpin(new QSpinBox(this))
    , m_seedEdit(new QLineEdit(QStringLiteral("42"), this))
    , m_execCombo(new QComboBox(this))
    , m_execInfo(new QLabel(this))
    , m_advancedToggle(new QToolButton(this))
    , m_advancedWidget(new QWidget(this))
    , m_cardLabel(new QLabel(this))
    , m_divergenceNote(new QLabel(this))
    , m_cloud(cloud)
{
	setWindowTitle(QStringLiteral("Apply Semantic Segmentation Model"));

	m_voxelSizeSpin->setRange(0.0, 1000.0);
	m_voxelSizeSpin->setDecimals(4);
	m_voxelSizeSpin->setSpecialValueText(QStringLiteral("none (no subsampling)"));
	m_voxelSizeSpin->setValue(0.08);

	m_numPointsSpin->setRange(16, 10000000);
	m_numPointsSpin->setValue(25536);

	m_patchBatchSpin->setRange(0, 64);
	m_patchBatchSpin->setValue(0);
	m_patchBatchSpin->setSpecialValueText(QStringLiteral("auto (recommended)"));
	m_patchBatchSpin->setToolTip(QStringLiteral("0 = auto: measure forward passes for increasing batch sizes at run start and keep the fastest per-patch size. The batch size only groups patches into one forward pass - it does not change the labels."));

	// execution control (EP jargon stays out of the labels): Auto resolves the
	// provider from the model manifest + the deployed runtime + an
	// out-of-process probe, and falls back to the CPU when the GPU fails
	auto* execModel = new QStandardItemModel(this);
	m_execCombo->setModel(execModel);
	m_execCombo->addItem(QStringLiteral("Auto (recommended) - best usable provider"));
	m_execCombo->addItem(QStringLiteral("CPU only"));
	m_execCombo->addItem(QStringLiteral("CUDA only - NVIDIA GPU, abort if unavailable"));
	m_execCombo->addItem(QStringLiteral("DirectML only - any DX12 GPU, abort if unavailable"));
	m_execCombo->setToolTip(QStringLiteral("Auto picks the first usable provider in this order:\n"
	                                       "1. the model manifest's execution.provider (cpu / cuda / dml / gpu),\n"
	                                       "2. CUDA, then DirectML when the manifest does not pin a GPU family,\n"
	                                       "3. the CPU as the final fallback.\n"
	                                       "A GPU candidate is used only when the deployed ONNX Runtime provides it AND a small out-of-process probe (session + one forward pass on this model) succeeds on this machine; the verdict is cached per runtime version.\n"
	                                       "CPU only: never touch the GPU.\n"
	                                       "CUDA only / DirectML only: abort with an error when that provider is unavailable or fails the probe - never silently run on the CPU."));

	// the DEPLOYED runtime decides which pinned choices can work: one
	// onnxruntime.dll carries either the CUDA or the DirectML package, never
	// both. Disable the choices it cannot provide (instead of letting the run
	// abort after Apply) and name what this deployment actually carries.
	m_execInfo->setStyleSheet(QStringLiteral("color: #666;"));
	m_execInfo->setWordWrap(true);
	if (!availableProviders.empty())
	{
		auto isProvided = [&availableProviders](const char* id)
		{
			return std::find(availableProviders.begin(), availableProviders.end(), id) != availableProviders.end();
		};

		QStringList providedNames;
		for (const char* id : {"CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"})
		{
			if (isProvided(id))
			{
				providedNames << QString::fromStdString(providerDisplayName(id));
			}
		}
		m_execInfo->setText(QStringLiteral("This ONNX Runtime build provides: %1.")
		                        .arg(providedNames.isEmpty() ? QStringLiteral("none of the offered providers") : providedNames.join(QStringLiteral(", "))));

		// QComboBox items are QStandardItems here, so a choice the runtime
		// cannot provide is disabled instead of dead-ending after Apply
		auto disableChoice = [execModel](int index, const QString& reason)
		{
			QStandardItem* item = execModel->item(index);
			if (!item)
			{
				return;
			}
			item->setEnabled(false);
			item->setText(item->text() + QStringLiteral(" - not in this ONNX Runtime build"));
			item->setToolTip(reason);
		};
		if (!isProvided("CUDAExecutionProvider"))
		{
			disableChoice(2, QStringLiteral("This ONNX Runtime build does not include the CUDA execution provider. Deploy the ONNX Runtime GPU (CUDA) package next to CloudCompare.exe to enable it (see the plugin README, 'Runtime deployment')."));
		}
		if (!isProvided("DmlExecutionProvider"))
		{
			disableChoice(3, QStringLiteral("This ONNX Runtime build does not include the DirectML execution provider. Deploy the DirectML package (onnxruntime.dll + onnxruntime_providers_shared.dll + DirectML.dll) next to CloudCompare.exe to enable it (see the plugin README, 'DirectML (any DX12 GPU, Windows)')."));
		}
	}
	else
	{
		m_execInfo->hide(); // runtime not known (e.g. an early failure): offer all choices
	}

	// restore the last choice (persisted on Apply); a saved pinned choice that
	// this runtime cannot provide falls back to Auto instead of dead-ending
	{
		QSettings settings;
		bool      ok   = false;
		const int mode = settings.value(QStringLiteral("qSemSeg/executionMode")).toInt(&ok);
		if (ok && mode >= 0 && mode <= 3)
		{
			const QStandardItem* item   = execModel->item(mode);
			const bool           usable = item && item->isEnabled();
			m_execCombo->setCurrentIndex(usable ? mode : 0);
		}
	}

	// ---- model card (read-only view of what the model declares) ----
	m_cardLabel->setTextFormat(Qt::RichText);
	m_cardLabel->setWordWrap(true);
	m_cardLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);

	// ---- advanced section (collapsible; pre-filled from the model) ----
	auto* advancedLayout = new QFormLayout(m_advancedWidget);
	advancedLayout->addRow(QStringLiteral("Voxel size (subsampling):"), m_voxelSizeSpin);
	advancedLayout->addRow(QStringLiteral("Points per patch:"), m_numPointsSpin);
	advancedLayout->addRow(QStringLiteral("Patch batch size:"), m_patchBatchSpin);
	advancedLayout->addRow(QStringLiteral("Random seed:"), m_seedEdit);
	m_advancedWidget->setVisible(false);

	m_advancedToggle->setText(QStringLiteral("Advanced parameters"));
	m_advancedToggle->setCheckable(true);
	m_advancedToggle->setArrowType(Qt::RightArrow);
	m_advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
	m_advancedToggle->setStyleSheet(QStringLiteral("QToolButton { border: none; font-weight: bold; }"));
	connect(m_advancedToggle, &QToolButton::toggled, this, [this](bool checked)
	        {
		m_advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
		m_advancedWidget->setVisible(checked); });

	m_divergenceNote->setStyleSheet(QStringLiteral("color: #b36b00;"));
	m_divergenceNote->setWordWrap(true);

	// diverging from the model defaults is an inline note (never an error:
	// the user decides), the same at run time
	connect(m_voxelSizeSpin, &QDoubleSpinBox::valueChanged, this, [this](double)
	        { updateDivergenceNote(); });
	connect(m_numPointsSpin, &QSpinBox::valueChanged, this, [this](int)
	        { updateDivergenceNote(); });

	auto* browseButton = new QPushButton(QStringLiteral("Browse..."), this);
	connect(browseButton, &QPushButton::clicked, this, &SemSegDialog::browseModel);
	// editingFinished, not textChanged: refreshFromModel re-reads the model
	// file (tens of MB), which must not run per keystroke
	connect(m_modelEdit, &QLineEdit::editingFinished, this, [this]()
	        { refreshFromModel(); });

	auto* modelLayout = new QHBoxLayout();
	modelLayout->addWidget(m_modelEdit);
	modelLayout->addWidget(browseButton);

	auto* topFormLayout = new QFormLayout();
	topFormLayout->addRow(QStringLiteral("ONNX model file:"), modelLayout);
	topFormLayout->addRow(QStringLiteral("Execution:"), m_execCombo);
	topFormLayout->addRow(m_execInfo);

	auto* cardGroup  = new QGroupBox(QStringLiteral("Model"), this);
	auto* cardLayout = new QVBoxLayout(cardGroup);
	cardLayout->addWidget(m_cardLabel);

	auto* okButton     = new QPushButton(QStringLiteral("Apply"), this);
	auto* cancelButton = new QPushButton(QStringLiteral("Cancel"), this);
	okButton->setDefault(true);
	m_applyButton = okButton;
	connect(okButton, &QPushButton::clicked, this, [this]()
	        {
		QSettings settings;
		settings.setValue(QStringLiteral("qSemSeg/executionMode"), m_execCombo->currentIndex());
		accept(); });
	connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);

	auto* buttonLayout   = new QHBoxLayout();
	auto* validateButton = new QPushButton(QStringLiteral("Validate model"), this);
	validateButton->setToolTip(QStringLiteral("Runs every compatibility check for the selected model (manifest, model architecture, "
	                                          "graph IO contract, cloud features) without inferring, and shows a pass/fail report."));
	connect(validateButton, &QPushButton::clicked, this, &SemSegDialog::validateModel);
	buttonLayout->addWidget(validateButton);
	buttonLayout->addStretch();
	buttonLayout->addWidget(okButton);
	buttonLayout->addWidget(cancelButton);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->addLayout(topFormLayout);
	mainLayout->addWidget(cardGroup);
	mainLayout->addWidget(m_advancedToggle);
	mainLayout->addWidget(m_advancedWidget);
	mainLayout->addWidget(m_divergenceNote);
	mainLayout->addLayout(buttonLayout);

	// the intensity SF the run will use (explicit 'intensity' field, else the
	// displayed one) - shown in the card because the model may require it
	QString intensityName = QStringLiteral("none found");
	if (cloud)
	{
		int intensityIdx = cloud->getScalarFieldIndexByName("intensity");
		if (intensityIdx < 0)
		{
			const ccScalarField::Shared displayed = cloud->getCurrentDisplayedScalarField();
			if (displayed && displayed->size() == cloud->size())
			{
				intensityIdx = cloud->getScalarFieldIndexByName(displayed->getName());
			}
		}
		if (intensityIdx >= 0 && cloud->getCCScalarField(intensityIdx))
		{
			intensityName = QStringLiteral("\"%1\"").arg(QString::fromStdString(cloud->getCCScalarField(intensityIdx)->getName()));
		}
	}
	m_intensityName = intensityName;

	refreshFromModel();
}

void SemSegDialog::browseModel()
{
	const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Select ONNX model"), m_modelEdit->text(), QStringLiteral("ONNX models (*.onnx)"));
	if (!path.isEmpty())
	{
		m_modelEdit->setText(path);
		refreshFromModel();
	}
}

void SemSegDialog::refreshFromModel()
{
	const QString modelPath = m_modelEdit->text().trimmed();

	if (modelPath.isEmpty())
	{
		m_loadedModelPath.clear();
		m_modelVoxelSize = -1.0;
		m_modelNumPoints = 0;
		updateDivergenceNote();
		m_cardLabel->setText(QStringLiteral("No model selected - pick an .onnx file."));
		if (m_applyButton)
		{
			m_applyButton->setEnabled(true);
		}
		return;
	}

	// knowledge from the sidecar manifest and the model's own embedded
	// metadata (loadModelSpec merges both; precedence sidecar > embedded)
	QString   warning;
	ModelSpec spec = loadModelSpec(modelPath, &warning);

	m_modelVoxelSize = (spec.manifestFields & ModelSpec::mbVoxelSize) ? spec.voxelSize : -1.0;
	m_modelNumPoints = (spec.manifestFields & ModelSpec::mbNumPoints) ? spec.numPoints : 0;

	// pre-fill the advanced parameters with the model-declared values ONLY
	// when the model actually changed: editingFinished also fires on focus
	// loss without a text change, and a spurious refresh must not reset the
	// parameters the user already adjusted for the current model
	const bool pathChanged = modelPath != m_loadedModelPath;
	m_loadedModelPath      = modelPath;
	if (pathChanged)
	{
		if (m_modelVoxelSize >= 0.0)
		{
			m_voxelSizeSpin->setValue(spec.voxelSize);
		}
		if (m_modelNumPoints > 0)
		{
			m_numPointsSpin->setValue(spec.numPoints);
		}
	}
	updateDivergenceNote();

	// ---- model card ----
	QString knowledgeSource;
	if (spec.hasSidecarManifest && spec.hasEmbeddedMetadata)
	{
		knowledgeSource = QStringLiteral("sidecar manifest + embedded model metadata");
	}
	else if (spec.hasSidecarManifest)
	{
		knowledgeSource = QStringLiteral("sidecar manifest (<model>.onnx.meta.json)");
	}
	else if (spec.hasEmbeddedMetadata)
	{
		knowledgeSource = QStringLiteral("embedded model metadata");
	}
	else
	{
		knowledgeSource = QStringLiteral("defaults only (not specified by the model)");
	}

	QString classNames;
	if (!spec.labelNames.empty())
	{
		QStringList names;
		for (const std::string& name : spec.labelNames)
		{
			names << QString::fromStdString(name);
		}
		classNames = names.join(QStringLiteral(", "));
	}
	else if (!spec.labelValues.empty())
	{
		QStringList values;
		for (int value : spec.labelValues)
		{
			values << QString::number(value);
		}
		classNames = QStringLiteral("dataset labels ") + values.join(QStringLiteral(", ")) + QStringLiteral(" (no names)");
	}
	else
	{
		classNames = QStringLiteral("not specified (raw class indices 0..%1 will be written)").arg(spec.numClasses - 1);
	}

	QString featureColumns;
	if (spec.featureColumns.empty())
	{
		featureColumns = QStringLiteral("xyz only");
	}
	else
	{
		QStringList columns;
		for (const std::string& column : spec.featureColumns)
		{
			columns << QString::fromStdString(column);
		}
		featureColumns = QStringLiteral("xyz + ") + columns.join(QStringLiteral(", "));
	}

	QString ortStatus;
	if (!OrtLibrary::instance().version().empty())
	{
		ortStatus = QStringLiteral("ONNX Runtime %1 (loaded)").arg(QString::fromStdString(OrtLibrary::instance().version()));
	}
	else
	{
		ortStatus = QStringLiteral("checked at run start");
	}

	QString card = QStringLiteral("<b>%1</b> (%2)").arg(spec.modelName.toHtmlEscaped(), QFileInfo(modelPath).fileName().toHtmlEscaped());
	card += QStringLiteral("<br>Classes (%1): %2").arg(spec.numClasses).arg(classNames.toHtmlEscaped());
	// the card reports what the MODEL declares; the user's effective values
	// live in the Advanced fields (divergences are flagged by the note below)
	card += QStringLiteral("<br>Patch size: %1 points%2")
	            .arg(m_modelNumPoints > 0 ? m_modelNumPoints : m_numPointsSpin->value())
	            .arg(m_modelNumPoints > 0 ? QStringLiteral(" (model default)") : QString());
	// the subsampling menu: for subset modes the voxel spin is not used
	if (spec.subsamplingType == ModelSpec::SubsamplingType::Random)
	{
		card += QStringLiteral("<br>Subsampling: random subset of %1 points").arg(spec.subsamplingTarget);
	}
	else if (spec.subsamplingType == ModelSpec::SubsamplingType::Fps)
	{
		card += QStringLiteral("<br>Subsampling: farthest point sampling, %1 points").arg(spec.subsamplingTarget);
	}
	else if (spec.subsamplingType == ModelSpec::SubsamplingType::None)
	{
		card += QStringLiteral("<br>Subsampling: none (full cloud)");
	}
	else
	{
		card += QStringLiteral("<br>Voxel size: %1%2")
		            .arg(QString::number(m_modelVoxelSize >= 0.0 ? m_modelVoxelSize : m_voxelSizeSpin->value(), 'g', 6))
		            .arg(m_modelVoxelSize >= 0.0 ? QStringLiteral(" (model default)") : QString());
	}
	card += QStringLiteral("<br>Features: %1 (intensity SF: %2)").arg(featureColumns.toHtmlEscaped(), m_intensityName.toHtmlEscaped());
	if (spec.executionProvider == ModelSpec::ExecutionProvider::Cpu)
	{
		card += QStringLiteral("<br>Execution: CPU only (declared by the model)");
	}
	else if (spec.executionProvider == ModelSpec::ExecutionProvider::Cuda)
	{
		card += QStringLiteral("<br>Execution: CUDA family only, CPU fallback (declared by the model)");
	}
	else if (spec.executionProvider == ModelSpec::ExecutionProvider::Dml)
	{
		card += QStringLiteral("<br>Execution: DirectML family only, CPU fallback (declared by the model)");
	}
	else if (spec.executionProvider == ModelSpec::ExecutionProvider::Gpu)
	{
		card += QStringLiteral("<br>Execution: any GPU (CUDA, then DirectML), CPU fallback");
	}
	card += QStringLiteral("<br>Knowledge: %1").arg(knowledgeSource.toHtmlEscaped());
	card += QStringLiteral("<br>%1").arg(ortStatus.toHtmlEscaped());
	// hard manifest problems (unsupported schema version, unknown output.kind):
	// the run would misinterpret the model, so Apply stays disabled
	if (!spec.parseErrors.isEmpty())
	{
		for (const QString& e : spec.parseErrors)
		{
			card += QStringLiteral("<br><span style=\"color: #b00020;\"><b>%1</b></span>").arg(e.toHtmlEscaped());
		}
	}
	if (!warning.isEmpty())
	{
		card += QStringLiteral("<br><span style=\"color: #b00020;\">%1</span>").arg(warning.toHtmlEscaped());
	}
	m_cardLabel->setText(card);
	if (m_applyButton)
	{
		m_applyButton->setEnabled(spec.parseErrors.isEmpty());
	}
}

void SemSegDialog::updateDivergenceNote()
{
	QStringList diffs;
	if (m_modelVoxelSize >= 0.0 && m_voxelSizeSpin->value() != m_modelVoxelSize)
	{
		diffs << QStringLiteral("voxel size %1 (model default: %2)").arg(QString::number(m_voxelSizeSpin->value(), 'g', 6), QString::number(m_modelVoxelSize, 'g', 6));
	}
	if (m_modelNumPoints > 0 && m_numPointsSpin->value() != m_modelNumPoints)
	{
		QString diff = QStringLiteral("points per patch %1 (model default: %2)").arg(m_numPointsSpin->value()).arg(m_modelNumPoints);
		if (m_numPointsSpin->value() > m_modelNumPoints)
		{
			// the model's per-patch working buffers grow with this value (x
			// the batch size), so raising it can exceed the available memory
			diff += QStringLiteral(" - larger values sharply increase the memory needed per forward pass");
		}
		diffs << diff;
	}
	m_divergenceNote->setText(diffs.isEmpty() ? QString() : QStringLiteral("Note: differs from the model defaults - %1").arg(diffs.join(QStringLiteral("; "))));
}

void SemSegDialog::validateModel()
{
	const QString modelPath = m_modelEdit->text().trimmed();
	if (modelPath.isEmpty())
	{
		QMessageBox::warning(this, QStringLiteral("Validate model"), QStringLiteral("No model selected - pick an .onnx file first."));
		return;
	}

	// the same feature sources the run will use (resolved once in the
	// constructor: the cloud is fixed for the dialog's lifetime)
	const bool hasColors    = m_cloud != nullptr && m_cloud->hasColors();
	const bool hasIntensity = m_intensityName != QStringLiteral("none found");

	QString    report;
	const bool ok = buildModelValidationReport(modelPath, hasColors, hasIntensity, report);

	QMessageBox box(this);
	box.setWindowTitle(QStringLiteral("Validate model"));
	box.setIcon(ok ? QMessageBox::Information : QMessageBox::Warning);
	box.setTextFormat(Qt::RichText);
	box.setText(report);
	box.exec();
}

QString SemSegDialog::modelPath() const
{
	return m_modelEdit->text().trimmed();
}

double SemSegDialog::voxelSize() const
{
	return m_voxelSizeSpin->value();
}

int SemSegDialog::numPoints() const
{
	return m_numPointsSpin->value();
}

int SemSegDialog::patchBatch() const
{
	return m_patchBatchSpin->value();
}

quint64 SemSegDialog::seed() const
{
	bool          ok   = false;
	const quint64 seed = m_seedEdit->text().trimmed().toULongLong(&ok);
	return ok ? seed : 42; // garbage/cleared field: keep the documented default
}

SemSegDialog::ExecMode SemSegDialog::executionMode() const
{
	switch (m_execCombo->currentIndex())
	{
	case 1:
		return ExecMode::CpuOnly;
	case 2:
		return ExecMode::CudaOnly;
	case 3:
		return ExecMode::DmlOnly;
	case 0:
	default:
		return ExecMode::Auto;
	}
}
