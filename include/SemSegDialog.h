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

#include <QDialog>
#include <QString>
#include <string>
#include <vector>

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;
class QToolButton;
class QWidget;
class ccPointCloud;

//! Dialog to pick the ONNX model and the inference parameters
/** Zero-config flow: the user picks a model and presses Apply - the model
    card shows what the model declares (classes, patch size, voxel size,
    required features, knowledge source), the advanced parameters are
    pre-filled from the manifest / embedded metadata and live in a
    collapsible "Advanced" section. **/
class SemSegDialog : public QDialog
{
	Q_OBJECT

  public:
	//! Where the model runs. Auto verifies the GPU out-of-process first.
	enum class ExecMode
	{
		Auto     = 0, //!< manifest hint, else CUDA, else DirectML; CPU fallback
		CpuOnly  = 1, //!< never touch the GPU
		CudaOnly = 2, //!< NVIDIA CUDA only (aborts when unavailable); kept at 2 for settings compatibility
		DmlOnly  = 3  //!< DirectML only (aborts when unavailable)
	};

	//! \param availableProviders ORT provider ids the deployed runtime lists
	/** When non-empty, the pinned execution choices the runtime cannot provide
	    are disabled (with the reason in the item text/tooltip) and a status
	    line names the providers this deployment carries. **/
	explicit SemSegDialog(const QString& defaultModelPath, const ccPointCloud* cloud = nullptr, QWidget* parent = nullptr, const std::vector<std::string>& availableProviders = {});

	QString  modelPath() const;
	double   voxelSize() const;
	int      numPoints() const;
	int      patchBatch() const;
	quint64  seed() const;
	ExecMode executionMode() const;

  private:
	void browseModel();
	void refreshFromModel();
	void updateDivergenceNote();
	void validateModel();

	QLineEdit*      m_modelEdit;
	QDoubleSpinBox* m_voxelSizeSpin;
	QSpinBox*       m_numPointsSpin;
	QSpinBox*       m_patchBatchSpin;
	QLineEdit*      m_seedEdit;
	QComboBox*      m_execCombo;
	QLabel*         m_execInfo; //!< "this ONNX Runtime build provides: ..." line
	QToolButton*    m_advancedToggle;
	QWidget*        m_advancedWidget;
	QLabel*         m_cardLabel;
	QLabel*         m_divergenceNote;
	QString         m_intensityName; //!< SF the run would use (card display)

	//! the cloud the plugin was invoked on (feature checks in "Validate model")
	/** Valid for the dialog's whole lifetime: plugin actions are modal in
	    CloudCompare, so the cloud cannot change while the dialog is open. **/
	const ccPointCloud* m_cloud = nullptr;

	//! model-declared defaults for the divergence note (<0 / <=0 = unknown)
	double m_modelVoxelSize = -1.0;
	int    m_modelNumPoints = 0;

	//! model path whose values are currently pre-filled into the spin boxes
	/** QLineEdit::editingFinished also fires on focus loss without a text
	    change - without this guard such a spurious refresh would silently
	    reset the user's advanced parameters to the model defaults. **/
	QString m_loadedModelPath;

	//! the Apply action (disabled while the model manifest has hard errors)
	QPushButton* m_applyButton = nullptr;
};
