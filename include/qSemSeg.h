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

#include "ccStdPluginInterface.h"

//! qSemSeg plugin: semantic segmentation of the selected point cloud with a pretrained ONNX model
/** The model (e.g. FKAConv) is run with ONNX Runtime, loaded dynamically at runtime
    (CUDA execution provider on GPU-capable Windows systems, CPU fallback).
    The predicted labels are added to the cloud as a new scalar field.
**/
class qSemSeg : public QObject
    , public ccStdPluginInterface
{
	Q_OBJECT
	Q_INTERFACES(ccPluginInterface ccStdPluginInterface)

	Q_PLUGIN_METADATA(IID "cccorp.cloudcompare.plugin.qSemSeg" FILE "../info.json")

  public:
	explicit qSemSeg(QObject* parent = nullptr);
	~qSemSeg() override = default;

	// Inherited from ccStdPluginInterface
	void            onNewSelection(const ccHObject::Container& selectedEntities) override;
	QList<QAction*> getActions() override;

  private:
	//! "Apply Semantic Segmentation Model" action
	void doAction();

	QAction* m_action;
};
