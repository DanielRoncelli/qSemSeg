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

#include <QColor>
#include <ccColorScale.h>
#include <string>
#include <vector>

//! Maps model class indices (argmax of the logits) back to dataset labels
/** The models are trained on the dataset's non-ignored labels in sorted order
    (e.g. ParisLille3D: labels 1..9 -> class indices 0..8), so the raw argmax
    must be mapped through the manifest's labels.values before it is written to
    the cloud - writing raw indices would shift every label by one and collide
    with the ignored labels (e.g. 0 = 'unclassified'). **/
struct LabelMap
{
	std::vector<int>         values;           // dataset label per class index
	std::vector<std::string> names;            // optional class names, same indexing
	bool                     identity = false; // true = no label table (raw indices)
};

//! Builds the label map from the manifest (identity + warning when absent)
LabelMap buildLabelMap(const ModelSpec& spec, std::string* warning = nullptr);

//! Applies the label map to one label per point
/** Throws when a class index is outside the map (manifest inconsistent with
    the model's num_classes). **/
std::vector<int> mapClassIndicesToLabels(const std::vector<int>& classIndices, const LabelMap& labelMap);

//! Deterministic per-class display palette (golden-ratio hue walk)
/** Fixed saturation/value, hue spaced by the golden ratio so neighbouring
    class indices are maximally distinct; no RNG, stable across runs. **/
QColor paletteQColor(int index);

//! The palette color as a 0-255 RGB triplet
ccColor::Rgb paletteColor(int index);

//! Discrete per-class color scale for the label scalar field
/** One solid band per class in the class palette color, ticked with the
    dataset label + class name from the manifest (CloudCompare renders the
    tick text of a scale's custom labels instead of the numeric value).
    Each band is CENTERED on its label value: it spans the value interval
    between the midpoints to the neighbouring classes, so the legend row at
    a class's tick - and every point carrying that class - resolves to that
    class's exact palette color despite the steps-based quantization of
    ccScalarField::getColor (including the first and last bands, whose
    colors sit at relative positions 0.0 and (C-2)/(C-1) - both reachable
    by the ramp texture; a left-anchored layout would pin the last class's
    color to a sub-texel strip at relative position 1.0 that can never
    render). Every class gets a tick - named when available, text-less
    otherwise - so the scale always spans ALL classes: the legend renders
    the span of the tick set, and a missing first tick would clip the first
    class's band off the bottom of the bar. Needs at least 2 named classes
    (the ramp shows custom labels only from 2 on); returns null otherwise
    (the default numeric ramp is kept). **/
ccColorScale::Shared classColorScale(const LabelMap& labelMap, int numClasses);
