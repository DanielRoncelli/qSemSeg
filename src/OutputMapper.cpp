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

#include "OutputMapper.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

LabelMap buildLabelMap(const ModelSpec& spec, std::string* warning)
{
	LabelMap labelMap;
	if (spec.labelValues.empty() || spec.labelValues.size() != static_cast<size_t>(spec.numClasses))
	{
		// no usable label table: keep the raw class indices (the reference
		// pipeline falls back to identity the same way, with a warning)
		labelMap.identity = true;
		labelMap.values.resize(std::max<size_t>(spec.labelValues.size(), static_cast<size_t>(spec.numClasses)));
		for (size_t i = 0; i < labelMap.values.size(); i++)
		{
			labelMap.values[i] = static_cast<int>(i);
		}
		if (warning)
		{
			*warning = "the manifest carries no labels.values table - writing raw class indices";
		}
		return labelMap;
	}
	labelMap.values = spec.labelValues;
	labelMap.names  = spec.labelNames;
	return labelMap;
}

std::vector<int> mapClassIndicesToLabels(const std::vector<int>& classIndices, const LabelMap& labelMap)
{
	std::vector<int> labels;
	labels.reserve(classIndices.size());
	for (int classIndex : classIndices)
	{
		if (classIndex < 0 || static_cast<size_t>(classIndex) >= labelMap.values.size())
		{
			throw std::runtime_error("predicted class index " + std::to_string(classIndex) + " is outside the label map (0.."
			                         + std::to_string(labelMap.values.size() - 1) + ")");
		}
		labels.push_back(labelMap.values[classIndex]);
	}
	return labels;
}

QColor paletteQColor(int index)
{
	const double hue = std::fmod(index * 0.618033988749895, 1.0);
	return QColor::fromHsvF(hue, 0.72, 0.95);
}

ccColor::Rgb paletteColor(int index)
{
	const QColor color = paletteQColor(index);
	return ccColor::Rgb(static_cast<ColorCompType>(color.red()),
	                    static_cast<ColorCompType>(color.green()),
	                    static_cast<ColorCompType>(color.blue()));
}

ccColorScale::Shared classColorScale(const LabelMap& labelMap, int numClasses)
{
	if (numClasses < 2)
	{
		return nullptr;
	}

	// dataset label per class (fallback: the raw class index); the scale
	// spans ALL classes so every point value maps inside the scale
	std::vector<std::pair<double, int>> bands; // (label value, class index)
	double                              lo = std::numeric_limits<double>::max();
	double                              hi = std::numeric_limits<double>::lowest();
	for (int c = 0; c < numClasses; ++c)
	{
		const int datasetLabel = c < static_cast<int>(labelMap.values.size()) ? labelMap.values[c] : c;
		lo                     = std::min(lo, static_cast<double>(datasetLabel));
		hi                     = std::max(hi, static_cast<double>(datasetLabel));
		bands.emplace_back(static_cast<double>(datasetLabel), c);
	}
	if (hi - lo <= 0.0)
	{
		return nullptr;
	}
	const double range = hi - lo;
	for (auto& band : bands)
	{
		band.first = (band.first - lo) / range; // relative position in [0, 1]
	}
	std::sort(bands.begin(), bands.end());
	// duplicate dataset label values (a degenerate manifest, or a
	// values table shorter than numClasses whose fallback indices
	// collide) would put two bands at the same relative position;
	// a zero-width stop interval AT POSITION 0 makes update()'s linear
	// interpolation divide by zero (alpha = 0/0 -> NaN -> UB color
	// cast) for the first ramp texel. Keep the FIRST class (lowest
	// index) for a shared label value - interior same-position pairs
	// are never selected by update()'s interval search, but the front
	// pair is (relativePos 0.0 samples it directly).
	bands.erase(std::unique(bands.begin(), bands.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b)
	                        { return a.first == b.first; }),
	            bands.end());

	// one tick per class, named when available. Classes without a name keep
	// a text-less tick: the legend renders the span of the tick set, so a
	// missing FIRST tick would start the bar at the second class's label
	// and clip the first class's band off the bottom. (std::set sorts by
	// value and silently merges duplicate label values - the first class
	// wins, consistent with the band dedupe above.)
	ccColorScale::LabelSet ticks;
	int                    namedTicks = 0;
	for (int c = 0; c < numClasses; ++c)
	{
		const int datasetLabel = c < static_cast<int>(labelMap.values.size()) ? labelMap.values[c] : c;
		QString   name;
		if (c < static_cast<int>(labelMap.names.size()))
		{
			name = QString::fromStdString(labelMap.names[c]);
		}
		if (!name.isEmpty())
		{
			++namedTicks;
		}
		ticks.insert(ccColorScale::Label(static_cast<double>(datasetLabel), name));
	}
	if (namedTicks < 2)
	{
		return nullptr;
	}

	auto scale = ccColorScale::Create(QStringLiteral("SemSeg classes"));
	scale->setAbsolute(lo, hi);

	// one solid band per class, CENTERED on its label value: class k's color
	// occupies the value interval between the midpoints to its neighbours
	// (first band: from lo to the first midpoint; last band: from the last
	// midpoint to hi). The tick value v_k therefore sits in the MIDDLE of its
	// band - half a gap away from each edge - so the steps-based sampling of
	// ccScalarField::getColor (3-arg getColorByRelativePos: two floor()
	// quantizations, up to ~1/steps + 1/MAX_STEPS relative error, i.e. ~0.005
	// at the default 256 steps) always resolves a class value - and the legend
	// row at that class's tick - to that class's EXACT palette color. This
	// includes the first and last bands: their colors sit at relative
	// positions 0.0 and around (C-2)/(C-1), both reachable (the last band
	// would be a ~0.001-wide strip at relative position 1.0 with left-anchored
	// bands, past the highest reachable texel - its color could never render).
	// Transitions between two classes sit at the midpoints; a thin
	// interpolation strip (delta) separates the two solid regions so no two
	// stops ever share a position (a zero-width stop interval AT POSITION 0
	// would make update()'s linear interpolation divide by zero - alpha 0/0
	// -> NaN -> UB color cast - for the first ramp texel).
	double minGap = 1.0;
	for (size_t i = 1; i < bands.size(); ++i)
	{
		minGap = std::min(minGap, bands[i].first - bands[i - 1].first);
	}
	const double delta = std::min(0.0002, minGap / 4.0);
	scale->insert(ccColorScaleElement(0.0, paletteQColor(bands.front().second)), false);
	for (size_t i = 0; i + 1 < bands.size(); ++i)
	{
		const double mid = (bands[i].first + bands[i + 1].first) / 2.0;
		scale->insert(ccColorScaleElement(mid - delta, paletteQColor(bands[i].second)), false);
		scale->insert(ccColorScaleElement(mid, paletteQColor(bands[i + 1].second)), false);
	}
	scale->insert(ccColorScaleElement(1.0, paletteQColor(bands.back().second)), false);
	for (const ccColorScale::Label& tick : ticks)
	{
		scale->customLabels().insert(tick);
	}

	scale->update();
	scale->setReadOnly(true);
	return scale;
}
