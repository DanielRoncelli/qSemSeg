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

// qSemSegProbe: out-of-process warm-up probe for the qSemSeg plugin.
//
// Runs exactly what the plugin runs (ONNX Runtime load -> session creation
// with the requested execution providers -> manifest/IO validation -> one
// warm-up inference on zero-filled inputs), but in its own process, so that
// a NATIVE crash (e.g. an access violation inside a broken NVIDIA driver's
// nvcuda64.dll) only kills the probe - never CloudCompare. Such a crash
// cannot be caught in-process, so process isolation is the only reliable
// detection.
//
// Usage:
//   qSemSegProbe <onnxruntime-library> <model.onnx> [providers]
// where [providers] is a comma-separated list of execution providers, e.g.
// "cpu", "dml" or "CUDAExecutionProvider,CPUExecutionProvider"
// (default: cpu). The FIRST provider is the one under test.
//
// Exit code 0 = the requested configuration works.
// Exit code 8 = the session did not run on the first requested provider
// (it fell back to the CPU). Any other exit code (including a native crash,
// reported by QProcess as an abnormal exit status) = the configuration is
// unusable.

#include "ModelSpec.h"
#include "OnnxRuntime.h"
#include "SlidingWindow.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
#ifdef _WIN32
	// Never let a crashing driver pop the Windows Error Reporting dialog:
	// QProcess would wait forever on it. This must happen before anything else.
	SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif

	if (argc < 3)
	{
		std::fprintf(stderr, "usage: qSemSegProbe <onnxruntime-library> <model.onnx> [providers comma-separated, default cpu]\n");
		return 3;
	}
	const std::string ortPath   = argv[1];
	const std::string modelPath = argv[2];

	std::vector<std::string> providers;
	if (argc >= 4)
	{
		std::string token;
		for (const char* p = argv[3];; ++p)
		{
			if (*p == ',' || *p == '\0')
			{
				if (token == "cpu")
				{
					token = "CPUExecutionProvider";
				}
				else if (token == "cuda")
				{
					token = "CUDAExecutionProvider";
				}
				else if (token == "dml")
				{
					token = "DmlExecutionProvider";
				}
				if (!token.empty())
				{
					providers.push_back(token);
				}
				token.clear();
				if (*p == '\0')
				{
					break;
				}
			}
			else
			{
				token += *p;
			}
		}
	}
	if (providers.empty())
	{
		providers.push_back("CPUExecutionProvider");
	}

	// 1) load the ONNX Runtime shared library
	std::string error;
	if (!OrtLibrary::instance().load({ortPath}, &error))
	{
		std::fprintf(stderr, "PROBE FAIL (runtime load): %s\n", error.c_str());
		return 4;
	}

	// 2) create the session with the requested providers
	OrtSession session;
	if (!session.create(modelPath, providers, &error))
	{
		std::fprintf(stderr, "PROBE FAIL (session): %s\n", error.c_str());
		return 5;
	}
	// the FIRST requested provider is the one under test: OrtSession::create
	// falls back to the CPU when it cannot be initialized, which for a probe
	// must count as a failure (an optimistic "OK" would make the plugin plan a
	// GPU run that silently lands on the CPU)
	if (session.activeProvider() != providers.front())
	{
		std::fprintf(stderr, "PROBE FAIL (provider): requested %s, but the session runs on %s\n%s", providers.front().c_str(), session.activeProvider().c_str(), session.providerNotes().c_str());
		return 8;
	}

	// 3) manifest (sidecar, when present) + IO validation, exactly like the plugin
	ModelSpec spec = loadModelSpec(QString::fromStdString(modelPath), nullptr);
	fillSpecFromSession(spec, session);
	std::string ioError = validateSpecAgainstIO(spec, readIOContract(session));
	if (!ioError.empty())
	{
		std::fprintf(stderr, "PROBE FAIL (model validation): %s\n", ioError.c_str());
		return 6;
	}

	// 4) one warm-up inference on the real graph
	if (!probeSession(session, spec, &error))
	{
		std::fprintf(stderr, "PROBE FAIL (inference): %s\n", error.c_str());
		return 7;
	}

	std::printf("PROBE OK: provider=%s runtime=%s model=%s\n",
	            session.activeProvider().c_str(),
	            OrtLibrary::instance().version().c_str(),
	            modelPath.c_str());
	return 0;
}
