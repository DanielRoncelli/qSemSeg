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

// Vendored ONNX Runtime C API (MIT license, see third_party/onnxruntime_c_api.h)
#include "onnxruntime_c_api.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

//! Release traits for the ONNX Runtime C API objects
template <typename T>
struct OrtObjectTraits;

template <>
struct OrtObjectTraits<OrtEnv>
{
	static void release(const OrtApi* api, OrtEnv* p)
	{
		api->ReleaseEnv(p);
	}
};
template <>
struct OrtObjectTraits<OrtSessionOptions>
{
	static void release(const OrtApi* api, OrtSessionOptions* p)
	{
		api->ReleaseSessionOptions(p);
	}
};
template <>
struct OrtObjectTraits<OrtSession>
{
	static void release(const OrtApi* api, OrtSession* p)
	{
		api->ReleaseSession(p);
	}
};
template <>
struct OrtObjectTraits<OrtMemoryInfo>
{
	static void release(const OrtApi* api, OrtMemoryInfo* p)
	{
		api->ReleaseMemoryInfo(p);
	}
};
template <>
struct OrtObjectTraits<OrtValue>
{
	static void release(const OrtApi* api, OrtValue* p)
	{
		api->ReleaseValue(p);
	}
};
template <>
struct OrtObjectTraits<OrtTypeInfo>
{
	static void release(const OrtApi* api, OrtTypeInfo* p)
	{
		api->ReleaseTypeInfo(p);
	}
};
template <>
struct OrtObjectTraits<OrtTensorTypeAndShapeInfo>
{
	static void release(const OrtApi* api, OrtTensorTypeAndShapeInfo* p)
	{
		api->ReleaseTensorTypeAndShapeInfo(p);
	}
};
template <>
struct OrtObjectTraits<OrtStatus>
{
	static void release(const OrtApi* api, OrtStatus* p)
	{
		api->ReleaseStatus(p);
	}
};
template <>
struct OrtObjectTraits<OrtModelMetadata>
{
	static void release(const OrtApi* api, OrtModelMetadata* p)
	{
		api->ReleaseModelMetadata(p);
	}
};

//! Deleter calling the right OrtApi::Release* function
template <typename T>
struct OrtDeleter
{
	const OrtApi* api = nullptr;
	void          operator()(T* p) const
	{
		if (p && api)
		{
			OrtObjectTraits<T>::release(api, p);
		}
	}
};

//! RAII pointer for ONNX Runtime C API objects
template <typename T>
using OrtPtr = std::unique_ptr<T, OrtDeleter<T>>;

template <typename T>
OrtPtr<T> makeOrtPtr(const OrtApi* api, T* p)
{
	return OrtPtr<T>(p, OrtDeleter<T>{api});
}

//! Actionable hint appended to a run error that is a memory-allocation failure
/** ONNX Runtime reports e.g. "Failed to allocate memory for requested buffer
    of size N" when a node's working buffer does not fit. With FKAConv-style
    models those buffers grow with (patch points x batch size), so the usual
    fix is a smaller patch size (the model's trained value) and/or batch
    size 1. Returns an empty string for other errors. **/
std::string allocationFailureHint(const std::string& ortMessage);

//! Full-path candidates for discovering the ONNX Runtime shared library
/** Application dir, plugins dir and the deployment layouts, in that order;
    OrtLibrary::load tries these first and falls back to the OS search path
    (rejected when too old for the vendored API header). Shared by the plugin
    run and the "Validate model" dialog action. Requires a QCoreApplication
    (it uses applicationDirPath). **/
std::vector<std::string> defaultOrtLibraryCandidates();

//! Execution providers the loaded runtime can create sessions with
/** ORT provider ids, e.g. "CPUExecutionProvider", "CUDAExecutionProvider",
    "DmlExecutionProvider". Empty when no runtime is loaded. Official ORT
    builds carry either the CUDA or the DirectML provider (never both), so
    this is the authoritative list for the Auto-selection logic: an EP that
    is not listed cannot be used no matter what the dialog or manifest says.
    A provider whose shared library is missing is not listed either. **/
std::vector<std::string> availableExecutionProviders();

//! True for the GPU providers the plugin knows (CUDA, DirectML)
bool isGpuExecutionProvider(const std::string& provider);

//! Short human-readable name for an ORT provider id
/** "CPU", "CUDA (NVIDIA GPU)", "DirectML (any DX12 GPU)"; unknown ids are
    returned unchanged. **/
std::string providerDisplayName(const std::string& provider);

//! Loads the ONNX Runtime shared library dynamically at runtime
/** The plugin has no build-time dependency on ONNX Runtime: the library
    (onnxruntime.dll on Windows, libonnxruntime.so on Linux, ...) is loaded
    with LoadLibrary/dlopen and its C API is retrieved with OrtGetApiBase.
**/
class OrtLibrary
{
  public:
	static OrtLibrary& instance();

	//! Tries to load the ONNX Runtime shared library
	/** \param explicitPaths full paths to try first (e.g. next to the application executable)
	    \param errorMessage set to the failure reason if load() returns false
	    \return true if the library was loaded and its API retrieved
	**/
	bool load(const std::vector<std::string>& explicitPaths = {}, std::string* errorMessage = nullptr);

	bool isLoaded() const
	{
		return m_api != nullptr;
	}

	//! ONNX Runtime C API (nullptr if not loaded)
	const OrtApi* api() const
	{
		return m_api;
	}

	//! ONNX Runtime version string (empty if not loaded)
	std::string version() const;

	//! Full path of the loaded ONNX Runtime shared library (empty if not loaded)
	/** Resolved from the loaded module (GetModuleFileName / dladdr), not from
	    the candidate name, so it is correct even when the OS search path found
	    the library (e.g. a stale DLL shadowing the deployed one). **/
	const std::string& loadedPath() const
	{
		return m_loadedPath;
	}

	//! Diagnostics of the last load() attempt: every candidate tried and why it was rejected
	const std::string& loadReport() const
	{
		return m_loadReport;
	}

	//! Non-empty reason when the CUDA execution provider DLLs are missing next to the loaded runtime
	/** Empty when the CUDA provider files are present (or the library is not
	    loaded); GPU support may still fail for other reasons (broken driver,
	    old NVIDIA runtime), in which case session creation / warm-up reports it. **/
	std::string cudaProviderHint() const;

  private:
	OrtLibrary() = default;
	~OrtLibrary();
	OrtLibrary(const OrtLibrary&)            = delete;
	OrtLibrary& operator=(const OrtLibrary&) = delete;

	void*             m_module  = nullptr; // HMODULE on Windows, dlopen handle elsewhere
	const OrtApiBase* m_apiBase = nullptr;
	const OrtApi*     m_api     = nullptr;
	std::string       m_loadedPath;
	std::string       m_loadReport;
};

//! ONNX model session (wraps OrtSession), with verified execution-provider fallback
class OrtSession
{
  public:
	//! Creates a session for the given model file
	/** Providers are tried in the requested order, but a provider is only accepted if
	    session creation with it actually succeeds (ORT would otherwise silently fall
	    back to CPU, e.g. when the NVIDIA runtime DLLs are missing). If no requested
	    provider works, a default (CPU) session is attempted as a last resort.
	    \param modelPath path to the .onnx file
	    \param requestedProviders e.g. {"CUDAExecutionProvider", "CPUExecutionProvider"}
	    \param errorMessage set to the failure reason if create() returns false
	**/
	bool create(const std::string& modelPath, const std::vector<std::string>& requestedProviders, std::string* errorMessage = nullptr);

	bool isValid() const
	{
		return m_session != nullptr;
	}

	//! Provider actually used by the session (e.g. "CUDAExecutionProvider")
	const std::string& activeProvider() const
	{
		return m_activeProvider;
	}

	//! Why requested providers were skipped or failed during the last create()
	/** One line per provider, in request order, e.g.
	    "DmlExecutionProvider: not provided by this ONNX Runtime build".
	    Empty when every requested provider was tried without a recorded
	    problem. Lets the plugin explain an Auto fallback to the CPU. **/
	const std::string& providerNotes() const
	{
		return m_providerNotes;
	}

	//! Underlying ONNX Runtime C API (valid once create() succeeded)
	const OrtApi* api() const
	{
		return m_api;
	}

	std::vector<std::string>               inputNames;
	std::vector<std::string>               outputNames;
	std::vector<ONNXTensorElementDataType> inputTypes;
	//! Template shape per input: -1 = dynamic batch, -2 = dynamic num_points
	std::vector<std::vector<int64_t>>      inputShapes;
	std::vector<ONNXTensorElementDataType> outputTypes;
	//! Template shape per output: -1 = dynamic batch, -2 = dynamic num_points
	std::vector<std::vector<int64_t>> outputShapes;

	//! Creates a float32 tensor wrapping the given data (data must outlive the session run)
	OrtPtr<OrtValue> makeTensorF32(const std::vector<int64_t>& shape, const std::vector<float>& data, std::string* errorMessage = nullptr) const;
	//! Creates an int64 tensor wrapping the given data (data must outlive the session run)
	OrtPtr<OrtValue> makeTensorI64(const std::vector<int64_t>& shape, const std::vector<int64_t>& data, std::string* errorMessage = nullptr) const;
	//! Creates an int32 tensor wrapping the given data (data must outlive the session run)
	OrtPtr<OrtValue> makeTensorI32(const std::vector<int64_t>& shape, const std::vector<int32_t>& data, std::string* errorMessage = nullptr) const;

	//! Runs the model; inputs must be in inputNames order
	std::vector<OrtPtr<OrtValue>> run(const std::vector<OrtPtr<OrtValue>>& inputs, std::string* errorMessage = nullptr) const;

	//! Returns the shape of a tensor value
	static std::vector<int64_t> tensorShape(const OrtApi* api, const OrtValue* value);
	//! Returns the raw float32 data of a tensor value (nullptr if not float32)
	static const float* tensorDataF32(const OrtApi* api, const OrtValue* value, size_t* elementCount = nullptr);

	//! Resolves a template shape (-1 = batch, -2 = num_points) to a concrete one
	static std::vector<int64_t> resolveShape(const std::vector<int64_t>& tmpl, int64_t batch, int64_t numPoints);

	//! Looks up one custom metadata entry of the model (e.g. key "semseg.manifest")
	/** \return true when the key exists (valueOut filled), false when the key is
	    absent; errorMessage is set on ORT-level failures (not on a missing key). **/
	bool lookupModelMetadata(const std::string& key, std::string* valueOut, std::string* errorMessage = nullptr) const;

  private:
	bool createWithProvider(const std::vector<char>& modelData, const std::string& provider, std::string* providerError);
	bool readIOInfo(std::string* errorMessage);

	const OrtApi*         m_api = nullptr;
	OrtPtr<OrtEnv>        m_env;
	OrtPtr<OrtSession>    m_session;
	OrtPtr<OrtMemoryInfo> m_memoryInfo;
	std::string           m_activeProvider;
	std::string           m_providerNotes;
};
