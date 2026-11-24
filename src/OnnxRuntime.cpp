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

#include "OnnxRuntime.h"

#include <QCoreApplication>
#include <QStringList>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{
#ifdef _WIN32
	std::wstring widen(const std::string& s)
	{
		int          n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
		std::wstring w(static_cast<size_t>(n > 0 ? n - 1 : 0), L'\0');
		if (n > 0)
		{
			MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
		}
		return w;
	}

	std::string narrow(const wchar_t* w)
	{
		int         n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
		std::string s(static_cast<size_t>(n > 0 ? n - 1 : 0), '\0');
		if (n > 0)
		{
			WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
		}
		return s;
	}
#endif

	typedef const OrtApiBase* (*OrtGetApiBaseFn)(void);

	//! Extracts the error message of an OrtStatus and releases it
	std::string statusError(const OrtApi* api, OrtStatus* status)
	{
		if (!status)
		{
			return {};
		}
		std::string message = api->GetErrorMessage(status);
		api->ReleaseStatus(status);
		return message;
	}

	//! Checks an ORT call, throwing std::runtime_error on failure
	void check(const OrtApi* api, OrtStatus* status, const char* what)
	{
		if (status)
		{
			std::string message = statusError(api, status);
			throw std::runtime_error(std::string(what) + ": " + message);
		}
	}

	//! Resolves the full path of a loaded shared library (for diagnostics)
	/** Returns the actual on-disk location, not the (possibly bare) name that
	    was used to load it, so a library found through the OS search path is
	    reported with its true location. **/
	std::string modulePath(void* module)
	{
#ifdef _WIN32
		wchar_t path[MAX_PATH + 1] = {};
		if (module && GetModuleFileNameW(reinterpret_cast<HMODULE>(module), path, MAX_PATH))
		{
			return narrow(path);
		}
		return {};
#else
		Dl_info info{};
		if (module && dladdr(module, &info) && info.dli_fname)
		{
			return info.dli_fname;
		}
		return {};
#endif
	}
} // namespace

OrtLibrary& OrtLibrary::instance()
{
	static OrtLibrary s_instance;
	return s_instance;
}

OrtLibrary::~OrtLibrary()
{
	// NOTE: the library is deliberately left loaded until process exit.
	// Calling FreeLibrary while an ORT session used the CUDA execution
	// provider crashes inside the NVIDIA driver (nvcuda64.dll) during
	// context teardown. Unloading happens implicitly at process exit.
}

bool OrtLibrary::load(const std::vector<std::string>& explicitPaths, std::string* errorMessage)
{
	if (m_api)
	{
		return true; // already loaded
	}

	std::vector<std::string> candidates = explicitPaths;
#ifdef _WIN32
	candidates.push_back("onnxruntime.dll");
#else
	candidates.push_back("libonnxruntime.so.1");
	candidates.push_back("libonnxruntime.so");
#endif

	std::string report;
	for (const std::string& candidate : candidates)
	{
		// 1) load the shared library. Every module loaded here is deliberately
		//    kept resident (see the destructor note): unloading an ORT library
		//    that was used by a CUDA session crashes in the NVIDIA driver.
#ifdef _WIN32
		void* module = reinterpret_cast<void*>(LoadLibraryW(widen(candidate).c_str()));
		if (!module)
		{
			report += candidate + ": could not be loaded (Windows error " + std::to_string(GetLastError()) + ")\n";
			continue;
		}
#else
		void* module = dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
		if (!module)
		{
			const char* dlError = dlerror();
			report += candidate + ": could not be loaded (" + (dlError ? dlError : "unknown error") + ")\n";
			continue;
		}
#endif

		// 2) locate the C API entry point
#ifdef _WIN32
		auto getApiBase = reinterpret_cast<OrtGetApiBaseFn>(GetProcAddress(static_cast<HMODULE>(module), "OrtGetApiBase"));
#else
		auto getApiBase = reinterpret_cast<OrtGetApiBaseFn>(dlsym(module, "OrtGetApiBase"));
#endif
		if (!getApiBase)
		{
			report += candidate + ": OrtGetApiBase not found (not an ONNX Runtime library)\n";
			continue;
		}

		const OrtApiBase* apiBase = getApiBase();
		if (!apiBase)
		{
			report += candidate + ": OrtGetApiBase returned null\n";
			continue;
		}
		const char* versionStr = apiBase->GetVersionString();
		std::string version    = versionStr ? versionStr : "unknown";

		// 3) the vendored C API header requires a runtime API >= ORT_API_VERSION.
		//    An older runtime is recorded and SKIPPED (not fatal): the search
		//    continues, so a stale library on the OS search path (e.g. in the
		//    Windows system directory) cannot shadow a usable one next to the app.
		const OrtApi* api = apiBase->GetApi(ORT_API_VERSION);
		if (!api)
		{
			report += candidate + ": version " + version + " is too old for this plugin (C API "
			          + std::to_string(ORT_API_VERSION) + " required) - skipped\n";
			continue;
		}

		// success
		m_module     = module;
		m_apiBase    = apiBase;
		m_api        = api;
		m_loadedPath = modulePath(module);
		report += candidate + ": version " + version + " - loaded from " + (m_loadedPath.empty() ? candidate : m_loadedPath) + "\n";
		m_loadReport = report;
		return true;
	}

	m_loadReport = report;
	if (errorMessage)
	{
		*errorMessage = report
		                + "no usable ONNX Runtime shared library was found (C API version " + std::to_string(ORT_API_VERSION)
		                + " required). Download the runtime from https://onnxruntime.ai and deploy it next to the application"
		                  " (onnxruntime.dll + onnxruntime_providers_*.dll on Windows, libonnxruntime.so* on Linux)";
	}
	return false;
}

std::string OrtLibrary::cudaProviderHint() const
{
	if (!m_api || m_loadedPath.empty())
	{
		return {};
	}
	namespace fs       = std::filesystem;
	const fs::path dir = fs::path(m_loadedPath).parent_path();
#ifdef _WIN32
	const fs::path cudaDll   = dir / "onnxruntime_providers_cuda.dll";
	const fs::path sharedDll = dir / "onnxruntime_providers_shared.dll";
	if (!fs::exists(cudaDll))
	{
		return "onnxruntime_providers_cuda.dll is missing next to the ONNX Runtime library ("
		       + m_loadedPath + "): GPU support needs it plus onnxruntime_providers_shared.dll and the NVIDIA runtime DLLs";
	}
	if (!fs::exists(sharedDll))
	{
		return "onnxruntime_providers_shared.dll is missing next to the ONNX Runtime library ("
		       + m_loadedPath + "): the CUDA execution provider needs it";
	}
#else
	const fs::path cudaLib = dir / "libonnxruntime_providers_cuda.so";
	if (!fs::exists(cudaLib))
	{
		return "libonnxruntime_providers_cuda.so is missing next to the ONNX Runtime library ("
		       + m_loadedPath + "): GPU support needs it";
	}
#endif
	return {};
}

std::string OrtLibrary::version() const
{
	if (!m_apiBase)
	{
		return {};
	}
	const char* version = m_apiBase->GetVersionString();
	return version ? version : "";
}

std::vector<std::string> availableExecutionProviders()
{
	std::vector<std::string> providers;
	const OrtApi*            api = OrtLibrary::instance().api();
	if (!api)
	{
		return providers;
	}
	char**     names  = nullptr;
	int        count  = 0;
	OrtStatus* status = api->GetAvailableProviders(&names, &count);
	if (status)
	{
		api->ReleaseStatus(status);
		return providers;
	}
	for (int i = 0; i < count; ++i)
	{
		if (names[i])
		{
			providers.emplace_back(names[i]);
		}
	}
	// the release call returns a status (never fails per the C API docs, but
	// the pointer must still be released)
	status = api->ReleaseAvailableProviders(names, count);
	if (status)
	{
		api->ReleaseStatus(status);
	}
	return providers;
}

bool isGpuExecutionProvider(const std::string& provider)
{
	return provider == "CUDAExecutionProvider" || provider == "DmlExecutionProvider";
}

std::string providerDisplayName(const std::string& provider)
{
	if (provider == "CPUExecutionProvider")
	{
		return "CPU";
	}
	if (provider == "CUDAExecutionProvider")
	{
		return "CUDA (NVIDIA GPU)";
	}
	if (provider == "DmlExecutionProvider")
	{
		return "DirectML (any DX12 GPU)";
	}
	return provider;
}

bool OrtSession::create(const std::string& modelPath, const std::vector<std::string>& requestedProviders, std::string* errorMessage)
{
	OrtLibrary& ort = OrtLibrary::instance();
	if (!ort.isLoaded())
	{
		if (errorMessage)
		{
			*errorMessage = "ONNX Runtime library is not loaded";
		}
		return false;
	}
	m_api = ort.api();

	// re-creation support (the plugin retries on the CPU when a provider
	// produced NaN/Inf): release the previous session BEFORE the environment
	// it was created from - a session must not outlive its environment
	m_session.reset();
	m_memoryInfo.reset();
	m_env.reset();
	m_activeProvider.clear();
	m_providerNotes.clear();

	// read the model file (CreateSessionFromArray avoids wchar path issues on Windows)
	std::ifstream modelFile(modelPath, std::ios::binary | std::ios::ate);
	if (!modelFile)
	{
		if (errorMessage)
		{
			*errorMessage = "cannot open model file: " + modelPath;
		}
		return false;
	}
	std::vector<char> modelData(static_cast<size_t>(modelFile.tellg()));
	modelFile.seekg(0);
	modelFile.read(modelData.data(), static_cast<std::streamsize>(modelData.size()));
	if (!modelFile)
	{
		if (errorMessage)
		{
			*errorMessage = "cannot read model file: " + modelPath;
		}
		return false;
	}

	// environment + CPU memory info for tensor creation
	OrtEnv* env = nullptr;
	check(m_api, m_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "qSemSeg", &env), "CreateEnv");
	m_env = makeOrtPtr(m_api, env);

	OrtMemoryInfo* memoryInfo = nullptr;
	check(m_api, m_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memoryInfo), "CreateCpuMemoryInfo");
	m_memoryInfo = makeOrtPtr(m_api, memoryInfo);

	// providers actually available on this machine
	char** availableProviders = nullptr;
	int    availableCount     = 0;
	check(m_api, m_api->GetAvailableProviders(&availableProviders, &availableCount), "GetAvailableProviders");
	std::vector<std::string> available;
	for (int i = 0; i < availableCount; ++i)
	{
		available.emplace_back(availableProviders[i]);
	}
	if (OrtStatus* st = m_api->ReleaseAvailableProviders(availableProviders, availableCount))
	{
		// cleanup only: the provider names have already been copied above
		m_api->ReleaseStatus(st);
	}

	std::string lastProviderError;
	for (const std::string& provider : requestedProviders)
	{
		bool listed = false;
		for (const std::string& a : available)
		{
			listed = listed || (a == provider);
		}
		if (!listed)
		{
			// e.g. the DirectML provider on a CUDA build, or CUDA without its
			// provider DLLs. Recorded so the caller can tell the user why it
			// ended up on another provider.
			m_providerNotes += provider + ": not provided by this ONNX Runtime build (" + OrtLibrary::instance().loadedPath() + ")\n";
			continue;
		}
		if (createWithProvider(modelData, provider, &lastProviderError))
		{
			m_activeProvider = provider;
			if (!readIOInfo(errorMessage))
			{
				return false;
			}
			return true;
		}
		m_providerNotes += provider + ": " + lastProviderError + "\n";
	}

	// last resort: default options (CPU provider)
	if (lastProviderError.empty())
	{
		lastProviderError = "none of the requested providers is available on this machine";
	}
	if (createWithProvider(modelData, "CPUExecutionProvider", &lastProviderError))
	{
		m_activeProvider = "CPUExecutionProvider";
		if (!readIOInfo(errorMessage))
		{
			return false;
		}
		return true;
	}

	if (errorMessage)
	{
		*errorMessage = "could not create an ONNX session for " + modelPath + " (" + lastProviderError + ")";
	}
	return false;
}

bool OrtSession::createWithProvider(const std::vector<char>& modelData, const std::string& provider, std::string* providerError)
{
	OrtSessionOptions* options = nullptr;
	OrtStatus*         status  = m_api->CreateSessionOptions(&options);
	if (status)
	{
		if (providerError)
		{
			*providerError = statusError(m_api, status);
		}
		return false;
	}
	OrtPtr<OrtSessionOptions> optionsPtr = makeOrtPtr(m_api, options);

	// optional tuning: a failure here only means ORT keeps its own defaults,
	// so the status is released and the session creation goes on
	if (OrtStatus* st = m_api->SetIntraOpNumThreads(options, 0)) // 0 = default (all cores)
	{
		m_api->ReleaseStatus(st);
	}
	// ORT severities: 0 verbose, 1 info, 2 warning, 3 error. Info would dump
	// hundreds of graph-optimization lines (the DirectML provider especially)
	// into the console / the probe output, so warnings and errors only.
	if (OrtStatus* st = m_api->SetSessionLogSeverityLevel(options, 2))
	{
		m_api->ReleaseStatus(st);
	}

	if (provider == "DmlExecutionProvider")
	{
		// DirectML's default graph-fusion pass aborts session creation with
		// E_INVALIDARG on large exported graphs (reproduced with the
		// SphereFormer export). Running without fusion is supported, is
		// numerically equivalent (max |diff| 5e-4, identical argmax on the
		// plugin's real patch) and is the configuration the plugin was
		// validated against. Must be set before the provider is appended.
		status = m_api->AddSessionConfigEntry(options, "ep.dml.disable_graph_fusion", "1");
		if (status)
		{
			if (providerError)
			{
				*providerError = "DirectML provider configuration failed: " + statusError(m_api, status);
			}
			return false;
		}
	}

	if (provider == "CUDAExecutionProvider")
	{
		OrtCUDAProviderOptions cudaOptions{};
		cudaOptions.device_id = 0;
		status                = m_api->SessionOptionsAppendExecutionProvider_CUDA(options, &cudaOptions);
		if (status)
		{
			if (providerError)
			{
				*providerError = "CUDAExecutionProvider cannot be initialized: " + statusError(m_api, status);
			}
			return false;
		}
	}
	else if (provider != "CPUExecutionProvider")
	{
		// CPU is the built-in default provider, nothing to append
		status = m_api->SessionOptionsAppendExecutionProvider(options, provider.c_str(), nullptr, nullptr, 0);
		if (status)
		{
			if (providerError)
			{
				*providerError = provider + " cannot be initialized: " + statusError(m_api, status);
			}
			return false;
		}
	}

	OrtSession* session = nullptr;
	status              = m_api->CreateSessionFromArray(m_env.get(), modelData.data(), modelData.size(), options, &session);
	if (status)
	{
		if (providerError)
		{
			*providerError = "session creation with " + provider + " failed: " + statusError(m_api, status);
		}
		return false;
	}
	m_session = makeOrtPtr(m_api, session);
	return true;
}

bool OrtSession::readIOInfo(std::string* errorMessage)
{
	// readIOInfo is called once per create(); clearing first keeps create()
	// safe to call again on the same object (e.g. the plugin's CPU fallback
	// after a provider produced NaN/Inf values)
	inputNames.clear();
	outputNames.clear();
	inputTypes.clear();
	inputShapes.clear();
	outputTypes.clear();
	outputShapes.clear();

	OrtAllocator* allocator = nullptr;
	check(m_api, m_api->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions");

	size_t inputCount = 0;
	check(m_api, m_api->SessionGetInputCount(m_session.get(), &inputCount), "SessionGetInputCount");
	for (size_t i = 0; i < inputCount; ++i)
	{
		char* name = nullptr;
		check(m_api, m_api->SessionGetInputName(m_session.get(), i, allocator, &name), "SessionGetInputName");
		inputNames.emplace_back(name ? name : "");
		allocator->Free(allocator, name);

		OrtTypeInfo* typeInfo = nullptr;
		check(m_api, m_api->SessionGetInputTypeInfo(m_session.get(), i, &typeInfo), "SessionGetInputTypeInfo");
		OrtPtr<OrtTypeInfo> typeInfoPtr = makeOrtPtr(m_api, typeInfo);

		const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
		check(m_api, m_api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo), "CastTypeInfoToTensorInfo");

		ONNXTensorElementDataType elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		check(m_api, m_api->GetTensorElementType(tensorInfo, &elementType), "GetTensorElementType");
		inputTypes.push_back(elementType);

		size_t dimCount = 0;
		check(m_api, m_api->GetDimensionsCount(tensorInfo, &dimCount), "GetDimensionsCount");
		std::vector<int64_t> dims(dimCount, 0);
		check(m_api, m_api->GetDimensions(tensorInfo, dims.data(), dimCount), "GetDimensions");

		// normalize: symbolic/dynamic dims come back as <= 0; dim 0 = batch (-1),
		// any other dynamic dim = num_points (-2)
		for (size_t d = 0; d < dims.size(); ++d)
		{
			if (dims[d] <= 0)
			{
				dims[d] = (d == 0) ? -1 : -2;
			}
		}
		inputShapes.push_back(dims);

		// tensorInfo is borrowed from typeInfo: do NOT release it here
		// (typeInfoPtr above owns it)
	}

	size_t outputCount = 0;
	check(m_api, m_api->SessionGetOutputCount(m_session.get(), &outputCount), "SessionGetOutputCount");
	for (size_t i = 0; i < outputCount; ++i)
	{
		char* name = nullptr;
		check(m_api, m_api->SessionGetOutputName(m_session.get(), i, allocator, &name), "SessionGetOutputName");
		outputNames.emplace_back(name ? name : "");
		allocator->Free(allocator, name);

		OrtTypeInfo* typeInfo = nullptr;
		check(m_api, m_api->SessionGetOutputTypeInfo(m_session.get(), i, &typeInfo), "SessionGetOutputTypeInfo");
		OrtPtr<OrtTypeInfo> typeInfoPtr = makeOrtPtr(m_api, typeInfo);

		const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
		check(m_api, m_api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo), "CastTypeInfoToTensorInfo");

		ONNXTensorElementDataType elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		check(m_api, m_api->GetTensorElementType(tensorInfo, &elementType), "GetTensorElementType");
		outputTypes.push_back(elementType);

		size_t dimCount = 0;
		check(m_api, m_api->GetDimensionsCount(tensorInfo, &dimCount), "GetDimensionsCount");
		std::vector<int64_t> dims(dimCount, 0);
		check(m_api, m_api->GetDimensions(tensorInfo, dims.data(), dimCount), "GetDimensions");

		// same normalization as the inputs: symbolic/dynamic dims come back as
		// <= 0; dim 0 = batch (-1), any other dynamic dim = num_points (-2)
		for (size_t d = 0; d < dims.size(); ++d)
		{
			if (dims[d] <= 0)
			{
				dims[d] = (d == 0) ? -1 : -2;
			}
		}
		outputShapes.push_back(dims);

		// tensorInfo is borrowed from typeInfo: do NOT release it here
		// (typeInfoPtr above owns it)
	}

	// NOTE: the allocator from GetAllocatorWithDefaultOptions is owned by the
	// framework and must NOT be released (docs of OrtApi::GetAllocatorWithDefaultOptions)
	return true;
}

OrtPtr<OrtValue> OrtSession::makeTensorF32(const std::vector<int64_t>& shape, const std::vector<float>& data, std::string* errorMessage) const
{
	size_t elementCount = 1;
	for (int64_t d : shape)
	{
		elementCount *= static_cast<size_t>(d);
	}
	if (data.size() != elementCount)
	{
		if (errorMessage)
		{
			*errorMessage = "makeTensorF32: data size does not match the shape";
		}
		return nullptr;
	}

	OrtValue*  value  = nullptr;
	OrtStatus* status = m_api->CreateTensorWithDataAsOrtValue(m_memoryInfo.get(),
	                                                          const_cast<float*>(data.data()),
	                                                          data.size() * sizeof(float),
	                                                          shape.data(),
	                                                          shape.size(),
	                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
	                                                          &value);
	if (status)
	{
		if (errorMessage)
		{
			*errorMessage = "CreateTensorWithDataAsOrtValue (float32) failed: " + statusError(m_api, status);
		}
		return nullptr;
	}
	return makeOrtPtr(m_api, value);
}

OrtPtr<OrtValue> OrtSession::makeTensorI64(const std::vector<int64_t>& shape, const std::vector<int64_t>& data, std::string* errorMessage) const
{
	size_t elementCount = 1;
	for (int64_t d : shape)
	{
		elementCount *= static_cast<size_t>(d);
	}
	if (data.size() != elementCount)
	{
		if (errorMessage)
		{
			*errorMessage = "makeTensorI64: data size does not match the shape";
		}
		return nullptr;
	}

	OrtValue*  value  = nullptr;
	OrtStatus* status = m_api->CreateTensorWithDataAsOrtValue(m_memoryInfo.get(),
	                                                          const_cast<int64_t*>(data.data()),
	                                                          data.size() * sizeof(int64_t),
	                                                          shape.data(),
	                                                          shape.size(),
	                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
	                                                          &value);
	if (status)
	{
		if (errorMessage)
		{
			*errorMessage = "CreateTensorWithDataAsOrtValue (int64) failed: " + statusError(m_api, status);
		}
		return nullptr;
	}
	return makeOrtPtr(m_api, value);
}

OrtPtr<OrtValue> OrtSession::makeTensorI32(const std::vector<int64_t>& shape, const std::vector<int32_t>& data, std::string* errorMessage) const
{
	size_t elementCount = 1;
	for (int64_t d : shape)
	{
		elementCount *= static_cast<size_t>(d);
	}
	if (data.size() != elementCount)
	{
		if (errorMessage)
		{
			*errorMessage = "makeTensorI32: data size does not match the shape";
		}
		return nullptr;
	}

	OrtValue*  value  = nullptr;
	OrtStatus* status = m_api->CreateTensorWithDataAsOrtValue(m_memoryInfo.get(),
	                                                          const_cast<int32_t*>(data.data()),
	                                                          data.size() * sizeof(int32_t),
	                                                          shape.data(),
	                                                          shape.size(),
	                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
	                                                          &value);
	if (status)
	{
		if (errorMessage)
		{
			*errorMessage = "CreateTensorWithDataAsOrtValue (int32) failed: " + statusError(m_api, status);
		}
		return nullptr;
	}
	return makeOrtPtr(m_api, value);
}

std::string allocationFailureHint(const std::string& ortMessage)
{
	std::string lowered = ortMessage;
	for (char& c : lowered)
	{
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	if (lowered.find("failed to allocate memory") == std::string::npos && lowered.find("out of memory") == std::string::npos && lowered.find("bad_alloc") == std::string::npos)
	{
		return {};
	}
	return " (hint: the model ran out of memory for this input size - the per-patch working buffers "
	       "grow with patch points x batch size, so lower the points-per-patch toward the model's "
	       "trained value and/or set the patch batch size to 1, then retry)";
}

std::vector<std::string> defaultOrtLibraryCandidates()
{
	std::vector<std::string> candidates;
	const QString            appDir = QCoreApplication::applicationDirPath();
	const QStringList        dirs   = {
        appDir,
        appDir + QStringLiteral("/plugins"),
        appDir + QStringLiteral("/../CloudCompare"),
        appDir + QStringLiteral("/../install/CloudCompare"),
    };
#ifdef _WIN32
	const QLatin1String libName("/onnxruntime.dll");
	for (const QString& dir : dirs)
	{
		candidates.push_back(QString(dir + libName).toStdString());
	}
#else
	const QStringList libNames = {QStringLiteral("/libonnxruntime.so.1"), QStringLiteral("/libonnxruntime.so")};
	for (const QString& dir : dirs)
	{
		for (const QString& name : libNames)
		{
			candidates.push_back(QString(dir + name).toStdString());
		}
	}
#endif
	return candidates;
}

std::vector<OrtPtr<OrtValue>> OrtSession::run(const std::vector<OrtPtr<OrtValue>>& inputs, std::string* errorMessage) const
{
	std::vector<const char*> inputNamesC;
	for (const std::string& name : inputNames)
	{
		inputNamesC.push_back(name.c_str());
	}
	std::vector<const OrtValue*> inputValues;
	for (const OrtPtr<OrtValue>& value : inputs)
	{
		inputValues.push_back(value.get());
	}
	std::vector<const char*> outputNamesC;
	for (const std::string& name : outputNames)
	{
		outputNamesC.push_back(name.c_str());
	}
	std::vector<OrtValue*> rawOutputs(outputNames.size(), nullptr);

	OrtStatus* status = m_api->Run(m_session.get(),
	                               nullptr,
	                               inputNamesC.data(),
	                               inputValues.data(),
	                               inputValues.size(),
	                               outputNamesC.data(),
	                               outputNamesC.size(),
	                               rawOutputs.data());
	if (status)
	{
		if (errorMessage)
		{
			std::string message = "Run failed: " + statusError(m_api, status);
			message += allocationFailureHint(message);
			*errorMessage = message;
		}
		for (OrtValue* output : rawOutputs)
		{
			if (output)
			{
				m_api->ReleaseValue(output);
			}
		}
		return {};
	}

	std::vector<OrtPtr<OrtValue>> outputs;
	for (OrtValue* output : rawOutputs)
	{
		outputs.push_back(makeOrtPtr(m_api, output));
	}
	return outputs;
}

std::vector<int64_t> OrtSession::tensorShape(const OrtApi* api, const OrtValue* value)
{
	OrtTensorTypeAndShapeInfo* info = nullptr;
	if (OrtStatus* st = api->GetTensorTypeAndShape(value, &info))
	{
		api->ReleaseStatus(st);
		return {};
	}
	if (!info)
	{
		return {};
	}
	OrtPtr<OrtTensorTypeAndShapeInfo> infoPtr = makeOrtPtr(api, info);

	size_t dimCount = 0;
	if (OrtStatus* st = api->GetDimensionsCount(info, &dimCount))
	{
		api->ReleaseStatus(st);
		return {};
	}
	std::vector<int64_t> dims(dimCount, 0);
	if (dimCount > 0)
	{
		if (OrtStatus* st = api->GetDimensions(info, dims.data(), dimCount))
		{
			api->ReleaseStatus(st);
			return {};
		}
	}
	return dims;
}

const float* OrtSession::tensorDataF32(const OrtApi* api, const OrtValue* value, size_t* elementCount)
{
	OrtTensorTypeAndShapeInfo* info = nullptr;
	if (OrtStatus* st = api->GetTensorTypeAndShape(value, &info))
	{
		api->ReleaseStatus(st);
		return nullptr;
	}
	if (!info)
	{
		return nullptr;
	}
	OrtPtr<OrtTensorTypeAndShapeInfo> infoPtr = makeOrtPtr(api, info);

	ONNXTensorElementDataType elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	if (OrtStatus* st = api->GetTensorElementType(info, &elementType))
	{
		api->ReleaseStatus(st);
		return nullptr;
	}
	if (elementType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	{
		return nullptr;
	}

	const void* data = nullptr;
	if (OrtStatus* st = api->GetTensorData(value, &data))
	{
		api->ReleaseStatus(st);
		return nullptr;
	}
	if (!data)
	{
		return nullptr;
	}

	if (elementCount)
	{
		size_t dimCount = 0;
		if (OrtStatus* st = api->GetDimensionsCount(info, &dimCount))
		{
			api->ReleaseStatus(st);
			return nullptr;
		}
		std::vector<int64_t> dims(dimCount, 0);
		if (dimCount > 0)
		{
			if (OrtStatus* st = api->GetDimensions(info, dims.data(), dimCount))
			{
				api->ReleaseStatus(st);
				return nullptr;
			}
		}
		size_t count = 1;
		for (int64_t d : dims)
		{
			count *= static_cast<size_t>(d);
		}
		*elementCount = count;
	}

	return static_cast<const float*>(data);
}

std::vector<int64_t> OrtSession::resolveShape(const std::vector<int64_t>& tmpl, int64_t batch, int64_t numPoints)
{
	std::vector<int64_t> out = tmpl;
	for (int64_t& d : out)
	{
		if (d == -1)
		{
			d = batch;
		}
		else if (d == -2)
		{
			d = numPoints;
		}
	}
	return out;
}

bool OrtSession::lookupModelMetadata(const std::string& key, std::string* valueOut, std::string* errorMessage) const
{
	if (!m_session || !m_api)
	{
		if (errorMessage)
		{
			*errorMessage = "no session";
		}
		return false;
	}

	OrtModelMetadata* rawMetadata = nullptr;
	if (OrtStatus* st = m_api->SessionGetModelMetadata(m_session.get(), &rawMetadata))
	{
		if (errorMessage)
		{
			*errorMessage = m_api->GetErrorMessage(st);
		}
		m_api->ReleaseStatus(st);
		return false;
	}
	OrtPtr<OrtModelMetadata> metadata = makeOrtPtr(m_api, rawMetadata);

	// the default allocator must not be released (OrtLibrary gotcha 2)
	OrtAllocator* allocator = nullptr;
	if (OrtStatus* st = m_api->GetAllocatorWithDefaultOptions(&allocator))
	{
		if (errorMessage)
		{
			*errorMessage = m_api->GetErrorMessage(st);
		}
		m_api->ReleaseStatus(st);
		return false;
	}

	char* rawValue = nullptr;
	if (OrtStatus* st = m_api->ModelMetadataLookupCustomMetadataMap(metadata.get(), allocator, key.c_str(), &rawValue))
	{
		if (errorMessage)
		{
			*errorMessage = m_api->GetErrorMessage(st);
		}
		m_api->ReleaseStatus(st);
		return false;
	}
	if (!rawValue)
	{
		return false; // key not present
	}
	if (valueOut)
	{
		*valueOut = rawValue;
	}
	if (OrtStatus* st = m_api->AllocatorFree(allocator, rawValue))
	{
		// the value has already been copied out: a failed free is not actionable
		m_api->ReleaseStatus(st);
	}
	return true;
}
