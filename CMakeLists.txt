# qSemSeg: semantic segmentation of point clouds with ONNX models
#
# ONNX Runtime is loaded dynamically at runtime (LoadLibrary/dlopen + OrtGetApiBase),
# so this plugin has NO build-time dependency on ONNX Runtime and always builds.
# To deploy the ONNX Runtime shared libraries next to CloudCompare/ccViewer at
# install time, point ONNXRUNTIME_DLL_DIR to the folder containing them
# (e.g. the lib/ folder of onnxruntime-win-x64-gpu-<version>.zip).

option( PLUGIN_STANDARD_QSEMSEG "Semantic segmentation of point clouds with ONNX models (ONNX Runtime, CUDA GPU with CPU fallback)" OFF )

if ( PLUGIN_STANDARD_QSEMSEG )
	project( QSEMSEG_PLUGIN )

	AddPlugin( NAME ${PROJECT_NAME} )

	# vendored ONNX Runtime C API header (MIT)
	target_include_directories( ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party )

	# worker-thread inference (QtConcurrent::run + QFutureWatcher).
	# NOTE: the plain target_link_libraries signature on purpose - AddPlugin
	# already used it for this target (mixing keyword/plain signatures fails).
	target_link_libraries( ${PROJECT_NAME} Qt6::Concurrent )

	# dynamic loading of the ONNX Runtime shared library at runtime
	# NOTE: plain signature, like the Qt6::Concurrent line above (mixing
	# keyword/plain signatures for one target is a configure error).
	if ( NOT WIN32 )
		target_link_libraries( ${PROJECT_NAME} ${CMAKE_DL_LIBS} )
	endif()

	set( ONNXRUNTIME_DLL_DIR "" CACHE PATH "Folder with the ONNX Runtime shared libraries (onnxruntime*.dll / libonnxruntime*.so*) to deploy next to the apps (optional)" )
	if ( ONNXRUNTIME_DLL_DIR )
		if ( WIN32 )
			file( GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_DLL_DIR}/onnxruntime*.dll" )
			# the DirectML package keeps its GPU backend in a separate DLL that
			# does not match the onnxruntime* pattern; without it the DirectML
			# provider cannot be initialized
			if ( EXISTS "${ONNXRUNTIME_DLL_DIR}/DirectML.dll" )
				list( APPEND ONNXRUNTIME_DLLS "${ONNXRUNTIME_DLL_DIR}/DirectML.dll" )
			endif()
		else()
			file( GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_DLL_DIR}/libonnxruntime*.so*" )
		endif()
		if ( ONNXRUNTIME_DLLS )
			copy_files( "${ONNXRUNTIME_DLLS}" "${CLOUDCOMPARE_DEST_FOLDER}" 1 )
			if ( OPTION_BUILD_CCVIEWER )
				copy_files( "${ONNXRUNTIME_DLLS}" "${CCVIEWER_DEST_FOLDER}" 1 )
			endif()
		else()
			message( WARNING "PLUGIN_STANDARD_QSEMSEG: ONNXRUNTIME_DLL_DIR='${ONNXRUNTIME_DLL_DIR}' contains no ONNX Runtime shared library" )
		endif()
	endif()

	add_subdirectory( include )
	add_subdirectory( src )

	if ( BUILD_TESTING )
		add_subdirectory( test )
	endif()

	# ---- qSemSegProbe: out-of-process warm-up probe (crash isolation) ----
	# Runs the plugin's exact startup sequence (ORT load, session creation with
	# the requested providers, manifest/IO validation, one zero-input inference)
	# in its own process: a native crash inside a broken execution provider
	# (e.g. a bad NVIDIA driver) kills the probe, never CloudCompare. The
	# plugin spawns it via QProcess before creating a CUDA session.
	add_executable( qSemSegProbe
		${CMAKE_CURRENT_SOURCE_DIR}/probe/main.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/ModelAdapter.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/ModelSpec.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/OnnxRuntime.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/Preprocess.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/SlidingWindow.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/SphereFormerAdapter.cpp
	)

	target_include_directories( qSemSegProbe PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${CMAKE_CURRENT_SOURCE_DIR}/third_party
	)

	target_link_libraries( qSemSegProbe
		PRIVATE
			Qt6::Core
			QCC_DB_LIB
			QCC_IO_LIB
	)

	if ( NOT WIN32 )
		target_link_libraries( qSemSegProbe PRIVATE ${CMAKE_DL_LIBS} )
	endif()

	# self-contained on Windows (same recipe as TestSemSeg): the probe may be
	# run manually from the build tree, so the Qt / MinGW runtime DLLs must sit
	# next to it
	if( WIN32 )
		add_custom_command( TARGET qSemSegProbe POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:Qt6::Core> $<TARGET_FILE_DIR:qSemSegProbe>
			COMMAND_EXPAND_LISTS
		)
		if( MINGW )
			get_filename_component( MINGW_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY )
			foreach( MINGW_RUNTIME_DLL libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll )
				if( EXISTS ${MINGW_BIN_DIR}/${MINGW_RUNTIME_DLL} )
					add_custom_command( TARGET qSemSegProbe POST_BUILD
						COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MINGW_BIN_DIR}/${MINGW_RUNTIME_DLL} $<TARGET_FILE_DIR:qSemSegProbe>
						COMMAND_EXPAND_LISTS
					)
				endif()
			endforeach()
		endif()
	endif()

	# install next to the application executable (the first qSemSegProbe
	# lookup candidate): the Windows deployment already puts the Qt / MinGW
	# runtime DLLs there, so the probe starts self-contained. The plugins
	# folder has no runtime DLLs beside it - a probe there would die on the
	# missing-DLL dialog before main() (hanging the QProcess spawn).
	install( TARGETS qSemSegProbe
		RUNTIME DESTINATION ${CLOUDCOMPARE_DEST_FOLDER}
	)
	# keep the Qt libraries findable after install on Linux (best effort: the
	# CI deployment story is tracked separately)
	set_property( TARGET qSemSegProbe PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE )
endif()
