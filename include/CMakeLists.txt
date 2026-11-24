
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/ModelAdapter.h
		${CMAKE_CURRENT_LIST_DIR}/ModelSpec.h
		${CMAKE_CURRENT_LIST_DIR}/OnnxRuntime.h
		${CMAKE_CURRENT_LIST_DIR}/OutputMapper.h
		${CMAKE_CURRENT_LIST_DIR}/Preprocess.h
		${CMAKE_CURRENT_LIST_DIR}/qSemSeg.h
		${CMAKE_CURRENT_LIST_DIR}/SemSegDialog.h
		${CMAKE_CURRENT_LIST_DIR}/SlidingWindow.h
		${CMAKE_CURRENT_LIST_DIR}/SphereFormerAdapter.h
)

target_include_directories( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)
