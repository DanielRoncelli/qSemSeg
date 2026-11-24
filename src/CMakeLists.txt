
target_sources( ${PROJECT_NAME}
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/ModelAdapter.cpp
		${CMAKE_CURRENT_LIST_DIR}/ModelSpec.cpp
		${CMAKE_CURRENT_LIST_DIR}/OnnxRuntime.cpp
		${CMAKE_CURRENT_LIST_DIR}/OutputMapper.cpp
		${CMAKE_CURRENT_LIST_DIR}/Preprocess.cpp
		${CMAKE_CURRENT_LIST_DIR}/qSemSeg.cpp
		${CMAKE_CURRENT_LIST_DIR}/SemSegDialog.cpp
		${CMAKE_CURRENT_LIST_DIR}/SlidingWindow.cpp
		${CMAKE_CURRENT_LIST_DIR}/SphereFormerAdapter.cpp
)
