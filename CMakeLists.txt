cmake_minimum_required(VERSION 3.18)
project(cuda_detector LANGUAGES CXX)

option(DETECTOR_CUDA "Build the native CUDA inference backend" OFF)
option(DETECTOR_OPENCV "Read JPEG and PNG images with OpenCV" OFF)
option(DETECTOR_TENSORRT "Build the optional TensorRT inference backend" OFF)

add_library(detector_core native/src/network.cpp native/src/image.cpp native/src/training.cpp
  native/src/yolo_loss.cpp native/src/classifier_trainer.cpp native/src/idx.cpp native/src/metrics.cpp)
target_include_directories(detector_core PUBLIC native/include)
target_compile_features(detector_core PUBLIC cxx_std_17)

if(DETECTOR_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(detector_core PRIVATE native/src/cuda_backend.cu native/src/cuda_classifier_trainer.cu
    native/src/cuda_training.cu)
  target_link_libraries(detector_core PUBLIC CUDA::cudart)
  target_compile_definitions(detector_core PUBLIC DETECTOR_HAS_CUDA=1)
  set_target_properties(detector_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(DETECTOR_TENSORRT)
  if(NOT DETECTOR_CUDA)
    message(FATAL_ERROR "DETECTOR_TENSORRT requires DETECTOR_CUDA=ON")
  endif()
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
  find_library(TENSORRT_LIBRARY nvinfer REQUIRED)
  target_sources(detector_core PRIVATE native/src/trt_session.cpp)
  target_include_directories(detector_core PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(detector_core PUBLIC ${TENSORRT_LIBRARY})
  target_compile_definitions(detector_core PUBLIC DETECTOR_HAS_TENSORRT=1)
endif()

if(DETECTOR_OPENCV)
  find_package(OpenCV REQUIRED COMPONENTS imgcodecs imgproc)
  target_link_libraries(detector_core PUBLIC ${OpenCV_LIBS})
  target_include_directories(detector_core PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(detector_core PRIVATE DETECTOR_HAS_OPENCV=1)
endif()

add_executable(cuda_detector native/src/main.cpp)
target_link_libraries(cuda_detector PRIVATE detector_core)

add_executable(cuda_detector_train native/src/train_main.cpp)
target_link_libraries(cuda_detector_train PRIVATE detector_core)

add_executable(cuda_detector_eval native/src/evaluate_main.cpp)
target_link_libraries(cuda_detector_eval PRIVATE detector_core)

add_executable(cuda_classifier native/src/classifier_main.cpp)
target_link_libraries(cuda_classifier PRIVATE detector_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(detector_tests native/tests/smoke.cpp)
  target_link_libraries(detector_tests PRIVATE detector_core)
  add_test(NAME detector_smoke COMMAND detector_tests)

  add_executable(detector_training_tests native/tests/training_smoke.cpp)
  target_link_libraries(detector_training_tests PRIVATE detector_core)
  add_test(NAME detector_training_smoke COMMAND detector_training_tests)

  add_executable(detector_overfit_tests native/tests/overfit_smoke.cpp)
  target_link_libraries(detector_overfit_tests PRIVATE detector_core)
  add_test(NAME detector_overfit_smoke COMMAND detector_overfit_tests)

  add_executable(detector_metrics_tests native/tests/metrics_smoke.cpp)
  target_link_libraries(detector_metrics_tests PRIVATE detector_core)
  add_test(NAME detector_metrics_smoke COMMAND detector_metrics_tests)

  add_executable(detector_classifier_tests native/tests/classifier_smoke.cpp)
  target_link_libraries(detector_classifier_tests PRIVATE detector_core)
  add_test(NAME detector_classifier_smoke COMMAND detector_classifier_tests)

  add_executable(detector_idx_tests native/tests/idx_smoke.cpp)
  target_link_libraries(detector_idx_tests PRIVATE detector_core)
  add_test(NAME detector_idx_smoke COMMAND detector_idx_tests)

  if(DETECTOR_CUDA)
    add_executable(detector_cuda_training_tests native/tests/cuda_training_smoke.cpp)
    target_link_libraries(detector_cuda_training_tests PRIVATE detector_core)
    add_test(NAME detector_cuda_training_smoke COMMAND detector_cuda_training_tests)
  endif()

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME classifier_cli_cpu
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_classifier_cli.py
        $<TARGET_FILE:cuda_classifier> ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/classifier-linear.cfg)
    if(DETECTOR_CUDA)
      add_test(NAME classifier_cli_cuda
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_classifier_cli.py
          $<TARGET_FILE:cuda_classifier> ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/classifier-linear.cfg --cuda)
      add_test(NAME classifier_cli_cuda_conv
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_classifier_cli.py
          $<TARGET_FILE:cuda_classifier> ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/classifier-conv.cfg --cuda)
    endif()
    add_test(NAME detector_train_augmentation
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_detector_train_augmentation.py
        $<TARGET_FILE:cuda_detector_train> ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/mini.cfg)
    add_test(NAME voc2007_conversion
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_convert_voc2007.py)
    add_test(NAME darknet_voc_preparation
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_prepare_darknet_voc.py)
    add_test(NAME darknet_checkpoint_monitor
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_monitor_darknet_run.py)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import numpy, onnx, onnxruntime"
      RESULT_VARIABLE ONNX_TEST_IMPORT_RESULT
      OUTPUT_QUIET ERROR_QUIET)
    if(ONNX_TEST_IMPORT_RESULT EQUAL 0)
      add_test(NAME darknet_onnx_export
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/native/tests/test_export_darknet_onnx.py)
    endif()
  endif()
endif()
