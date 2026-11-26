cmake_minimum_required(VERSION 3.20)
project(ekf2core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(PX4 ${CMAKE_CURRENT_SOURCE_DIR}/px4)
set(EKF ${PX4}/modules/ekf2/EKF)

add_library(ekf2core SHARED
  wrapper/ekf_c_api.cpp
  # --- ecl EKF core (unconditional set from EKF/CMakeLists.txt) ---
  ${EKF}/control.cpp
  ${EKF}/covariance.cpp
  ${EKF}/ekf.cpp
  ${EKF}/ekf_helper.cpp
  ${EKF}/estimator_interface.cpp
  ${EKF}/height_control.cpp
  ${EKF}/velocity_fusion.cpp
  ${EKF}/position_fusion.cpp
  ${EKF}/yaw_fusion.cpp
  ${EKF}/imu_down_sampler/imu_down_sampler.cpp
  ${EKF}/aid_sources/fake_height_control.cpp
  ${EKF}/aid_sources/fake_pos_control.cpp
  ${EKF}/aid_sources/ZeroGyroUpdate.cpp
  ${EKF}/aid_sources/ZeroVelocityUpdate.cpp
  # --- external vision aiding (CONFIG_EKF2_EXTERNAL_VISION) ---
  ${EKF}/aid_sources/external_vision/ev_control.cpp
  ${EKF}/aid_sources/external_vision/ev_height_control.cpp
  ${EKF}/aid_sources/external_vision/ev_pos_control.cpp
  ${EKF}/aid_sources/external_vision/ev_vel_control.cpp
  ${EKF}/aid_sources/external_vision/ev_yaw_control.cpp
  # --- subdir libs ---
  ${EKF}/bias_estimator/bias_estimator.cpp
  ${EKF}/output_predictor/output_predictor.cpp
  # --- geo / lat_lon_alt (local NED helpers) ---
  ${PX4}/lib/geo/geo.cpp
  ${PX4}/lib/lat_lon_alt/lat_lon_alt.cpp
)

target_include_directories(ekf2core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/shims   # MUST be first: shadows px4_platform_common/*, uORB/topics/*
  ${EKF}                              # "common.h", relative EKF includes
  ${EKF}/python                       # <ekf_derivation/generated/state.h>
  ${PX4}                              # <lib/atmosphere/...>, <lib/ringbuffer/...>, <lib/lat_lon_alt/...>
  ${PX4}/lib                          # <mathlib/...>, <matrix/matrix/...>
  ${PX4}/lib/matrix                   # <matrix/math.hpp>
)

target_compile_definitions(ekf2core PRIVATE
  _USE_MATH_DEFINES                   # MSVC: expose M_PI etc from <cmath>
  CONFIG_EKF2_EXTERNAL_VISION
  NOMINMAX
  _CRT_SECURE_NO_WARNINGS
  MODULE_NAME="ekf2"
)

# /fp:precise (NOT /fp:fast) — the filter's numerical stability needs
# non-reassociated float math (GCC build uses -fno-associative-math).
target_compile_options(ekf2core PRIVATE /fp:precise /bigobj /EHsc /wd4267 /wd4244 /wd4305 /wd4305)

set_target_properties(ekf2core PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
