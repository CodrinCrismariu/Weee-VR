# Offline XRSLAM replay (Linux, WSL): the app's XRSLAM + Ceres + wrapper, built with the app's flags.
#   cmake -S scripts/replay -B ~/xrslam-replay -G Ninja -DTHIRD_PARTY=$HOME/weee-thirdparty/ThirdParty
#   ninja -C ~/xrslam-replay && ~/xrslam-replay/xrslam-replay <recording-dir>
cmake_minimum_required(VERSION 3.16)
project(xrslam_replay CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -g0 -w -include cstdint")   # GCC: XRSLAM relies on clang's implicit <cstdint>

set(THIRD_PARTY "$ENV{HOME}/weee-thirdparty/ThirdParty" CACHE PATH "scripts/fetch-xrslam.sh output")
set(APP "${CMAKE_CURRENT_SOURCE_DIR}/../../native/WeeeVR/Sources/XRSLAMCore")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video features2d calib3d)
find_package(Threads REQUIRED)

file(GLOB_RECURSE CERES_SOURCES "${THIRD_PARTY}/ceres/internal/ceres/*.cc")
file(GLOB_RECURSE XRSLAM_SOURCES "${THIRD_PARTY}/xrslam/src/*.cpp" "${THIRD_PARTY}/xrslam/extra/src/*.cpp")

add_executable(xrslam-replay replay.cpp log_stdout.cpp "${APP}/WeeeSLAM.cpp" "${APP}/Relocalizer.cpp" ${XRSLAM_SOURCES} ${CERES_SOURCES})
target_include_directories(xrslam-replay PRIVATE
    "${APP}/include"
    "${THIRD_PARTY}/eigen"
    "${THIRD_PARTY}/ceres/include"
    "${THIRD_PARTY}/ceres/config"
    "${THIRD_PARTY}/ceres/internal"
    "${THIRD_PARTY}/ceres/internal/ceres/miniglog"
    "${THIRD_PARTY}/xrslam/include"
    "${THIRD_PARTY}/xrslam/src"
    "${THIRD_PARTY}/xrslam/extra/include"
    ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(xrslam-replay PRIVATE XRSLAM_REPLAY_TRACE)   # keyframe/landmark trace (stderr)
target_link_libraries(xrslam-replay PRIVATE ${OpenCV_LIBS} Threads::Threads)
