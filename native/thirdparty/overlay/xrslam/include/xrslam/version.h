// Hand-written stand-in for the header XRSLAM's CMake generates from xrslam/cmake/version.h.in.
// iOS build: threading on (feature tracker + sliding-window solver on their own threads), debug
// inspection on (exposes landmarks and timings), debug asserts off (they abort()).
#ifndef XRSLAM_VERSION_H
#define XRSLAM_VERSION_H

// clang-format off

#define XRSLAM_MAJOR_VERSION 0
#define XRSLAM_MINOR_VERSION 5
#define XRSLAM_PATCH_VERSION 0

#define XRSLAM_VERSION_STRING "xrslam v0.5.0"

#define xrslam xrslam_0_5_0

#define XRSLAM_ENABLE_DEBUG_INSPECTION
#define XRSLAM_ENABLE_THREADING
#define XRSLAM_IOS
// clang-format on

#endif // XRSLAM_VERSION_H
