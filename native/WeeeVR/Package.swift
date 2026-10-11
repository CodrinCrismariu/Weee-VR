// swift-tools-version: 6.0

import Foundation
import PackageDescription

// Third-party C/C++ (XRSLAM, Ceres, Eigen, OpenCV, AprilTag) is assembled by scripts/fetch-xrslam.sh and mirrored
// into ThirdParty/ by scripts/dev.sh.
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let thirdParty = "\(root)/ThirdParty"

// Always optimised: xtool dev builds Debug, and unoptimised Eigen/Ceres is 10-50x slower.
let cxxFlags: [String] = [
    "-O3", "-DNDEBUG", "-g0", "-w",
    "-D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES",       // Eigen 3.3 uses std::binary_function etc.
    "-I\(thirdParty)/eigen",
    "-I\(thirdParty)/opencv/include",
    "-I\(thirdParty)/ceres/config",
    "-I\(thirdParty)/ceres/internal",
    "-I\(thirdParty)/ceres/internal/ceres/miniglog",
    "-I\(thirdParty)/xrslam/src",
    "-I\(thirdParty)/xrslam/extra/include",
]

let package = Package(
    name: "WeeeVR",
    platforms: [.iOS(.v17)],
    products: [
        // An xtool project should contain exactly one library product,
        // representing the main app.
        .library(name: "WeeeVR", targets: ["WeeeVR"]),
    ],
    targets: [
        .target(
            name: "WeeeVR",
            dependencies: ["XRSLAMCore"],
            // MediaPipe hand models (scripts/hands/fetch-models.py), mirrored in by scripts/dev.sh.
            resources: [.copy("HandModels")],
            // Camera/ARKit/display-link callbacks arrive on our own threads; Swift 6 strict
            // concurrency checking would fight that design, so compile in Swift 5 mode.
            // Optimised even in xtool's Debug builds: the render thread and hand pipeline are hot.
            swiftSettings: [.swiftLanguageMode(.v5), .unsafeFlags(["-O"])]
        ),
        // Our C interface to XRSLAM (include/XRSLAMCore.h) plus its os_log logging, the hand-model
        // crops and the paddle tag detector.
        .target(
            name: "XRSLAMCore",
            dependencies: ["XRSLAM", "AprilTag"],
            cSettings: [.unsafeFlags(["-O3", "-DNDEBUG", "-g0"])],
            cxxSettings: [.unsafeFlags(cxxFlags)],
            linkerSettings: [
                .unsafeFlags(["-L\(thirdParty)/opencv/lib", "-lopencv2"]),
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreVideo"),
                .linkedLibrary("z"),
                .linkedLibrary("c++"),
            ]
        ),
        // openxrlab/xrslam core (Apache-2.0) + its OpenCV image class.
        .target(
            name: "XRSLAM",
            dependencies: ["CeresSolver"],
            path: "ThirdParty/xrslam",
            sources: ["src", "extra/src"],
            publicHeadersPath: "include",
            cxxSettings: [.unsafeFlags(cxxFlags)]
        ),
        // AprilRobotics apriltag (BSD-2), core + tagStandard41h12: the smart paddle's tag.
        .target(
            name: "AprilTag",
            path: "ThirdParty/apriltag",
            sources: ["src"],
            publicHeadersPath: "include",
            cSettings: [.unsafeFlags(["-O3", "-DNDEBUG", "-g0", "-w"])]
        ),
        // Ceres Solver 1.14 (BSD), used by XRSLAM's sliding-window bundle adjustment.
        .target(
            name: "CeresSolver",
            path: "ThirdParty/ceres",
            sources: ["internal/ceres"],
            publicHeadersPath: "include",
            cxxSettings: [.unsafeFlags(cxxFlags)]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
