// In-session relocalization: a bounded database of keyframes (ORB descriptors where XRSLAM observed
// its triangulated landmarks, with their 3D positions in the session world) and camera localization
// against it (Harris corners like XRSLAM's detector, descriptor matching, PnP RANSAC). Used after a tracking loss to put the new XRSLAM
// map back into the world the user was in, so virtual content returns to where it was.
//
// Bounds: at most `kMaxKeyframes` keyframes of at most `kMaxPoints` points (~0.5 MB). Keyframes are
// added only for new viewpoints (no stored one within 15 cm and 15 deg), the oldest is replaced
// when full. Not thread-safe: one owner thread (the wrapper's relocalization step).
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>
#include <vector>

namespace weee {

struct CameraPose {                 // world <- camera, OpenCV camera axes
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

class Relocalizer {
  public:
    static constexpr size_t kMaxKeyframes = 80;
    static constexpr size_t kMaxPoints = 150;

    explicit Relocalizer(const cv::Matx33d &K);

    void clear();
    size_t keyframe_count() const { return keyframes.size(); }

    // Store a keyframe if no stored one sees from nearly the same viewpoint. `image` is the
    // undistorted 8-bit gray camera image and `pose` its camera pose; `pixels[i]` is where the tracker
    // observed landmark `points[i]` in it. Pose and points in the session world. Returns true when added.
    bool add_if_new(const cv::Mat &image, const CameraPose &pose, const std::vector<cv::Point2f> &pixels,
                    const std::vector<Eigen::Vector3d> &points);

    // Camera pose of `image` in the session world, if it shows enough of the stored keyframes.
    std::optional<CameraPose> locate(const cv::Mat &image, int *inliers = nullptr);

  private:
    struct Keyframe {
        CameraPose pose;
        cv::Mat descriptors;                     // N x 32 (ORB)
        std::vector<cv::Point3f> points;         // N, session world
    };

    cv::Matx33d K;
    cv::Ptr<cv::ORB> orb;
    std::vector<Keyframe> keyframes;
    size_t next_slot = 0;                        // ring position once full
};

} // namespace weee
