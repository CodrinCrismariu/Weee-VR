#include "Relocalizer.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace weee {

namespace {

constexpr double kNewViewDistance = 0.15;               // m
constexpr double kNewViewAngle = 15 * M_PI / 180;       // rad
constexpr int kBorder = 20;                             // px, ORB needs a margin around a keypoint
constexpr int kMinInliers = 25;
constexpr size_t kMinKeyframePoints = 30;
constexpr double kMinInlierRatio = 0.3;

double angle_between(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
    return Eigen::AngleAxisd(a.transpose() * b).angle();
}

/// Harris corners as XRSLAM detects its features (GFTT, quality 1e-3, Harris), denser, away from the
/// border ORB needs: the same physical corners the stored observations came from.
std::vector<cv::Point2f> detect_corners(const cv::Mat &image) {
    std::vector<cv::Point2f> corners;
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(kBorder, kBorder, image.cols - 2 * kBorder, image.rows - 2 * kBorder)).setTo(255);
    cv::goodFeaturesToTrack(image, corners, 600, 1e-3, 8, mask, 3, true);
    return corners;
}

} // namespace

Relocalizer::Relocalizer(const cv::Matx33d &K) : K(K) {
    // Upright descriptors (angle 0) at single-scale corners for both keyframes and queries: the camera
    // is head-mounted and gravity-aligned, and it returns to a place at a similar distance.
    orb = cv::ORB::create(500, 1.2f, 1, kBorder, 0, 2, cv::ORB::HARRIS_SCORE, 31, 15);
}

void Relocalizer::clear() {
    keyframes.clear();
    next_slot = 0;
}

bool Relocalizer::add_if_new(const cv::Mat &image, const CameraPose &pose, const std::vector<cv::Point2f> &pixels,
                             const std::vector<Eigen::Vector3d> &points) {
    for (const auto &k : keyframes) {
        if ((k.pose.p - pose.p).norm() < kNewViewDistance && angle_between(k.pose.R, pose.R) < kNewViewAngle) {
            return false;
        }
    }

    // Descriptors where the tracker observed each landmark (exact 2D-3D pairs of this frame).
    std::vector<cv::KeyPoint> keypoints;
    for (size_t i = 0; i < pixels.size() && i < points.size(); ++i) {
        const auto &u = pixels[i];
        if (u.x < kBorder || u.y < kBorder || u.x >= image.cols - kBorder || u.y >= image.rows - kBorder) continue;
        keypoints.emplace_back(u, 31.f, 0.f, 0.f, 0, int(i));
    }
#ifdef XRSLAM_REPLAY_TRACE
    fprintf(stderr, "KEYFRAME observations=%zu usable=%zu\n", pixels.size(), keypoints.size());
#endif
    if (keypoints.size() < kMinKeyframePoints) return false;
    if (keypoints.size() > kMaxPoints) {
        // Spread evenly rather than keeping the first ones.
        std::vector<cv::KeyPoint> thinned;
        for (size_t i = 0; i < kMaxPoints; ++i) thinned.push_back(keypoints[i * keypoints.size() / kMaxPoints]);
        keypoints.swap(thinned);
    }

    Keyframe keyframe;
    keyframe.pose = pose;
    orb->compute(image, keypoints, keyframe.descriptors);      // may drop keypoints; class_id maps back
    if (keypoints.size() < kMinKeyframePoints) return false;
    keyframe.points.reserve(keypoints.size());
    for (const auto &kp : keypoints) {
        const auto &p = points[size_t(kp.class_id)];
        keyframe.points.emplace_back(float(p.x()), float(p.y()), float(p.z()));
    }

    if (keyframes.size() < kMaxKeyframes) {
        keyframes.push_back(std::move(keyframe));
    } else {
        keyframes[next_slot] = std::move(keyframe);
        next_slot = (next_slot + 1) % kMaxKeyframes;
    }
    return true;
}

std::optional<CameraPose> Relocalizer::locate(const cv::Mat &image, int *inliers_out) {
    if (inliers_out) *inliers_out = 0;
    if (keyframes.empty()) return std::nullopt;
    std::vector<cv::KeyPoint> keypoints;
    for (const auto &corner : detect_corners(image)) keypoints.emplace_back(corner, 31.f, 0.f, 0.f, 0);
    cv::Mat descriptors;
    orb->compute(image, keypoints, descriptors);
    if (keypoints.size() < size_t(kMinInliers)) return std::nullopt;

    // Match per keyframe: overlapping keyframes hold the same physical point several times, so a
    // ratio test over the whole database would reject exactly the good matches. The (up to) three
    // keyframes with the most matches are combined, each query point once (its closest match).
    // A pair counts when it passes the ratio test and is mutual (the keyframe point's best query
    // match is that query point): with ~100 points per keyframe the ratio test alone lets through
    // many chance matches.
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> good(keyframes.size()), knn;
    std::vector<cv::DMatch> reverse;
    for (size_t k = 0; k < keyframes.size(); ++k) {
        knn.clear();                                     // knnMatch/match append
        reverse.clear();
        matcher.knnMatch(descriptors, keyframes[k].descriptors, knn, 2);
        matcher.match(keyframes[k].descriptors, descriptors, reverse);
        for (const auto &m : knn) {
            if (m.size() == 2 && m[0].distance <= 50 && m[0].distance < 0.8f * m[1].distance &&
                reverse[size_t(m[0].trainIdx)].trainIdx == m[0].queryIdx) {
                good[k].push_back(m[0]);
            }
        }
    }
    std::vector<size_t> order(keyframes.size());
    for (size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return good[a].size() > good[b].size(); });

    std::vector<int> slot_of_query(keypoints.size(), -1);
    std::vector<cv::Point3f> object;
    std::vector<cv::Point2f> pixels;
    std::vector<float> distances;
    for (size_t r = 0; r < std::min<size_t>(3, order.size()); ++r) {
        for (const auto &m : good[order[r]]) {
            const cv::Point3f &point = keyframes[order[r]].points[size_t(m.trainIdx)];
            int &slot = slot_of_query[size_t(m.queryIdx)];
            if (slot < 0) {
                slot = int(object.size());
                object.push_back(point);
                pixels.push_back(keypoints[size_t(m.queryIdx)].pt);
                distances.push_back(m.distance);
            } else if (m.distance < distances[size_t(slot)]) {
                object[size_t(slot)] = point;
                distances[size_t(slot)] = m.distance;
            }
        }
    }
#ifdef XRSLAM_REPLAY_TRACE
    fprintf(stderr, "LOCATE keyframes=%zu query=%zu best=%zu,%zu,%zu combined=%zu\n", keyframes.size(),
            keypoints.size(), good[order[0]].size(), order.size() > 1 ? good[order[1]].size() : 0,
            order.size() > 2 ? good[order[2]].size() : 0, object.size());
#endif
    if (object.size() < size_t(kMinInliers)) return std::nullopt;

    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    if (!cv::solvePnPRansac(object, pixels, cv::Mat(K), cv::noArray(), rvec, tvec, false, 200, 3.0f, 0.99,
                            inliers, cv::SOLVEPNP_EPNP)) {
        return std::nullopt;
    }
    if (inliers_out) *inliers_out = int(inliers.size());
    if (int(inliers.size()) < kMinInliers || double(inliers.size()) < kMinInlierRatio * double(object.size())) {
        return std::nullopt;
    }
    // Refine on the inliers.
    std::vector<cv::Point3f> object_in;
    std::vector<cv::Point2f> pixels_in;
    for (int i : inliers) {
        object_in.push_back(object[size_t(i)]);
        pixels_in.push_back(pixels[size_t(i)]);
    }
    cv::solvePnP(object_in, pixels_in, cv::Mat(K), cv::noArray(), rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    cv::Matx33d R_cw;
    cv::Rodrigues(rvec, R_cw);
    CameraPose pose;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) pose.R(r, c) = R_cw(c, r);          // R_wc = R_cw^T
    Eigen::Vector3d t(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    pose.p = -pose.R * t;
    return pose;
}

} // namespace weee
