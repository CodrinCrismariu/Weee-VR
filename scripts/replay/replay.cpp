// Offline XRSLAM replay of an app recording (Documents/calibration/<name>: frames.bin, gyro.csv,
// accel.csv), through the same C wrapper (WeeeSLAM.cpp) and configuration as the app, so recordings
// from different iOS versions can be compared without the phone in the loop.
//
// usage: xrslam-replay <recording-dir> [--speed 1] [--offset-ms <Kalibr 2.80>] [--start s]
//                      [--blackout from:to ...]
//   --speed 1        real time (XRSLAM's worker threads see the same pacing as on the phone)
//   --start s        skip the first s seconds (start the tracker on a still camera)
//   --blackout a:b   black frames from a to b seconds (camera covered / phone put down); repeatable
// Prints the tracker state every 0.5 s (recording time) and a summary.
#include "XRSLAMCore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Sample {
    double t, x, y, z;
};

std::vector<Sample> load(const std::string &path) {
    std::vector<Sample> out;
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);                     // header
    while (std::getline(in, line)) {
        Sample s{};
        if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf", &s.t, &s.x, &s.y, &s.z) == 4) out.push_back(s);
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <recording-dir> [--speed x] [--offset-ms ms] [--start s] [--blackout a:b]\n",
                     argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    double speed = 1.0, offsetMS = 2.803353317080429, start = 0;
    std::vector<std::pair<double, double>> blackouts;
    std::string csvPath;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--speed") {
            speed = std::atof(argv[i + 1]);
        } else if (flag == "--offset-ms") {
            offsetMS = std::atof(argv[i + 1]);
        } else if (flag == "--start") {
            start = std::atof(argv[i + 1]);
        } else if (flag == "--blackout") {
            double a = 0, b = 0;
            std::sscanf(argv[i + 1], "%lf:%lf", &a, &b);
            blackouts.emplace_back(a, b);
        } else if (flag == "--csv") {
            csvPath = argv[i + 1];
        }
    }

    // native/WeeeVR/Sources/WeeeVR/UltraWideCalibration.swift (Kalibr, iOS 26) + XRSLAMSource.create
    WeeeSLAMConfig c{};
    c.width = 640;
    c.height = 480;
    c.fx = 257.9145215410328;
    c.fy = 257.8673064412606;
    c.cx = 319.5941429924307;
    c.cy = 239.9762933638708;
    const double d[4] = {0.0034029127124460606, -0.0014675942563807136, 0.00024581282280963855,
                         0.0002738096442588946};
    std::memcpy(c.distortion, d, sizeof d);
    const double q[4] = {-0.7065187261265476, 0.7076940406874447, 0.0001866749548838519, 0.0006321079488362277};
    std::memcpy(c.q_bc, q, sizeof q);
    const double p[3] = {0.03260575770312041, 0.017402273216762406, -0.0047772004282071205};
    std::memcpy(c.p_bc, p, sizeof p);
    c.time_offset = offsetMS / 1000;
    c.window_size = 10;
    c.tracker_frequent = 3;
    c.max_keypoints = 300;
    c.min_keypoint_distance = 25;
    c.solver_iterations = 10;
    c.solver_time_limit = 0.1;

    auto gyro = load(dir + "/gyro.csv"), accel = load(dir + "/accel.csv");
    std::ifstream frames(dir + "/frames.bin", std::ios::binary);
    if (gyro.empty() || accel.empty() || !frames) {
        std::fprintf(stderr, "missing gyro.csv / accel.csv / frames.bin in %s\n", dir.c_str());
        return 1;
    }
    WeeeSLAM *slam = weee_slam_create(&c);
    if (!slam) return 1;
    // Relocalization step at 4 Hz on its own thread, as in the app (XRSLAMSource).
    std::atomic<bool> done{false};
    std::thread reloc([&] {
        while (!done) {
            weee_slam_reloc_step(slam);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    });
    FILE *csv = csvPath.empty() ? nullptr : std::fopen(csvPath.c_str(), "w");
    if (csv) std::fprintf(csv, "t,status,aligned,map,session,x,y,z\n");

    size_t gi = 0, ai = 0;
    std::vector<uint8_t> pixels;
    std::minstd_rand noise(1);
    const auto wall0 = std::chrono::steady_clock::now();
    double t0 = -1, recordingStart = -1, nextPrint = 0, maxSpeed = 0, path = 0, trackedTime = 0, lastT = 0;
    double last[3] = {0, 0, 0};
    bool hadLast = false;
    int frameCount = 0, statusCounts[4] = {0, 0, 0, 0}, trackingEnded = 0, lastStatus = -1;
    while (true) {
        double t;
        uint32_t w, h;
        if (!frames.read(reinterpret_cast<char *>(&t), 8) || !frames.read(reinterpret_cast<char *>(&w), 4) ||
            !frames.read(reinterpret_cast<char *>(&h), 4))
            break;
        pixels.resize(size_t(w) * h);
        frames.read(reinterpret_cast<char *>(pixels.data()), pixels.size());
        if (recordingStart < 0) recordingStart = t;
        const double rel = t - recordingStart;          // all printed times are recording time
        if (rel < start) {
            while (gi < gyro.size() && gyro[gi].t <= t) gi++;
            while (ai < accel.size() && accel[ai].t <= t) ai++;
            continue;
        }
        for (const auto &[a, b] : blackouts) {
            // A covered lens: dark sensor noise (a flat image would let KLT "track" every point).
            if (rel >= a && rel < b) {
                for (auto &px : pixels) px = uint8_t(4 + noise() % 9);
            }
        }
        if (t0 < 0) {
            t0 = t;
            nextPrint = rel;
        }
        if (speed > 0) {
            auto due = wall0 + std::chrono::duration<double>((t - t0) / speed);
            std::this_thread::sleep_until(due);
        }
        // On the phone a frame reaches the tracker ~15 ms after capture; IMU up to then is in.
        while (gi < gyro.size() && gyro[gi].t <= t + 0.015) {
            weee_slam_push_gyro(slam, gyro[gi].t, gyro[gi].x, gyro[gi].y, gyro[gi].z);
            gi++;
        }
        while (ai < accel.size() && accel[ai].t <= t + 0.015) {
            const double g = -9.80665;
            weee_slam_push_accel(slam, accel[ai].t, accel[ai].x * g, accel[ai].y * g, accel[ai].z * g);
            ai++;
        }
        weee_slam_push_image(slam, t, pixels.data(), int(w), int(h), int(w));
        frameCount++;

        WeeeSLAMState s{};
        weee_slam_get_state(slam, &s);
        statusCounts[std::min(std::max(int(s.status), 0), 3)]++;
        if (lastStatus == WEEE_SLAM_TRACKING && s.status != WEEE_SLAM_TRACKING) {
            trackingEnded++;
            std::printf("t %6.2f  -- tracking ended (status %d)\n", rel, int(s.status));
        }
        lastStatus = s.status;
        const double v = std::sqrt(s.v_wb[0] * s.v_wb[0] + s.v_wb[1] * s.v_wb[1] + s.v_wb[2] * s.v_wb[2]);
        if (s.has_pose && s.status == WEEE_SLAM_TRACKING) {
            maxSpeed = std::max(maxSpeed, v);
            if (hadLast) {
                path += std::sqrt(std::pow(s.p_wc[0] - last[0], 2) + std::pow(s.p_wc[1] - last[1], 2) +
                                  std::pow(s.p_wc[2] - last[2], 2));
                trackedTime += t - lastT;
            }
            std::memcpy(last, s.p_wc, sizeof last);
            hadLast = true;
        } else {
            hadLast = false;
        }
        lastT = t;
        if (csv) {
            std::fprintf(csv, "%.4f,%d,%d,%llu,%llu,%.4f,%.4f,%.4f\n", rel, int(s.status), s.aligned,
                         (unsigned long long)s.map_id, (unsigned long long)s.session_id, s.p_wc[0], s.p_wc[1],
                         s.p_wc[2]);
        }
        if (rel >= nextPrint) {
            nextPrint += 0.5;
            std::printf("t %6.2f  status %d  pos %7.3f %7.3f %7.3f  |v| %6.3f  lm %3d  depth %5.2f  map %llu/%llu %s kf %d\n",
                        rel, int(s.status), s.p_wc[0], s.p_wc[1], s.p_wc[2], v, int(s.landmarks), s.depth_median,
                        (unsigned long long)s.session_id, (unsigned long long)s.map_id,
                        s.aligned ? "aligned" : "UNALIGNED", s.keyframes);
            std::fflush(stdout);
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    done = true;
    reloc.join();
    if (csv) std::fclose(csv);
    WeeeSLAMState s{};
    weee_slam_get_state(slam, &s);
    std::printf("SUMMARY %s: %d frames, status counts init/track/lost/other %d/%d/%d/%d, tracked %.1f s, "
                "tracking ended %d times, relocalized %d, keyframes %d, session %llu, path %.2f m, max |v| %.2f m/s, "
                "final pos %.3f %.3f %.3f\n",
                dir.c_str(), frameCount, statusCounts[0], statusCounts[1], statusCounts[2], statusCounts[3],
                trackedTime, trackingEnded, s.relocalizations, s.keyframes, (unsigned long long)s.session_id, path,
                maxSpeed, s.p_wc[0], s.p_wc[1], s.p_wc[2]);
    weee_slam_destroy(slam);
    return 0;
}
