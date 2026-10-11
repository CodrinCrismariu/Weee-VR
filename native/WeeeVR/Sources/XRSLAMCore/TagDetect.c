// The smart paddle's AprilTag: detection (AprilRobotics apriltag, tagStandard41h12) in a region of
// the camera's luma plane, then the tag pose from undistorted corners.

#include "XRSLAMCore.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "apriltag.h"
#include "apriltag_pose.h"
#include "tagStandard41h12.h"

struct WeeeTagDetector {
    apriltag_detector_t *detector;
    apriltag_family_t *family;
    int last_count;
    int last_ids[8];
};

int weee_tag_last_ids(const WeeeTagDetector *d, int *ids, int max) {
    if (!d) return 0;
    for (int i = 0; i < d->last_count && i < max && i < 8; i++) ids[i] = d->last_ids[i];
    return d->last_count;
}

WeeeTagDetector *weee_tag_create(int threads) {
    WeeeTagDetector *d = calloc(1, sizeof(WeeeTagDetector));
    d->family = tagStandard41h12_create();
    d->detector = apriltag_detector_create();
    // One corrected bit: the two-bit table for this 2115-code family is tens of MB, and only one
    // ID is accepted anyway.
    apriltag_detector_add_family_bits(d->detector, d->family, 1);
    d->detector->nthreads = threads;
    d->detector->quad_sigma = 0.0f;
    d->detector->refine_edges = true;
    d->detector->decode_sharpening = 0.25;
    return d;
}

void weee_tag_destroy(WeeeTagDetector *d) {
    if (!d) return;
    apriltag_detector_destroy(d->detector);
    tagStandard41h12_destroy(d->family);
    free(d);
}

// Pixel -> undistorted pixel (radtan, iterated as OpenCV's undistortPoints does).
static void undistort(const double k[4], const double dist[4], double *u, double *v) {
    double xd = (*u - k[2]) / k[0], yd = (*v - k[3]) / k[1];
    double x = xd, y = yd;
    for (int i = 0; i < 8; i++) {
        double r2 = x * x + y * y;
        double radial = 1 + dist[0] * r2 + dist[1] * r2 * r2;
        double dx = 2 * dist[2] * x * y + dist[3] * (r2 + 2 * x * x);
        double dy = dist[2] * (r2 + 2 * y * y) + 2 * dist[3] * x * y;
        x = (xd - dx) / radial;
        y = (yd - dy) / radial;
    }
    *u = x * k[0] + k[2];
    *v = y * k[1] + k[3];
}

// The homography from the ideal tag (corners (-1,-1), (1,-1), (1,1), (-1,1), AprilTag's order) to
// the given pixel corners: 8 linear equations, Gaussian elimination with partial pivoting.
static int homography(const double p[4][2], double h[9]) {
    static const double ideal[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    double a[8][9];
    for (int i = 0; i < 4; i++) {
        double x = ideal[i][0], y = ideal[i][1], u = p[i][0], v = p[i][1];
        double r0[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
        double r1[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        memcpy(a[2 * i], r0, sizeof(r0));
        memcpy(a[2 * i + 1], r1, sizeof(r1));
    }
    for (int c = 0; c < 8; c++) {
        int best = c;
        for (int r = c + 1; r < 8; r++)
            if (fabs(a[r][c]) > fabs(a[best][c])) best = r;
        if (fabs(a[best][c]) < 1e-12) return 0;
        if (best != c)
            for (int j = 0; j < 9; j++) { double t = a[c][j]; a[c][j] = a[best][j]; a[best][j] = t; }
        for (int r = 0; r < 8; r++) {
            if (r == c) continue;
            double f = a[r][c] / a[c][c];
            for (int j = c; j < 9; j++) a[r][j] -= f * a[c][j];
        }
    }
    for (int i = 0; i < 8; i++) h[i] = a[i][8] / a[i][i];
    h[8] = 1;
    return 1;
}

// A flat tag projects identically from its mirror pose behind the camera (turned half a turn in its
// plane, position negated), and the solver sometimes lands there: such a pose becomes its in-front twin.
static void copy_pose(const apriltag_pose_t *pose, double rotation[9], double translation[3]) {
    for (int i = 0; i < 9; i++) rotation[i] = pose->R->data[i];
    for (int i = 0; i < 3; i++) translation[i] = pose->t->data[i];
    if (translation[2] < 0) {
        for (int i = 0; i < 3; i++) {
            translation[i] = -translation[i];
            rotation[3 * i] = -rotation[3 * i];             // R * diag(-1, -1, 1): columns 0 and 1
            rotation[3 * i + 1] = -rotation[3 * i + 1];
        }
    }
}

int weee_tag_detect(WeeeTagDetector *d, const uint8_t *image, int stride, int x0, int y0, int w, int h,
                    float decimate, const int *ids, const double *sizes, int id_count, const double camera[4],
                    const double distortion[4], WeeeTagDetection *outs, int max_out) {
    if (!d || w < 16 || h < 16) return 0;
    image_u8_t im = {.width = w, .height = h, .stride = stride,
                     .buf = (uint8_t *)image + (size_t)y0 * stride + x0};
    d->detector->quad_decimate = decimate;
    zarray_t *detections = apriltag_detector_detect(d->detector, &im);
    d->last_count = zarray_size(detections);
    for (int i = 0; i < d->last_count && i < 8; i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        d->last_ids[i] = det->id;
    }

    int found = 0;
    for (int i = 0; i < zarray_size(detections) && found < max_out; i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        int k = 0;
        while (k < id_count && ids[k] != det->id) k++;
        if (k == id_count) continue;
        int duplicate = 0;
        for (int j = 0; j < found; j++) duplicate |= outs[j].id == det->id;
        if (duplicate) continue;
        double tag_size = sizes[k];
        WeeeTagDetection *out = &outs[found];

        // Back to full-image pixels, undistorted.
        double p[4][2];
        for (int c = 0; c < 4; c++) {
            p[c][0] = det->p[c][0] + x0;
            p[c][1] = det->p[c][1] + y0;
            undistort(camera, distortion, &p[c][0], &p[c][1]);
        }
        double hv[9];
        if (!homography(p, hv)) continue;

        apriltag_detection_t fixed = *det;
        memcpy(fixed.p, p, sizeof(p));
        fixed.H = matd_create_data(3, 3, hv);
        apriltag_detection_info_t info = {.det = &fixed, .tagsize = tag_size,
                                          .fx = camera[0], .fy = camera[1], .cx = camera[2], .cy = camera[3]};
        apriltag_pose_t pose1 = {0}, pose2 = {0};
        double err1 = 0, err2 = 0;
        estimate_tag_pose_orthogonal_iteration(&info, &err1, &pose1, &err2, &pose2, 50);

        memset(out, 0, sizeof(*out));
        out->id = det->id;
        out->hamming = det->hamming;
        out->margin = det->decision_margin;
        out->centre[0] = det->c[0] + x0;
        out->centre[1] = det->c[1] + y0;
        undistort(camera, distortion, &out->centre[0], &out->centre[1]);
        for (int c = 0; c < 4; c++) {
            out->corners[2 * c] = p[c][0];
            out->corners[2 * c + 1] = p[c][1];
        }
        if (pose1.R) {
            copy_pose(&pose1, out->rotation[0], out->translation[0]);
            out->error[0] = err1;
            out->solutions = 1;
        }
        if (pose2.R) {
            copy_pose(&pose2, out->rotation[out->solutions], out->translation[out->solutions]);
            out->error[out->solutions] = err2;
            out->solutions++;
        }
        if (pose1.R) matd_destroy(pose1.R);
        if (pose1.t) matd_destroy(pose1.t);
        if (pose2.R) matd_destroy(pose2.R);
        if (pose2.t) matd_destroy(pose2.t);
        matd_destroy(fixed.H);
        if (out->solutions > 0) found++;
    }
    apriltag_detections_destroy(detections);
    return found;
}
