/// Metal shaders, compiled at runtime with `makeLibrary(source:)` because Apple's offline
/// `metal` compiler only ships with Xcode on macOS (we build on Linux/WSL with xtool).
///
/// One full-screen triangle; the fragment shader does everything per pixel:
///   screen pixel -> lens tan-angle -> Cardboard pincushion (k1, k2) -> world ray from this eye's real
///   position at display time -> hit the depth-proxy sphere (radius D around the eyes) -> project that
///   point into the camera at its capture pose -> YCbCr sample (sharpened, per colour channel against
///   the lens's chromatic aberration, converted for the P3 screen).
/// Viewing the camera image from each eye's own position corrects the camera/eye offset, gives the
/// passthrough stereo disparity, and reprojects head rotation *and* translation, all exact at depth D.
/// A world-locked grid, a stereo wireframe cube, tracked 3D hands and the HUD are drawn in the same pass.
/// Keep `Uniforms` in sync with the Swift struct of the same name (all float4 for identical layout).
enum ShaderSource {
    static let metal = """
    #include <metal_stdlib>
    using namespace metal;

    struct Uniforms {
        float4 screen;      // drawable width px, height px, pixels per metre, screen-to-lens metres
        float4 lens;        // left lens centre x px, lens centre y px, right lens centre x px, max tan-angle
        float4 optics;      // k1, k2, fovScale, flags (1 grid, 2 cube, 4 hud, 8 hands, 16 game)
        float4 camK;        // fx, fy, cx, cy in camera-buffer pixels
        float4 cam;         // buffer width, buffer height, ipd metres, flip (0/1)
        float4 camRot0, camRot1, camRot2;   // camera <- world at capture time (columns)
        float4 camPos;      // camera position in world at capture time
        float4 world0, world1, world2;      // world <- view at display time (columns)
        float4 head;        // eye midpoint in world at display time; w = depth-proxy distance D (m)
        float4 cube;        // world centre xyz, half size
        float4 hud;         // HUD top-left offset from each lens centre (px), HUD width, height (px)
        float4 image;       // YCbCr matrix (0 BT.601, 1 BT.709), sharpening, black lift, lens-edge softness
        float4 chroma;      // red, blue tan-angle scale - 1 (lens chromatic aberration), sRGB -> P3 (0/1), 0
        float4 recording;   // flat view (game recording): 1, tan half-angles x, y, 0 -- else 0
    };

    struct VOut { float4 position [[position]]; };

    vertex VOut vrVertex(uint vid [[vertex_id]]) {
        float2 p = float2((vid << 1) & 2, vid & 2);
        VOut o;
        o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
        return o;
    }

    // Hands: HandModel.stride float4 per hand -- [0] valid, bone radius scale, highlight, chirality;
    // [1] world bounding sphere; [2..22] world joints (w = valid).
    constant int kHandStride = 23;
    constant int2 kBones[23] = {
        int2(0, 1), int2(1, 2), int2(2, 3), int2(3, 4),
        int2(0, 5), int2(5, 6), int2(6, 7), int2(7, 8),
        int2(0, 9), int2(9, 10), int2(10, 11), int2(11, 12),
        int2(0, 13), int2(13, 14), int2(14, 15), int2(15, 16),
        int2(0, 17), int2(17, 18), int2(18, 19), int2(19, 20),
        int2(5, 9), int2(9, 13), int2(13, 17) };

    // Ray / capsule intersection (Inigo Quilez); returns t or -1.
    static float capsuleHit(float3 ro, float3 rd, float3 pa, float3 pb, float r) {
        float3 ba = pb - pa, oa = ro - pa;
        float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa);
        float rdoa = dot(rd, oa), oaoa = dot(oa, oa);
        float a = baba - bard * bard;
        float b = baba * rdoa - baoa * bard;
        float c = baba * oaoa - baoa * baoa - r * r * baba;
        float h = b * b - a * c;
        if (h >= 0.0) {
            float t = (-b - sqrt(h)) / a;
            float y = baoa + t * bard;
            if (y > 0.0 && y < baba) return t;
            float3 oc = (y <= 0.0) ? oa : ro - pb;
            b = dot(rd, oc);
            c = dot(oc, oc) - r * r;
            h = b * b - c;
            if (h > 0.0) return -b - sqrt(h);
        }
        return -1.0;
    }

    // Camera texture coordinate of world point `p` (false: behind the camera).
    static bool projectToCamera(float3 p, constant Uniforms& u, float3x3 camFromWorld, thread float2& uv) {
        float3 camRay = camFromWorld * (p - u.camPos.xyz);
        if (camRay.z >= -1e-3) return false;
        float2 tanCam = camRay.xy / -camRay.z;
        uv = float2(u.camK.x * tanCam.x + u.camK.z, u.camK.w - u.camK.y * tanCam.y) / u.cam.xy;
        if (u.cam.w > 0.5) { uv = 1.0 - uv; }
        return true;
    }

    // The eye's ray (world) through eye-space direction `tanEye`.
    static float3 eyeRay(float2 tanEye, constant Uniforms& u, float3x3 worldFromView) {
        return normalize(worldFromView * normalize(float3(tanEye * u.optics.z, -1.0)));
    }

    // Distance along the eye's ray to the depth-proxy sphere around the eyes (no depth map).
    static float sphereDistance(float3 eye, float3 ray, constant Uniforms& u) {
        float3 rel = eye - u.head.xyz;
        float b = dot(ray, rel);
        float c = dot(rel, rel) - u.head.w * u.head.w;
        return -b + sqrt(max(b * b - c, 0.0));
    }

    // Camera luma at `uv`. With `amount` > 0: Catmull-Rom bicubic (5 bilinear taps, corners dropped)
    // plus an unsharp step against a 3x3 tent blur (4 diagonal bilinear taps), limited to the local
    // range so edges get no halos. Undoes some of the blur of resampling the image twice (the ISP's
    // lens correction, then this warp).
    static float lumaAt(texture2d<float> tex, sampler s, float2 uv, float2 size, float amount) {
        if (amount <= 0.0) return tex.sample(s, uv).r;
        float2 pos = uv * size;
        float2 centre = floor(pos - 0.5) + 0.5;
        float2 f = pos - centre;
        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        float2 w3 = f * f * (-0.5 + 0.5 * f);
        float2 w12 = w1 + w2;
        float2 t0 = (centre - 1.0) / size, t3 = (centre + 2.0) / size, t12 = (centre + w2 / w12) / size;
        float top = w12.x * w0.y, left = w0.x * w12.y, mid = w12.x * w12.y, right = w3.x * w12.y, bottom = w12.x * w3.y;
        float cubic = (tex.sample(s, float2(t12.x, t0.y)).r * top + tex.sample(s, float2(t0.x, t12.y)).r * left
                       + tex.sample(s, t12).r * mid + tex.sample(s, float2(t3.x, t12.y)).r * right
                       + tex.sample(s, float2(t12.x, t3.y)).r * bottom) / (top + left + mid + right + bottom);

        float2 d = 0.5 / size;
        float a = tex.sample(s, uv + float2(-d.x, -d.y)).r, b = tex.sample(s, uv + float2(d.x, -d.y)).r;
        float c = tex.sample(s, uv + float2(-d.x, d.y)).r, e = tex.sample(s, uv + float2(d.x, d.y)).r;
        float blur = (a + b + c + e) * 0.25;
        float lo = min(min(min(a, b), min(c, e)), cubic), hi = max(max(max(a, b), max(c, e)), cubic);
        return clamp(cubic + amount * (cubic - blur), lo, hi);
    }

    // Full-range YCbCr -> RGB with the buffer's matrix, then (optionally) sRGB/BT.709 primaries ->
    // Display P3: the layer is not colour managed, so its values go to the P3 panel as they are, and
    // unconverted video looks oversaturated there.
    static float3 cameraRGB(texture2d<float> lumaTex, texture2d<float> chromaTex, sampler s, float2 uv,
                            constant Uniforms& u) {
        float y = lumaAt(lumaTex, s, uv, u.cam.xy, u.image.y);
        float2 cbcr = chromaTex.sample(s, uv).rg - 128.0 / 255.0;
        float3 rgb = u.image.x > 0.5
            ? float3(y + 1.5748 * cbcr.y, y - 0.187324 * cbcr.x - 0.468124 * cbcr.y, y + 1.8556 * cbcr.x)
            : float3(y + 1.402 * cbcr.y, y - 0.344136 * cbcr.x - 0.714136 * cbcr.y, y + 1.772 * cbcr.x);
        rgb = saturate(rgb);
        if (u.chroma.z > 0.5) {
            // Linear light approximated by gamma 2 (a square and a square root instead of the sRGB
            // curve's powers, per pixel): greys stay exact, saturated colours within a percent or so.
            float3 lin = float3x3(float3(0.8225, 0.0332, 0.0171), float3(0.1774, 0.9669, 0.0724),
                                  float3(0.0, 0.0, 0.9108)) * (rgb * rgb);
            rgb = sqrt(max(lin, 0.0));
        }
        return rgb;
    }

    // ---- Ping pong (Game/PingPong.sceneData), all in the table frame (Game/Table.swift): origin at
    // the centre of the playing surface, y up, +z toward the player.
    // scene[0] origin xyz, yaw | [1] cos yaw, sin yaw, floor y, flags (1 ball, 2 player paddle,
    // 4 opponent paddle, 8 stickman) | [2] ball xyz, radius | [3] light direction, w scoreboard |
    // [4..7], [8..11] paddles: blade centre + visible, x, y (toward the tip), z (forehand face) axes |
    // [16..28] the stickman's joints (Game/Stickman.swift; [16].w the head's radius).
    constant float kTableHalfLength = 1.37;
    constant float kTableHalfWidth = 0.7625;
    constant float kTableThickness = 0.025;
    constant float kNetHeight = 0.1525;
    constant float kNetHalfSpan = 0.915;
    constant float kPaddleA = 0.075;            // blade semi-axes and half thickness
    constant float kPaddleB = 0.079;
    constant float kPaddleH = 0.006;
    constant float kHandleLength = 0.10;
    constant float kHandleHalfWidth = 0.0125;
    constant float kHandleHalfThickness = 0.011;

    // The scoreboard: in the net's plane, beside the table above the left net post, facing the player.
    constant float kScoreX0 = -1.135;
    constant float kScoreW = 0.44;
    constant float kScoreY0 = 0.21;
    constant float kScoreH = 0.22;

    constant int kTableTop = 0;
    constant int kTableFrame = 1;
    constant int kPost = 2;
    constant int kBall = 3;
    constant int kBlade = 4;
    constant int kHandle = 5;
    constant int kStick = 6;

    // The stickman: limbs as capsules between its joints (scene[16 + i]), and a round head.
    constant int kStickBase = 16;
    constant int2 kStickBones[12] = {
        int2(1, 0), int2(1, 2), int2(1, 3), int2(3, 4), int2(4, 5), int2(1, 6), int2(6, 7), int2(7, 8),
        int2(2, 9), int2(9, 10), int2(2, 11), int2(11, 12) };

    struct SceneHit {
        float t;
        float3 n;               // table frame
        int object;
        int which;              // paddle: 0 player, 1 opponent
        float3 local;           // hit point in the object's frame (paddle parts: paddle frame)
        float coverage;         // edge anti-aliasing: covered fraction of the pixel
    };

    // Ray / box: entry distance (> 0, -1 if missed or inside) and the entry face's normal.
    static float boxHit(float3 ro, float3 rd, float3 lo, float3 hi, thread float3& n) {
        float3 inv = 1.0 / rd;
        float3 t0 = (lo - ro) * inv;
        float3 t1 = (hi - ro) * inv;
        float3 tmin = min(t0, t1);
        float3 tmax = max(t0, t1);
        float tn = max(max(tmin.x, tmin.y), tmin.z);
        float tf = min(min(tmax.x, tmax.y), tmax.z);
        if (tn > tf || tn <= 0.0) return -1.0;
        if (tn == tmin.x) {
            n = float3(-sign(rd.x), 0.0, 0.0);
        } else if (tn == tmin.y) {
            n = float3(0.0, -sign(rd.y), 0.0);
        } else {
            n = float3(0.0, 0.0, -sign(rd.z));
        }
        return tn;
    }

    static bool boxAny(float3 ro, float3 rd, float3 lo, float3 hi) {
        float3 inv = 1.0 / rd;
        float3 t0 = (lo - ro) * inv;
        float3 t1 = (hi - ro) * inv;
        float3 tmin = min(t0, t1);
        float3 tmax = max(t0, t1);
        float tn = max(max(tmin.x, tmin.y), tmin.z);
        float tf = min(min(tmax.x, tmax.y), tmax.z);
        return tf >= max(tn, 0.0);
    }

    static void keepBox(float3 ro, float3 rd, float3 lo, float3 hi, int object, thread SceneHit& best) {
        float3 n;
        float t = boxHit(ro, rd, lo, hi, n);
        if (t > 0.0 && t < best.t) {
            best.t = t;
            best.n = n;
            best.object = object;
            best.which = -1;
            best.local = ro + rd * t;
            best.coverage = 1.0;
        }
    }

    // Table: playing surface, under-frame, legs, cross bars, net posts. `rd` with no zero components.
    static void tableHit(float3 ro, float3 rd, float floorY, thread SceneHit& best) {
        if (!boxAny(ro, rd, float3(-0.95, floorY, -1.45), float3(0.95, 0.2, 1.45))) return;
        keepBox(ro, rd, float3(-kTableHalfWidth, -kTableThickness, -kTableHalfLength),
                float3(kTableHalfWidth, 0.0, kTableHalfLength), kTableTop, best);
        keepBox(ro, rd, float3(-kTableHalfWidth + 0.06, -0.10, -kTableHalfLength + 0.06),
                float3(kTableHalfWidth - 0.06, -kTableThickness, kTableHalfLength - 0.06), kTableFrame, best);
        for (int i = 0; i < 4; i++) {
            float x = (i & 1) ? kTableHalfWidth - 0.12 : -(kTableHalfWidth - 0.12);
            float z = (i & 2) ? kTableHalfLength - 0.30 : -(kTableHalfLength - 0.30);
            keepBox(ro, rd, float3(x - 0.02, floorY, z - 0.02), float3(x + 0.02, -0.10, z + 0.02), kTableFrame, best);
        }
        for (int i = 0; i < 2; i++) {
            float z = i == 0 ? kTableHalfLength - 0.30 : -(kTableHalfLength - 0.30);
            keepBox(ro, rd, float3(-kTableHalfWidth + 0.12, -0.62, z - 0.012),
                    float3(kTableHalfWidth - 0.12, -0.58, z + 0.012), kTableFrame, best);
        }
        for (int i = 0; i < 2; i++) {
            float x = i == 0 ? kNetHalfSpan : -kNetHalfSpan;
            keepBox(ro, rd, float3(x - 0.012, -0.03, -0.012), float3(x + 0.012, kNetHeight + 0.012, 0.012), kPost, best);
        }
    }

    // Ball, with an anti-aliased silhouette: rays passing within half a pixel of it count partly.
    static void ballHit(float3 ro, float3 rd, float4 ball, float pixel, thread SceneHit& best) {
        float3 oc = ro - ball.xyz;
        float tca = -dot(oc, rd);
        if (tca <= 0.0) return;
        float m2 = max(dot(oc, oc) - tca * tca, 0.0);
        float cover = saturate((ball.w - sqrt(m2)) / (pixel * tca) + 0.5);
        if (cover <= 0.0) return;
        float t = tca - sqrt(max(ball.w * ball.w - m2, 0.0));
        if (t >= best.t) return;
        float3 q = ro + rd * t;
        best.t = t;
        best.n = normalize(q - ball.xyz);
        best.object = kBall;
        best.which = -1;
        best.local = q - ball.xyz;
        best.coverage = cover;
    }

    // Paddle at scene[base]: oval blade (an elliptic cylinder, anti-aliased rim) and handle box.
    static void paddleHit(float3 ro, float3 rd, constant float4* scene, int base, int which, float pixel,
                          thread SceneHit& best) {
        float4 c = scene[base];
        if (c.w < 0.5) return;
        float3 X = scene[base + 1].xyz, Y = scene[base + 2].xyz, Z = scene[base + 3].xyz;
        // Blade semi-axes and handle length (w of the axes; the standard sizes if not given).
        float pa = scene[base + 1].w > 0.0 ? scene[base + 1].w : kPaddleA;
        float pb = scene[base + 2].w > 0.0 ? scene[base + 2].w : kPaddleB;
        float handle = scene[base + 3].w > 0.0 ? scene[base + 3].w : kHandleLength;
        float reach = 0.5 * (max(pa, pb) + pb + handle) + 0.02;
        float3 bc = ro - (c.xyz - Y * (0.5 * (pb + handle) - 0.5 * max(pa, pb)));
        float bb = dot(rd, bc), cc = dot(bc, bc) - reach * reach;
        if (bb * bb - cc < 0.0 || (bb > 0.0 && cc > 0.0)) return;   // misses its bounding sphere
        float3 o = ro - c.xyz;
        float3 lo = float3(dot(o, X), dot(o, Y), dot(o, Z));
        float3 ld = float3(dot(rd, X), dot(rd, Y), dot(rd, Z));
        float2 axes = float2(pa, pb);
        // Faces.
        if (abs(ld.z) > 1e-6) {
            for (int side = 0; side < 2; side++) {
                float zc = side == 0 ? kPaddleH : -kPaddleH;
                float t = (zc - lo.z) / ld.z;
                if (t <= 0.0 || t >= best.t) continue;
                float2 q = lo.xy + ld.xy * t;
                float e = length(q / axes);
                float cover = saturate((1.0 - e) * pa / (pixel * t) + 0.5);
                if (cover <= 0.0) continue;
                best.t = t;
                best.n = side == 0 ? Z : -Z;
                best.object = kBlade;
                best.which = which;
                best.local = float3(q, zc);
                best.coverage = cover;
            }
        }
        // Rim.
        float2 ia = 1.0 / (axes * axes);
        float A = dot(ld.xy * ld.xy, ia);
        float B = 2.0 * dot(lo.xy * ld.xy, ia);
        float C = dot(lo.xy * lo.xy, ia) - 1.0;
        float D = B * B - 4.0 * A * C;
        if (A > 1e-8 && D >= 0.0) {
            float t = (-B - sqrt(D)) / (2.0 * A);
            float3 q = lo + ld * t;
            if (t > 0.0 && t < best.t && abs(q.z) <= kPaddleH) {
                float2 nl = normalize(q.xy * ia);
                best.t = t;
                best.n = X * nl.x + Y * nl.y;
                best.object = kBlade;
                best.which = which;
                best.local = q;
                best.coverage = 1.0;
            }
        }
        // Handle.
        float3 lr = select(ld, float3(1e-6), abs(ld) < 1e-6);
        float3 hn;
        float th = boxHit(lo, lr, float3(-kHandleHalfWidth, -pb - handle, -kHandleHalfThickness),
                          float3(kHandleHalfWidth, -pb + 0.01, kHandleHalfThickness), hn);
        if (th > 0.0 && th < best.t) {
            best.t = th;
            best.n = X * hn.x + Y * hn.y + Z * hn.z;
            best.object = kHandle;
            best.which = which;
            best.local = lo + ld * th;
            best.coverage = 1.0;
        }
    }

    static void stickHit(float3 ro, float3 rd, constant float4* scene, thread SceneHit& best) {
        constant float4* j = scene + kStickBase;
        float3 centre = j[2].xyz + float3(0.0, 0.25, 0.0);
        float3 oc = ro - centre;
        float bb = dot(rd, oc);
        if (bb * bb - (dot(oc, oc) - 1.3 * 1.3) < 0.0) return;      // misses it altogether
        // Head.
        float3 oh = ro - j[0].xyz;
        float hb = dot(rd, oh), hc = dot(oh, oh) - j[0].w * j[0].w, hd = hb * hb - hc;
        if (hd >= 0.0) {
            float t = -hb - sqrt(hd);
            if (t > 0.0 && t < best.t) {
                best.t = t;
                best.n = normalize(ro + rd * t - j[0].xyz);
                best.object = kStick;
                best.which = -1;
                best.local = float3(0.0);
                best.coverage = 1.0;
            }
        }
        // Limbs.
        for (int i = 0; i < 12; i++) {
            float3 a = j[kStickBones[i].x].xyz, b = j[kStickBones[i].y].xyz;
            float r = i < 2 ? 0.026 : 0.021;
            float t = capsuleHit(ro, rd, a, b, r);
            if (t > 0.0 && t < best.t) {
                float3 q = ro + rd * t, ab = b - a;
                best.t = t;
                best.n = normalize(q - (a + ab * saturate(dot(q - a, ab) / max(dot(ab, ab), 1e-8))));
                best.object = kStick;
                best.which = -1;
                best.local = float3(0.0);
                best.coverage = 1.0;
            }
        }
    }

    // Nearest opaque scene hit; `skipWhich`/`skipBall` leave out the object a partly covered edge
    // belongs to (to find what is behind it).
    static SceneHit traceScene(float3 ro, float3 rd, constant float4* scene, float pixel, bool skipBall, int skipWhich) {
        SceneHit best;
        best.t = 1e9;
        best.n = float3(0.0, 1.0, 0.0);
        best.object = -1;
        best.which = -1;
        best.local = float3(0.0);
        best.coverage = 1.0;
        float3 rs = select(rd, float3(1e-6), abs(rd) < 1e-6);
        tableHit(ro, rs, scene[1].z, best);
        int flags = int(scene[1].w);
        if ((flags & 1) && !skipBall) ballHit(ro, rd, scene[2], pixel, best);
        if ((flags & 2) && skipWhich != 0) paddleHit(ro, rd, scene, 4, 0, pixel, best);
        if ((flags & 4) && skipWhich != 1) paddleHit(ro, rd, scene, 8, 1, pixel, best);
        if (flags & 8) stickHit(ro, rd, scene, best);
        return best;
    }

    // Materials and light: one soft key light from above plus a sky/floor ambient; soft shadows of the
    // ball and paddles and the ball's contact darkening on the playing surface.
    static float3 shadeScene(SceneHit h, float3 ro, float3 rd, constant float4* scene, float pixel) {
        float3 L = scene[3].xyz;
        float3 n = h.n;
        float3 q = ro + rd * h.t;
        float3 albedo = float3(0.5);
        float spec = 0.05, shininess = 24.0;
        if (h.object == kTableTop) {
            if (n.y > 0.5) {
                float fp = pixel * h.t;
                float edge = min(kTableHalfWidth - abs(q.x), kTableHalfLength - abs(q.z));
                float line = 1.0 - smoothstep(0.02 - fp, 0.02 + fp, edge);
                line = max(line, 1.0 - smoothstep(0.0015 - fp, 0.0015 + fp, abs(q.x)));
                albedo = mix(float3(0.06, 0.20, 0.42), float3(0.92), line);
                spec = 0.06;
                shininess = 40.0;
            } else {
                albedo = float3(0.05, 0.15, 0.32);
            }
        } else if (h.object == kTableFrame) {
            albedo = float3(0.10, 0.10, 0.11);
            spec = 0.15;
            shininess = 30.0;
        } else if (h.object == kPost) {
            albedo = float3(0.08);
            spec = 0.2;
        } else if (h.object == kBall) {
            albedo = float3(0.95, 0.94, 0.90);
            spec = 0.35;
            shininess = 50.0;
        } else if (h.object == kBlade) {
            if (h.local.z > kPaddleH * 0.99) {
                albedo = float3(0.78, 0.08, 0.07);              // forehand rubber
            } else if (h.local.z < -kPaddleH * 0.99) {
                albedo = float3(0.05, 0.05, 0.055);             // backhand rubber
            } else {
                albedo = float3(0.80, 0.62, 0.40);              // wood edge
            }
            spec = 0.12;
            shininess = 20.0;
        } else if (h.object == kStick) {
            albedo = float3(0.07, 0.07, 0.08);                  // ink black, with a rim light below
            spec = 0.3;
            shininess = 30.0;
        } else {
            albedo = float3(0.72, 0.54, 0.34);                  // handle
            spec = 0.1;
        }

        float shadow = 1.0, ao = 1.0;
        if (h.object == kTableTop && n.y > 0.5) {
            float4 ball = scene[2];
            if (int(scene[1].w) & 1) {
                float height = max(ball.y, 0.0);
                float2 centre = ball.xz - L.xz * (ball.y / L.y);
                float d = length(q.xz - centre);
                shadow *= 1.0 - 0.6 * (1.0 - smoothstep(ball.w * 0.5, ball.w * (1.5 + 2.0 * height) + 0.003, d))
                    / (1.0 + 6.0 * height);
                float3 toBall = ball.xyz - q;
                float d2 = max(dot(toBall, toBall), 1e-6);
                ao *= 1.0 - 0.5 * saturate(ball.w * ball.w / d2) * saturate(toBall.y * rsqrt(d2));
            }
            for (int i = 0; i < 2; i++) {
                float4 c = scene[4 + 4 * i];
                if (c.w < 0.5 || c.y < 0.0) continue;
                float2 centre = c.xz - L.xz * (c.y / L.y);
                float d = length(q.xz - centre);
                shadow *= 1.0 - 0.35 * (1.0 - smoothstep(0.04, 0.10 + 0.25 * c.y, d)) / (1.0 + 3.0 * c.y);
            }
        }
        float3 ambient = mix(float3(0.32, 0.30, 0.28), float3(0.58, 0.60, 0.64), 0.5 + 0.5 * n.y);
        float diffuse = max(dot(n, L), 0.0) * shadow;
        float specular = pow(max(dot(n, normalize(L - rd)), 0.0), shininess) * spec * shadow;
        float3 rim = h.object == kStick ? float3(0.35) * pow(1.0 - saturate(dot(n, -rd)), 3.0) : float3(0.0);
        return albedo * (ambient * ao + 0.75 * diffuse) + specular + rim;
    }

    // Soft shadows of the table and the stickman on the (estimated) floor, darkening the passthrough.
    static float floorShadow(float3 ro, float3 rd, constant float4* scene) {
        float floorY = scene[1].z;
        if (rd.y > -1e-4) return 1.0;
        float t = (floorY - ro.y) / rd.y;
        if (t <= 0.0) return 1.0;
        float2 p = (ro + rd * t).xz;
        float2 q = abs(p);
        float d = max(q.x - kTableHalfWidth, q.y - kTableHalfLength);
        float shade = 1.0 - 0.28 * (1.0 - smoothstep(-0.25, 0.30, d));
        if (int(scene[1].w) & 8) {
            float s = length(p - scene[kStickBase + 2].xz);
            shade *= 1.0 - 0.3 * (1.0 - smoothstep(0.08, 0.45, s));
        }
        return shade;
    }

    // The net: distance along the ray (-1: missed) and the hit's (x, height).
    static float netHit(float3 ro, float3 rd, thread float2& q) {
        if (abs(rd.z) < 1e-6) return -1.0;
        float t = -ro.z / rd.z;
        if (t <= 0.0) return -1.0;
        float3 p = ro + rd * t;
        if (abs(p.x) > kNetHalfSpan || p.y < 0.0 || p.y > kNetHeight) return -1.0;
        q = p.xy;
        return t;
    }

    fragment half4 vrFragment(VOut in [[stage_in]],
                              constant Uniforms& u [[buffer(0)]],
                              constant float4* hands [[buffer(1)]],
                              constant float4* scene [[buffer(2)]],
                              texture2d<float> lumaTex [[texture(0)]],
                              texture2d<float> chromaTex [[texture(1)]],
                              texture2d<float> hudTex [[texture(2)]],
                              texture2d<float> scoreTex [[texture(3)]]) {
        constexpr sampler s(address::clamp_to_edge, filter::linear);

        float2 p = in.position.xy;
        // The flat view (game recording): a plain perspective picture from between the eyes, no lenses.
        bool flatView = u.recording.x > 0.5;
        bool rightEye = !flatView && p.x >= u.screen.x * 0.5;
        float2 lensCentre = float2(rightEye ? u.lens.z : u.lens.x, u.lens.y);

        // Screen position as a tan-angle from the lens axis (x right, y up), then the lens's
        // pincushion: what direction the eye actually sees this pixel in.
        float2 tanScreen = float2(p.x - lensCentre.x, lensCentre.y - p.y) / (u.screen.z * u.screen.w);
        float r2 = dot(tanScreen, tanScreen);
        float2 tanEye = tanScreen * (1.0 + u.optics.x * r2 + u.optics.y * r2 * r2);
        // A pixel's size as an angle (edge anti-aliasing).
        float pixelAngle = u.optics.z / (u.screen.z * u.screen.w) * (1.0 + 3.0 * u.optics.x * r2 + 5.0 * u.optics.y * r2 * r2);
        if (flatView) {
            tanEye = float2(p.x / u.screen.x * 2.0 - 1.0, 1.0 - p.y / u.screen.y * 2.0) * u.recording.yz;
            r2 = 0.0;
            pixelAngle = 2.0 * u.recording.z / u.screen.y;
        }
        float3 viewRay = normalize(float3(tanEye * u.optics.z, -1.0));

        float3x3 worldFromView = float3x3(u.world0.xyz, u.world1.xyz, u.world2.xyz);
        float3x3 camFromWorld = float3x3(u.camRot0.xyz, u.camRot1.xyz, u.camRot2.xyz);
        float3 worldRay = normalize(worldFromView * viewRay);
        float3 eye = flatView ? u.head.xyz
                              : u.head.xyz + worldFromView * float3((rightEye ? 0.5 : -0.5) * u.cam.z, 0.0, 0.0);

        int flags = int(u.optics.w);

        // Grid derivatives are taken before any divergent return (a branch on the uniform flags is
        // not divergent), and only when the grid is shown.
        float2 g = float2(0.0), fw = float2(1.0);
        if (flags & 1) {
            float az = atan2(worldRay.x, -worldRay.z);
            float el = asin(clamp(worldRay.y, -1.0, 1.0));
            g = float2(az, el) * (12.0 / M_PI_F);       // 15 degree cells
            fw = max(fwidth(g), float2(1e-4));
            fw.x = min(fw.x, 0.5);                      // atan2 seam behind the viewer
        }

        float radius = length(tanEye);
        if (!flatView && radius > u.lens.w) {
            return half4(0.0, 0.0, 0.0, 1.0);
        }

        // Passthrough. The lens bends red, green and blue slightly differently (chromatic aberration),
        // so with a correction set each channel is looked up along its own direction; the overlays
        // below use the green one.
        float3 col = float3(0.05);
        float3 rayGreen = eyeRay(tanEye, u, worldFromView);
        float seen = sphereDistance(eye, rayGreen, u);
        float2 uv;
        bool green = projectToCamera(eye + rayGreen * seen, u, camFromWorld, uv) && all(uv >= 0.0) && all(uv <= 1.0);
        if (all(u.chroma.xy == 0.0)) {
            if (green) { col = cameraRGB(lumaTex, chromaTex, s, uv, u); }
        } else {
            // The red and blue rays use the green ray's distance to the sphere.
            if (green) { col.g = cameraRGB(lumaTex, chromaTex, s, uv, u).g; }
            float3 rayRed = eyeRay(tanEye * (1.0 + u.chroma.x), u, worldFromView);
            if (projectToCamera(eye + rayRed * seen, u, camFromWorld, uv) && all(uv >= 0.0) && all(uv <= 1.0)) {
                col.r = cameraRGB(lumaTex, chromaTex, s, uv, u).r;
            }
            float3 rayBlue = eyeRay(tanEye * (1.0 + u.chroma.y), u, worldFromView);
            if (projectToCamera(eye + rayBlue * seen, u, camFromWorld, uv) && all(uv >= 0.0) && all(uv <= 1.0)) {
                col.b = cameraRGB(lumaTex, chromaTex, s, uv, u).b;
            }
        }

        if (flags & 1) {
            float2 d = abs(fract(g - 0.5) - 0.5) / fw;
            float line = 1.0 - saturate(min(d.x, d.y) - 0.5);
            float horizon = 1.0 - saturate(abs(g.y) / fw.y - 1.0);
            col = mix(col, float3(0.1, 1.0, 0.45), line * 0.5);
            col = mix(col, float3(1.0, 0.8, 0.1), horizon);
        }

        if (flags & 2) {
            // Stereo: each eye casts from its own position, so the cube has real depth.
            float3 ro = eye - u.cube.xyz;
            float h = u.cube.w;
            float3 rd = select(worldRay, float3(1e-6), abs(worldRay) < 1e-6);
            float3 inv = 1.0 / rd;
            float3 t0 = (-h - ro) * inv;
            float3 t1 = (h - ro) * inv;
            float3 tmin3 = min(t0, t1);
            float3 tmax3 = max(t0, t1);
            float tNear = max(max(tmin3.x, tmin3.y), tmin3.z);
            float tFar = min(min(tmax3.x, tmax3.y), tmax3.z);
            if (tNear > 0.0 && tFar > tNear) {
                float3 a = abs(ro + rd * tNear);
                float e = h * 0.06;
                int edges = int(a.x > h - e) + int(a.y > h - e) + int(a.z > h - e);
                float3 pink = float3(1.0, 0.25, 0.9);
                col = edges >= 2 ? pink : mix(col, pink, 0.15);
            }
        }

        // Ping pong, as the camera sees it: this pixel shows the camera image at the sphere point S, and
        // the scene is traced along that same camera ray, so it lands exactly where the passthrough does
        // in both eyes (the physics stays in true 3D). In the table frame. Edges of the ball and blades
        // are anti-aliased by blending with whatever is behind them.
        float sceneT = 1e9;
        int sceneObject = -1, sceneWhich = -1;
        float netT = -1.0;
        float2 netQ = float2(0.0);
        float scoreT = -1.0;
        float4 scoreCol = float4(0.0);
        if (flags & 16) {
            float3 S = eye + rayGreen * seen;
            float3 camRayOrigin = u.camPos.xyz;
            float3 camRayDir = normalize(S - camRayOrigin);
            // A screen pixel's size as an angle seen from the camera (the same patch of sphere).
            float pixel = pixelAngle * seen / max(length(S - camRayOrigin), 1e-3);
            float cy = scene[1].x, sy = scene[1].y;
            float3 dw = camRayOrigin - scene[0].xyz;
            float3 tro = float3(cy * dw.x - sy * dw.z, dw.y, sy * dw.x + cy * dw.z);
            float3 trd = float3(cy * camRayDir.x - sy * camRayDir.z, camRayDir.y, sy * camRayDir.x + cy * camRayDir.z);
            float3 passthrough = col * floorShadow(tro, trd, scene);
            SceneHit hit = traceScene(tro, trd, scene, pixel, false, -1);
            if (hit.t < 1e8) {
                float3 front = shadeScene(hit, tro, trd, scene, pixel);
                if (hit.coverage < 0.999) {
                    SceneHit back = traceScene(tro, trd, scene, pixel, hit.object == kBall,
                                               hit.object == kBlade ? hit.which : -1);
                    float3 behind = back.t < 1e8 ? shadeScene(back, tro, trd, scene, pixel) : passthrough;
                    front = mix(behind, front, hit.coverage);
                }
                col = front;
                sceneT = hit.t;
                sceneObject = hit.object;
                sceneWhich = hit.which;
            } else {
                col = passthrough;
            }
            netT = netHit(tro, trd, netQ);
            // The scoreboard (scene[3].w), facing the player: its texture is premultiplied RGBA.
            if (scene[3].w > 0.5 && trd.z < -1e-4) {
                float t = -tro.z / trd.z;
                float3 q = tro + trd * t;
                float2 st = float2((q.x - kScoreX0) / kScoreW, 1.0 - (q.y - kScoreY0) / kScoreH);
                if (t > 0.0 && all(st >= 0.0) && all(st <= 1.0)) {
                    scoreT = t;
                    scoreCol = scoreTex.sample(s, st);
                }
            }
        }

        float handT = 1e9;
        float3 handCol = float3(0.0);
        if (flags & 8) {
            // Tracked hands: capsules along the bones, raytraced from this eye (true stereo).
            for (int h = 0; h < 2; h++) {
                constant float4* hand = hands + h * kHandStride;
                if (hand[0].x < 0.5) continue;
                float4 bound = hand[1];
                float3 oc = eye - bound.xyz;
                float bb = dot(worldRay, oc);
                if (bb * bb - (dot(oc, oc) - bound.w * bound.w) < 0.0) continue;   // misses the hand
                int hit = -1;
                float best = handT;
                for (int i = 0; i < 23; i++) {
                    float4 a = hand[2 + kBones[i].x], b = hand[2 + kBones[i].y];
                    if (a.w < 0.5 || b.w < 0.5) continue;
                    float r = (i >= 20 ? 0.010 : (kBones[i].y % 4 == 0 ? 0.0075 : 0.0085)) * hand[0].y;
                    float th = capsuleHit(eye, worldRay, a.xyz, b.xyz, r);
                    if (th > 0.0 && th < best) { best = th; hit = i; }
                }
                if (hit >= 0) {
                    float3 a = hand[2 + kBones[hit].x].xyz, b = hand[2 + kBones[hit].y].xyz;
                    float3 hp = eye + worldRay * best;
                    float3 ab = b - a;
                    float3 n = normalize(hp - (a + ab * saturate(dot(hp - a, ab) / max(dot(ab, ab), 1e-8))));
                    float3 base = hand[0].w > 0.5 ? float3(0.35, 1.0, 0.6) : float3(0.3, 0.75, 1.0);
                    base = mix(base, float3(1.0, 0.85, 0.35), hand[0].z);       // a gesture charging
                    float lambert = 0.35 + 0.65 * saturate(dot(n, -worldRay));
                    float rim = pow(1.0 - saturate(dot(n, -worldRay)), 3.0);
                    handT = best;
                    handCol = base * lambert + rim * 0.5;
                }
            }
            // Hands (on the sphere) and the scene (as the camera sees it) have no common depth: the
            // paddle in your hand and the ball are drawn over the hands, the hands over the rest.
            bool heldInFront = sceneObject == kBall || ((sceneObject == kBlade || sceneObject == kHandle) && sceneWhich == 0);
            if (handT < 1e8 && !heldInFront) { col = mix(col, handCol, 0.85); }
            else { handT = 1e9; }
        }

        if (netT > 0.0 && netT < sceneT && handT >= 1e8) {
            // The net: a translucent dark mesh with a white top band and a cord at the bottom.
            bool band = netQ.y > kNetHeight - 0.015;
            bool cord = netQ.y < 0.005;
            float3 netCol = band ? float3(0.93) : float3(0.03, 0.04, 0.05);
            col = mix(col, netCol, band || cord ? 1.0 : 0.55);
        }

        if (scoreT > 0.0 && scoreT < sceneT && handT >= 1e8) {
            col = col * (1.0 - scoreCol.a) + scoreCol.rgb;
        }

        if ((flags & 16) && scene[12].w > 0.0) {
            // Smart paddle: the calibration ring around the big tag where the camera sees it.
            float pixel = pixelAngle;
            float ring = scene[13].w;
            float3 c0 = scene[12].xyz, c1 = scene[13].xyz, c2 = scene[14].xyz, c3 = scene[15].xyz;
            if (ring > -0.5) {
                // Ring around the tag in its plane, filling clockwise from the top as views come in.
                float3 centre = (c0 + c1 + c2 + c3) * 0.25;
                float3 e1 = normalize(c1 - c0), n = normalize(cross(c1 - c0, c3 - c0));
                float3 e2 = cross(n, e1);
                float denom = dot(worldRay, n);
                if (abs(denom) > 1e-4) {
                    float t = dot(centre - eye, n) / denom;
                    float3 q = eye + worldRay * t - centre;
                    float radiusTag = length(c0 - centre);
                    float rq = length(q) / radiusTag;
                    float band = saturate((0.12 - abs(rq - 1.45)) * radiusTag / (pixel * max(t, 1e-3)) + 0.5);
                    if (t > 0.0 && band > 0.0) {
                        float frac = fract(atan2(dot(q, e1), dot(q, -e2)) / (2.0 * M_PI_F) + 1.0);
                        float filled = ring >= 1.5 ? 1.0 : step(frac, ring);
                        float3 ringCol = ring >= 2.5 ? float3(1.0, 0.25, 0.2)
                                       : ring >= 1.5 ? float3(0.25, 1.0, 0.45) : float3(1.0, 0.82, 0.25);
                        col = mix(col, mix(float3(0.25), ringCol, filled), band * 0.9);
                    }
                }
            }
        }

        if (flags & 4) {
            // HUD: screen-space text above each lens centre (premultiplied alpha), same pass, no overlay layer.
            float2 q = p - lensCentre - u.hud.xy;
            if (all(q >= 0.0) && all(q < u.hud.zw)) {
                float4 h = hudTex.read(uint2(q));
                col = col * (1.0 - h.a) + h.rgb;
            }
        }

        if (flatView) { return half4(half3(saturate(col)), half(1.0)); }

        // OLED: lifting black slightly keeps pixels from switching fully off, which smears dark areas
        // when the head turns. Then a soft fade to true black over the outer part of each lens.
        col = u.image.z + saturate(col) * (1.0 - u.image.z);
        col *= 1.0 - smoothstep(u.lens.w * (1.0 - max(u.image.w, 1e-3)), u.lens.w, radius);
        return half4(half3(col), half(1.0));
    }
    """
}
