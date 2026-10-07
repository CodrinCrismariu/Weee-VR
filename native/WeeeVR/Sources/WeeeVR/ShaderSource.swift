/// Metal shaders, compiled at runtime with `makeLibrary(source:)` because Apple's offline
/// `metal` compiler only ships with Xcode on macOS (we build on Linux/WSL with xtool).
///
/// One full-screen triangle; the fragment shader does everything per pixel:
///   screen pixel -> lens tan-angle -> Cardboard pincushion (k1, k2) -> world ray from this eye's real
///   position at display time -> hit the depth-proxy sphere (radius D around the eyes) -> project that
///   point into the camera at its capture pose -> YCbCr sample.
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
        float4 optics;      // k1, k2, fovScale, flags (1 grid, 2 cube, 4 hud, 8 hands)
        float4 camK;        // fx, fy, cx, cy in camera-buffer pixels
        float4 cam;         // buffer width, buffer height, ipd metres, flip (0/1)
        float4 camRot0, camRot1, camRot2;   // camera <- world at capture time (columns)
        float4 camPos;      // camera position in world at capture time
        float4 world0, world1, world2;      // world <- view at display time (columns)
        float4 head;        // eye midpoint in world at display time; w = depth-proxy distance D (m)
        float4 cube;        // world centre xyz, half size
        float4 hud;         // HUD top-left offset from each lens centre (px), HUD width, height (px)
    };

    struct VOut { float4 position [[position]]; };

    vertex VOut vrVertex(uint vid [[vertex_id]]) {
        float2 p = float2((vid << 1) & 2, vid & 2);
        VOut o;
        o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
        return o;
    }

    // Hands: HandModel.stride float4 per hand -- [0] valid, bone radius scale, 0, chirality;
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

    constant float4x4 kYCbCrToRGB = float4x4(
        float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
        float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
        float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
        float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));

    fragment half4 vrFragment(VOut in [[stage_in]],
                              constant Uniforms& u [[buffer(0)]],
                              constant float4* hands [[buffer(1)]],
                              texture2d<float> lumaTex [[texture(0)]],
                              texture2d<float> chromaTex [[texture(1)]],
                              texture2d<float> hudTex [[texture(2)]]) {
        constexpr sampler s(address::clamp_to_edge, filter::linear);

        float2 p = in.position.xy;
        bool rightEye = p.x >= u.screen.x * 0.5;
        float2 lensCentre = float2(rightEye ? u.lens.z : u.lens.x, u.lens.y);

        // Screen position as a tan-angle from the lens axis (x right, y up), then the lens's
        // pincushion: what direction the eye actually sees this pixel in.
        float2 tanScreen = float2(p.x - lensCentre.x, lensCentre.y - p.y) / (u.screen.z * u.screen.w);
        float r2 = dot(tanScreen, tanScreen);
        float2 tanEye = tanScreen * (1.0 + u.optics.x * r2 + u.optics.y * r2 * r2);
        float3 viewRay = normalize(float3(tanEye * u.optics.z, -1.0));

        float3x3 worldFromView = float3x3(u.world0.xyz, u.world1.xyz, u.world2.xyz);
        float3x3 camFromWorld = float3x3(u.camRot0.xyz, u.camRot1.xyz, u.camRot2.xyz);
        float3 worldRay = normalize(worldFromView * viewRay);
        float3 eye = u.head.xyz + worldFromView * float3((rightEye ? 0.5 : -0.5) * u.cam.z, 0.0, 0.0);

        // Grid derivatives are taken before any divergent return.
        float az = atan2(worldRay.x, -worldRay.z);
        float el = asin(clamp(worldRay.y, -1.0, 1.0));
        float2 g = float2(az, el) * (12.0 / M_PI_F);   // 15 degree cells
        float2 fw = max(fwidth(g), float2(1e-4));
        fw.x = min(fw.x, 0.5);                          // atan2 seam behind the viewer

        if (length(tanEye) > u.lens.w) {
            return half4(0.0, 0.0, 0.0, 1.0);
        }

        // Passthrough: hit the depth-proxy sphere around the eyes, then look that point up in the camera.
        float3 col = float3(0.05);
        float3 rel = eye - u.head.xyz;
        float b = dot(worldRay, rel);
        float c = dot(rel, rel) - u.head.w * u.head.w;
        float t = -b + sqrt(max(b * b - c, 0.0));
        float3 camRay = camFromWorld * (eye + worldRay * t - u.camPos.xyz);
        int flags = int(u.optics.w);

        if (camRay.z < -1e-3) {
            float2 tanCam = camRay.xy / -camRay.z;
            float2 uv = float2(u.camK.x * tanCam.x + u.camK.z, u.camK.w - u.camK.y * tanCam.y) / u.cam.xy;
            if (u.cam.w > 0.5) { uv = 1.0 - uv; }
            if (all(uv >= 0.0) && all(uv <= 1.0)) {
                float4 ycc = float4(lumaTex.sample(s, uv).r, chromaTex.sample(s, uv).rg, 1.0);
                col = (kYCbCrToRGB * ycc).rgb;
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

        if (flags & 8) {
            // Tracked hands: capsules along the bones, raytraced from this eye (true stereo).
            for (int h = 0; h < 2; h++) {
                constant float4* hand = hands + h * kHandStride;
                if (hand[0].x < 0.5) continue;
                float4 bound = hand[1];
                float3 oc = eye - bound.xyz;
                float bb = dot(worldRay, oc);
                if (bb * bb - (dot(oc, oc) - bound.w * bound.w) < 0.0) continue;   // misses the hand
                float best = 1e9;
                int hit = -1;
                for (int i = 0; i < 23; i++) {
                    float4 a = hand[2 + kBones[i].x], b = hand[2 + kBones[i].y];
                    if (a.w < 0.5 || b.w < 0.5) continue;
                    float r = (i >= 20 ? 0.010 : (kBones[i].y % 4 == 0 ? 0.0075 : 0.0085)) * hand[0].y;
                    float th = capsuleHit(eye, worldRay, a.xyz, b.xyz, r);
                    if (th > 0.0 && th < best) { best = th; hit = i; }
                }
                if (hit >= 0) {
                    float3 a = hand[2 + kBones[hit].x].xyz, b = hand[2 + kBones[hit].y].xyz;
                    float3 p = eye + worldRay * best;
                    float3 ab = b - a;
                    float3 n = normalize(p - (a + ab * saturate(dot(p - a, ab) / max(dot(ab, ab), 1e-8))));
                    float3 base = hand[0].w > 0.5 ? float3(0.35, 1.0, 0.6) : float3(0.3, 0.75, 1.0);
                    float lambert = 0.35 + 0.65 * saturate(dot(n, -worldRay));
                    float rim = pow(1.0 - saturate(dot(n, -worldRay)), 3.0);
                    col = mix(col, base * lambert + rim * 0.5, 0.85);
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

        return half4(half3(saturate(col)), 1.0h);
    }
    """
}
