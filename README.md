# Weee Sports VR: iPhone 15 + Cardboard passthrough test bed

This is a native iOS app (Swift, Metal, AVFoundation and ARKit), built and installed entirely from Windows with
[xtool](https://github.com/xtool-org/xtool) running in WSL2. All rendering happens on the iPhone; the PC is only
used to build the app and read its logs.

## What the app does
- **6DoF tracking (default).** ARKit world tracking, Apple's on-device visual-inertial SLAM, tracks the phone's
  position and rotation and maintains a map. **Tapping the screen resets the map and the position.**
- **Eye-correct passthrough.**
  - The 1× camera sits about 48 mm left of, 11 mm above and 62 mm in front of your eyes.
  - The camera image is projected onto a sphere around your head at a "proxy depth" D, then viewed from each eye's
    real position.
  - This corrects the camera/eye offset, gives the passthrough real stereo depth, and reprojects head rotation
    *and* movement to display time.
  - Objects at distance D are exact. D follows ARKit's feature points automatically, or can be fixed in
    calibration.
- **Low-latency path.**
  - A display link with `preferredFrameLatency = 1` keeps at most two drawables in flight.
  - **Late-latching:** each frame renders just before its display deadline, using the newest camera frame and
    pose.
  - The HUD is drawn inside the same Metal pass, not as a separate layer on top.
- **Camera 3DoF mode** (calibration toggle). Plain AVFoundation capture plus the gyro: the lowest-latency
  comparison mode, and the only mode that can use the ultra-wide.
- **Measurement.** The HUD and the log show:
  - capture→photon latency (p50/p95/max) and camera→app time
  - render CPU/GPU time and missed screen refreshes
  - tracking state, position, depth and thermal state

## One-time setup
1. Run `scripts/setup-wsl.sh` inside WSL. This is already done on this PC: it installed Swift 6.4, usbmuxd,
   libimobiledevice and xtool into `Ubuntu-24.04`.
2. Download **Xcode 27 (.xip)** from <https://developer.apple.com/download/all/?q=Xcode> (a free Apple ID works)
   into `C:\Users\Useru\Downloads`.
3. Open a WSL terminal (`wsl -d Ubuntu-24.04`) and run `xtool setup`.
   - Choose **Password** login. This works with a free Apple ID, but it uses private Apple APIs, so a secondary
     Apple ID is safer.
   - When it asks for Xcode, give it `/mnt/c/Users/Useru/Downloads/<the file>.xip`.
   - Check that `swift sdk list` shows `darwin`.
4. **Connect the iPhone through Windows, not usbipd.** usbmuxd over usbipd breaks on large transfers (the log
   shows "message was too large (65536 bytes)"), so app installs hang. Instead:
   - Install **Apple Devices** from the Microsoft Store and open it once. This registers Apple Mobile Device
     Service on `127.0.0.1:27015`.
   - `%USERPROFILE%\.wslconfig` sets `networkingMode=mirrored`, so WSL can reach that port.
   - `scripts/setup-usbmux-relay.sh` (run once in WSL) installs a `socat` systemd service that exposes it as
     `/var/run/usbmuxd`.
   - Plug in the iPhone, unlock it and tap **Trust**. Leave it connected to Windows; **don't** usbipd-attach it.
     `idevice_id -l` in WSL should print its UDID.
5. On the iPhone, turn on **Settings → Privacy & Security → Developer Mode**. The phone restarts. The toggle
   appears after the first install attempt.

(`scripts/attach-iphone.ps1` is the usbipd route that doesn't work for installs. It's kept only for reference.)

## Build and run
```bash
# inside WSL, from the project folder
cd "/mnt/c/Users/Useru/Documents/Weee Sports - VR"
scripts/dev.sh          # build + install + launch on the iPhone
scripts/logs.sh         # live stats from the phone (one line every 2 s)
```
- First launch: trust your Apple ID under **Settings → General → VPN & Device Management**.
- Free Apple ID signatures expire after **7 days**. Re-run `scripts/dev.sh` to refresh.

## Controls (phone in landscape, home bar on the right)
| Gesture | Action |
|---|---|
| **Tap** (works with the Cardboard button) | **Reset tracking:** new map, position back to 0, cube placed 1 m ahead |
| Two-finger tap | HUD on/off |
| Three-finger tap | Reprojection on/off |
| Long press | Calibration: tracking mode / camera (only here), hands, XRSLAM trims, depth, eye offset, lenses… |

## Why tracking and passthrough use the 1× camera
A probe on this iPhone 15 (iOS 26.6.1) found:
- ARKit world tracking only offers 1× formats (best: 1920×1440 @ 60).
- Starting an ultra-wide capture session while ARKit runs interrupts ARKit: it dropped from 60 fps to 0.

So 0.5× passthrough with position tracking would need our own SLAM on the ultra-wide frames. That's a separate
project.

## Measuring latency
1. **On the device.** The HUD and `scripts/logs.sh` show capture→photon p50/p95. Compare:
   - main vs ultra-wide
   - 60 vs 120 fps capture
   - Camera mode vs ARKit mode
   - late-latch on vs off, reprojection on vs off
2. **Ground truth.** Open `tools/latency-clock.html` full-screen on the PC monitor and point the phone at it.
   Photograph both screens at once with another camera; the difference between the two times on screen is the
   glass-to-glass latency.
3. **Swim test.** Look at the green grid / pink cube and turn your head quickly, with reprojection on and then off.

## iPhone 15 camera notes
- **Main camera:** 48 MP, 26 mm, f/1.6, OIS. This is the default; it is the best in low light.
- **Ultra-wide:** 12 MP, 13 mm, fixed focus. Wider, but softer and noisier.
- **No LiDAR and no telephoto.**
- **One camera, two eye views.** The rear lenses are about 1.5 cm apart (eyes are about 63 mm apart), so the
  stereo comes from the depth-proxy reprojection, not from two cameras.
- **Cut a hole in the Cardboard flap over the camera.** In landscape the camera sits behind the upper-left of the
  screen.
- **In dim light the image gets darker rather than slower.** The frame rate is locked, so iOS can't stretch the
  exposure. Test in bright light.
