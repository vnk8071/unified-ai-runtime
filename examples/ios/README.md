<!-- SPDX-License-Identifier: Apache-2.0 -->
# iOS example

A small UIKit app that runs UAIRT on an iPhone: the core and the CoreML backend are linked into the app (iOS cannot load plugins; see
[docs/backends/coreml.md](../../docs/backends/coreml.md#linked-into-an-app-ios)). At start-up it prints, to stdout and on screen, the result of

1. the `reference` backend,
2. a bundled CoreML model (`convnet.mlmodelc`) under each `compute_units` setting (`cpu_only`, `cpu_and_gpu`, `cpu_and_ne`, `all`), with latency,
3. optionally a bundled detector network and one input image, saving the raw outputs to the app's `Documents` so they can be compared with a Mac,
4. a live camera detector behind a button (below).

Checked on an iPhone 12 (iOS 18.1.1) with Xcode 27.1: all four settings run. The small model's outputs equal the Mac's CPU result to 6e-8. The YOLOv9t network
(from the [Hugging Face YOLOv9-CoreML repository](../../docs/backends/coreml.md#models-with-a-built-in-nms-stage), AGPL-3.0, not in this repository) took 21 ms on the CPU, 24 ms
on CPU and GPU, and 3.3 ms with the Neural Engine, after a 1.4 s first load there; its scores on `cpu_only` are identical to the Mac's, the Neural Engine and GPU differ by up to 0.014 (half precision),
and `prepost.nms` on every setting's output gives the same five detections as on the Mac (a bus and four people).

**Camera.** With a detector bundled, the app has a **Start camera** button: it opens the back camera, stretches each frame to the model's 640 x 640 input
(Accelerate), runs the network through UAIRT's CoreML backend, decodes the boxes and applies NMS in plain C (`detector.c`), and draws the boxes and labels over the
preview. The **Compute** button cycles `all`, `cpu_only`, `cpu_and_gpu` and `cpu_and_ne`, reloading the model, so you can see the difference live; the status line shows the
inference time and frame rate. The first start asks for camera permission. For a check without touching the phone, launch with `--camera-seconds N --exit-after`: it runs
the camera for N seconds and prints the frames, the average inference time and which classes it saw. Checked on an iPhone 12: 208 frames in 8 s with the Neural Engine at
10.6 ms average inference per frame (higher than the 3.3 ms of the still image because the camera and scaling run at the same time). The on-screen layout (buttons, box
overlay) was not checked by a screenshot.

You need Xcode, an Apple developer account (a free personal team works) and a paired iPhone with Developer Mode on. The models are not in this
repository: make them with `coremltools` (`tests/compare_with_coremltools.py` has small ones) and compile them for iOS with
`xcrun coremlcompiler compile model.mlpackage out/ --platform ios --deployment-target 16.0`.

```bash
cmake -S examples/ios -B build-ios -G Xcode -DCMAKE_SYSTEM_NAME=iOS \
      -DUAIRT_IOS_TEAM_ID=<team id> -DUAIRT_IOS_BUNDLE_ID=com.example.uairt \
      -DUAIRT_IOS_MODEL=out/convnet.mlmodelc \
      -DUAIRT_IOS_DETECTOR=out/detector.mlmodelc -DUAIRT_IOS_DETECTOR_INPUT=input_bgra.raw      # the last two are optional
xcodebuild -project build-ios/uairt_ios_example.xcodeproj -scheme UairtExample -configuration Release \
           -destination generic/platform=iOS -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> build-ios/Release-iphoneos/UairtExample.app
xcrun devicectl device process launch --device <udid> --console --terminate-existing <bundle id> --exit-after
```

`-allowProvisioningUpdates` registers the bundle identifier and a development profile under your team. With a personal team the phone must also trust
the certificate once (Settings, General, VPN & Device Management) before the first launch, and the phone must be unlocked for `install` and `launch`.
The app targets iOS 16; a higher deployment target will not install on an older phone. The detector's input is the model's image input as raw bytes
(1 x height x width x 4, BGRA).
