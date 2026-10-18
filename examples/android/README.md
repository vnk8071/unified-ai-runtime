<!-- SPDX-License-Identifier: Apache-2.0 -->
# Android examples

| File | What it is | Checked |
|---|---|---|
| `run_on_device.sh` | Cross-compiles with the NDK, pushes to a device over `adb`, runs the quickstart and the core, session and generation tests there; with `--gguf model.gguf` it also builds llama.cpp (CPU, dotprod+i8mm) and generates text | Run on a Qualcomm SM8850 board (Android 16): all 7 checks pass, with and without a model |
| `app/` | An Android app that detects objects in a picture with UAIRT's TFLite backend (a quantized YOLOX): a **Sample image** button, a **Choose image** button (the system photo picker), boxes drawn over the picture | Built into an APK, installed on the same board and run: 5 detections on the sample, both buttons tapped, screenshot checked |

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk          # NDK r26 or newer
examples/android/run_on_device.sh                       # core checks
examples/android/run_on_device.sh --gguf model.gguf     # plus text generation
```

## The app

It needs no Gradle and no Android Studio: `app/build_apk.sh` uses the NDK for the native libraries and the Android SDK's `aapt2`, `javac`, `d8`, `zipalign` and `apksigner`
for the package, and signs it with a throw-away debug key. You supply what this repository does not ship: a JDK 17 or newer, the SDK (`sdkmanager "platforms;android-35" "build-tools;35.0.0"`),
a w8a8 YOLOX `.tflite` with outputs `boxes`, `scores` and `class_idx` (AI Hub's YOLOX-TFLite) with its class-name list, a JPEG for the sample button, and `libtensorflowlite_c.so` built for
`arm64-v8a` ([docs/backends/tflite.md](../../docs/backends/tflite.md#android)).

```bash
ANDROID_NDK_HOME=... JAVA_HOME=... MODEL=yolox.tflite LABELS=labels.txt SAMPLE=picture.jpg \
TFLITE_INCLUDE_DIR=<TensorFlow source root> TFLITE_LIBRARY=<libtensorflowlite_c.so> examples/android/app/build_apk.sh build-android-app
adb install -r build-android-app/uairt-example.apk
adb shell am start -n dev.uairt.example/.MainActivity --es mode sample      # runs the sample at start; the result is in `adb logcat -s UAIRT`
```

On the Qualcomm board the sample (a street scene with a bus) gives a bus at 93% and four people at 89, 89, 87 and 53%, in about 50 ms on the CPU, close to the iPhone
result for the same picture in [../ios](../ios/README.md). The picture is letterboxed into 640 x 640 in Java, and `src/main/cpp/uairt_jni.c` dequantizes the model's outputs and
applies the confidence threshold (0.25) and class-aware NMS (0.45). The plugin is loaded by path from the app's native library directory, so the manifest asks for
`extractNativeLibs`. The APK is about 10 MB with stripped libraries.

Not done: a camera mode (CameraX needs Gradle and Maven libraries), running a picture picked from the gallery (the board has none; the picker itself opens), and the NPU: the QNN backend
could run the same model as a `.dlc` ([docs/backends/qnn.md](../../docs/backends/qnn.md#android)), but its vendor libraries (about 100 MB) are Qualcomm's and would have to be
bundled from your own QAIRT SDK.
