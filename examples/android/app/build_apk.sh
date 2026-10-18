#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds the example app into an APK without Gradle: the NDK compiles UAIRT, its TFLite backend plugin and the JNI layer; the Android SDK's
# aapt2, javac, d8, zipalign and apksigner package them. The APK is signed with a throw-away debug key.
#
#   ANDROID_NDK_HOME=... ANDROID_HOME=... JAVA_HOME=... \
#   MODEL=yolox.tflite LABELS=labels.txt SAMPLE=picture.jpg \
#   TFLITE_INCLUDE_DIR=<TensorFlow source root> TFLITE_LIBRARY=<libtensorflowlite_c.so built for arm64-v8a> \
#   examples/android/app/build_apk.sh [output-dir]
#
# MODEL is a w8a8 YOLOX .tflite with outputs boxes, scores and class_idx (AI Hub's YOLOX-TFLite); LABELS lists its class names, one per line.
# None of these files is in this repository. docs/backends/tflite.md says how to build the TFLite library for Android.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME}" "${JAVA_HOME:?set JAVA_HOME to a JDK 17 or newer}"
: "${MODEL:?set MODEL to the .tflite file}" "${LABELS:?set LABELS to its class-name list}" "${SAMPLE:?set SAMPLE to a JPEG for the Sample button}"
: "${TFLITE_INCLUDE_DIR:?set TFLITE_INCLUDE_DIR}" "${TFLITE_LIBRARY:?set TFLITE_LIBRARY}"
sdk="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}"
bt="$(ls -d "$sdk"/build-tools/* | sort -V | tail -1)"
platform="$(ls -d "$sdk"/platforms/android-* | sort -V | tail -1)/android.jar"
[ -f "$platform" ] && [ -x "$bt/aapt2" ] || { echo "the Android SDK needs a platform and build-tools (sdkmanager \"platforms;android-35\" \"build-tools;35.0.0\")" >&2; exit 1; }
export PATH="$JAVA_HOME/bin:$bt:$PATH"
llvm="$(echo "$ANDROID_NDK_HOME"/toolchains/llvm/prebuilt/*)/bin"

out="$(cd "${1:-$root/build-android-app}" 2>/dev/null && pwd || (mkdir -p "${1:-$root/build-android-app}" && cd "${1:-$root/build-android-app}" && pwd))"
rm -rf "$out/stage" "$out/classes" "$out/dex" && mkdir -p "$out/stage/lib/arm64-v8a" "$out/stage/assets" "$out/classes" "$out/dex"

echo "== native: UAIRT core, TFLite backend plugin, JNI layer"
export TFLITE_INCLUDE_DIR TFLITE_LIBRARY
cmake -S "$root" -B "$out/cmake" -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 -DANDROID_STL=none -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DUAIRT_BUILD_TESTS=OFF \
  -DUAIRT_BUILD_EXAMPLES=OFF -DUAIRT_BUILD_TFLITE=ON >/dev/null
cmake --build "$out/cmake" -j8 --target uairt uairt_backend_tflite >/dev/null
"$llvm/aarch64-linux-android29-clang" -shared -fPIC -O2 -Wall -Wextra -I "$root/include" "$here/src/main/cpp/uairt_jni.c" \
  -L "$out/cmake" -luairt -llog -o "$out/stage/lib/arm64-v8a/libuairt_jni.so"
cp "$out/cmake/libuairt.so" "$out/cmake/libuairt_backend_tflite.so" "$TFLITE_LIBRARY" "$out/stage/lib/arm64-v8a/"
for f in "$out"/stage/lib/arm64-v8a/*.so; do "$llvm/llvm-strip" --strip-unneeded "$f"; done

echo "== java and resources"
cp "$MODEL" "$out/stage/assets/yolox.tflite"
cp "$LABELS" "$out/stage/assets/labels.txt"
cp "$SAMPLE" "$out/stage/assets/sample.jpg"
javac --release 17 -Xlint:all -cp "$platform" -d "$out/classes" "$here"/src/main/java/dev/uairt/example/*.java
d8 --lib "$platform" --min-api 29 --output "$out/dex" "$out"/classes/dev/uairt/example/*.class
aapt2 link -o "$out/base.apk" --manifest "$here/AndroidManifest.xml" -I "$platform" -A "$out/stage/assets" \
  --min-sdk-version 29 --target-sdk-version 35

echo "== package and sign"
cp "$out/base.apk" "$out/unsigned.apk"
cp "$out/dex/classes.dex" "$out/stage/"
(cd "$out/stage" && zip -q -r "$out/unsigned.apk" classes.dex lib)
zipalign -p -f 4 "$out/unsigned.apk" "$out/aligned.apk"
[ -f "$out/debug.keystore" ] || keytool -genkeypair -keystore "$out/debug.keystore" -storepass android -keypass android -alias debug \
  -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=UAIRT example debug" >/dev/null 2>&1
apksigner sign --ks "$out/debug.keystore" --ks-pass pass:android --key-pass pass:android --out "$out/uairt-example.apk" "$out/aligned.apk"
apksigner verify "$out/uairt-example.apk"
echo "built $out/uairt-example.apk ($(du -h "$out/uairt-example.apk" | cut -f1))"
