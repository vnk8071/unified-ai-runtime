// SPDX-License-Identifier: Apache-2.0
package dev.uairt.example;

/** The JNI layer (src/main/cpp/uairt_jni.c): UAIRT's TFLite backend running a quantized YOLOX detector. */
final class Native {
    static {
        System.loadLibrary("uairt_jni");   // brings in libuairt.so
    }

    private Native() {}

    static native String version();

    /** Returns "" on success, otherwise the error message. */
    static native String load(String pluginDir, String modelPath, int threads);

    /** pixels: 640 x 640 ARGB. Returns [inference ms, n, n x (x1, y1, x2, y2, score, class)] in input pixels, or null. */
    static native float[] detect(int[] pixels, float confidence, float iou);
}
