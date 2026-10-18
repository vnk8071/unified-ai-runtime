// SPDX-License-Identifier: Apache-2.0
package dev.uairt.example;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Detects objects in a picture with UAIRT. "Sample image" uses the picture bundled in the app, "Choose image" opens the system picker.
 * For a check without touching the screen: adb shell am start -n dev.uairt.example/.MainActivity --es mode sample, then read logcat -s UAIRT.
 */
public class MainActivity extends Activity {
    private static final String TAG = "UAIRT";
    private static final int SIDE = 640, PICK = 1;

    private ImageView imageView;
    private OverlayView overlay;
    private TextView status;
    private Button chooseButton, sampleButton;
    private String[] labels = new String[0];
    private volatile boolean ready;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF000000);
        status = new TextView(this);
        status.setTextColor(0xFFFFFFFF);
        status.setPadding(24, 24, 24, 24);
        status.setText("loading the model...");
        FrameLayout picture = new FrameLayout(this);
        imageView = new ImageView(this);
        imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        overlay = new OverlayView(this);
        picture.addView(imageView, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        picture.addView(overlay, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        LinearLayout buttons = new LinearLayout(this);
        buttons.setGravity(Gravity.CENTER);
        chooseButton = new Button(this);
        chooseButton.setText("Choose image");
        sampleButton = new Button(this);
        sampleButton.setText("Sample image");
        chooseButton.setEnabled(false);
        sampleButton.setEnabled(false);
        buttons.addView(chooseButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        buttons.addView(sampleButton, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        root.addView(status, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(picture, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        root.addView(buttons, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        // Android 15 and later draw the app edge to edge: keep the content clear of the status bar and the navigation bar.
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars());
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return WindowInsets.CONSUMED;
        });
        setContentView(root);

        chooseButton.setOnClickListener(v -> {
            Intent pick = new Intent(Intent.ACTION_GET_CONTENT).setType("image/*").addCategory(Intent.CATEGORY_OPENABLE);
            startActivityForResult(pick, PICK);
        });
        sampleButton.setOnClickListener(v -> detectSample());

        final String mode = getIntent().getStringExtra("mode");
        new Thread(() -> {
            String error = loadModel();
            runOnUiThread(() -> {
                if (!error.isEmpty()) {
                    status.setText("could not load the model: " + error);
                    return;
                }
                ready = true;
                chooseButton.setEnabled(true);
                sampleButton.setEnabled(true);
                status.setText("UAIRT " + Native.version() + ": ready");
                if ("sample".equals(mode)) {
                    detectSample();
                }
            });
        }).start();
    }

    /** Copies the model out of the APK (the backend opens it by path) and loads it. Called off the main thread. */
    private String loadModel() {
        try {
            File model = new File(getFilesDir(), "yolox.tflite");
            if (!model.exists()) {
                try (InputStream in = getAssets().open("yolox.tflite"); FileOutputStream out = new FileOutputStream(model)) {
                    byte[] buffer = new byte[1 << 16];
                    for (int n; (n = in.read(buffer)) > 0; ) {
                        out.write(buffer, 0, n);
                    }
                }
            }
            try (InputStream in = getAssets().open("labels.txt")) {
                labels = new String(in.readAllBytes(), StandardCharsets.UTF_8).trim().split("\n");
            }
            return Native.load(getApplicationInfo().nativeLibraryDir, model.getPath(), 4);
        } catch (IOException e) {
            return e.toString();
        }
    }

    private void detectSample() {
        try (InputStream in = getAssets().open("sample.jpg")) {
            detect(BitmapFactory.decodeStream(in), "sample image");
        } catch (IOException e) {
            status.setText("no sample image in this build");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != PICK || resultCode != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        try {
            Uri uri = data.getData();
            Bitmap bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), uri), (decoder, info, source) -> {
                decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);          // getPixels needs a software bitmap
                int longest = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
                if (longest > 2000) {
                    decoder.setTargetSampleSize(longest / 1000);
                }
            });
            detect(bitmap, "chosen image");
        } catch (IOException e) {
            status.setText("could not read the picture: " + e.getMessage());
        }
    }

    /** Letterboxes the picture into 640 x 640 (grey padding keeps the aspect ratio), runs the detector off the main thread, draws the boxes. */
    private void detect(final Bitmap picture, final String title) {
        imageView.setImageBitmap(picture);
        overlay.setBoxes(new ArrayList<>(), picture.getWidth(), picture.getHeight());
        status.setText(title + ": detecting...");
        new Thread(() -> {
            float scale = Math.min((float) SIDE / picture.getWidth(), (float) SIDE / picture.getHeight());
            float width = picture.getWidth() * scale, height = picture.getHeight() * scale;
            float padX = (SIDE - width) / 2, padY = (SIDE - height) / 2;
            Bitmap input = Bitmap.createBitmap(SIDE, SIDE, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(input);
            canvas.drawColor(0xFF727272);
            canvas.drawBitmap(picture, null, new RectF(padX, padY, padX + width, padY + height), new Paint(Paint.FILTER_BITMAP_FLAG));
            int[] pixels = new int[SIDE * SIDE];
            input.getPixels(pixels, 0, SIDE, 0, 0, SIDE, SIDE);

            float[] r = Native.detect(pixels, 0.25f, 0.45f);
            if (r == null) {
                runOnUiThread(() -> status.setText("detection failed (see logcat -s UAIRT)"));
                return;
            }
            int n = (int) r[1];
            final List<OverlayView.Box> boxes = new ArrayList<>();
            StringBuilder summary = new StringBuilder();
            for (int i = 0; i < n; i++) {
                int k = 2 + 6 * i;
                RectF box = new RectF(Math.max(0, (r[k] - padX) / scale), Math.max(0, (r[k + 1] - padY) / scale),
                                      Math.min(picture.getWidth(), (r[k + 2] - padX) / scale),
                                      Math.min(picture.getHeight(), (r[k + 3] - padY) / scale));
                int cls = (int) r[k + 5];
                String name = cls >= 0 && cls < labels.length ? labels[cls].trim() : "class " + cls;
                String text = String.format("%s %.0f%%", name, r[k + 4] * 100);
                boxes.add(new OverlayView.Box(box, text));
                summary.append(String.format("%n  %s at [%d, %d, %d, %d]", text, (int) box.left, (int) box.top, (int) box.right, (int) box.bottom));
            }
            final float ms = r[0];
            Log.i(TAG, String.format("%s: %d detections of %dx%d, inference %.1f ms (TFLite CPU)%s", title, n, picture.getWidth(),
                                     picture.getHeight(), ms, summary));
            runOnUiThread(() -> {
                overlay.setBoxes(boxes, picture.getWidth(), picture.getHeight());
                status.setText(String.format("%s | TFLite CPU | inference %.1f ms | %d objects", title, ms, boxes.size()));
            });
        }).start();
    }
}
