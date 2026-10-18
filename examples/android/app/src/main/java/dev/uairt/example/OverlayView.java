// SPDX-License-Identifier: Apache-2.0
package dev.uairt.example;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

/** Draws boxes over a picture shown aspect-fit in the same area. Boxes are in the picture's own pixels. */
final class OverlayView extends View {
    static final class Box {
        final RectF rect;
        final String text;

        Box(RectF rect, String text) {
            this.rect = rect;
            this.text = text;
        }
    }

    private final Paint stroke = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint label = new Paint(Paint.ANTI_ALIAS_FLAG);
    private List<Box> boxes = new ArrayList<>();
    private int pictureWidth = 1, pictureHeight = 1;

    OverlayView(Context context) {
        super(context);
        stroke.setStyle(Paint.Style.STROKE);
        stroke.setStrokeWidth(4);
        stroke.setColor(Color.GREEN);
        fill.setColor(Color.argb(220, 0, 255, 0));
        label.setColor(Color.BLACK);
        label.setTextSize(34);
        label.setFakeBoldText(true);
    }

    void setBoxes(List<Box> boxes, int pictureWidth, int pictureHeight) {
        this.boxes = boxes;
        this.pictureWidth = Math.max(1, pictureWidth);
        this.pictureHeight = Math.max(1, pictureHeight);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        float scale = Math.min((float) getWidth() / pictureWidth, (float) getHeight() / pictureHeight);
        float left = (getWidth() - pictureWidth * scale) / 2, top = (getHeight() - pictureHeight * scale) / 2;
        for (Box box : boxes) {
            RectF r = new RectF(left + box.rect.left * scale, top + box.rect.top * scale, left + box.rect.right * scale,
                                top + box.rect.bottom * scale);
            canvas.drawRect(r, stroke);
            float width = label.measureText(box.text) + 8;
            canvas.drawRect(r.left, Math.max(0, r.top - 40), r.left + width, Math.max(0, r.top - 40) + 40, fill);
            canvas.drawText(box.text, r.left + 4, Math.max(0, r.top - 40) + 31, label);
        }
    }
}
