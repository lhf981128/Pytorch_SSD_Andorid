package org.pytorch.helloworld;

import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.Tensor;

import java.nio.FloatBuffer;

public class imageUtil {
    public static float[] TORCHVISION_NORM_MEAN_RGB = new float[]{127.0f, 127.0f, 127.0f};
    public static float[] TORCHVISION_NORM_STD_RGB = new float[]{128.0f, 128.0f, 128.0f};

    /**
     * Creates new {@link org.pytorch.Tensor} from full {@link android.graphics.Bitmap}, normalized
     * with specified in parameters mean and std.
     *
     * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
     * @param normStdRGB  standard deviation for RGB channels normalization, length must equal 3, RGB order
     */
    public static Tensor bitmapToFloat32Tensor(
            final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[]) {
        checkNormMeanArg(normMeanRGB);
        checkNormStdArg(normStdRGB);

        return bitmapToFloat32Tensor(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB);
    }

    /**
     * Writes tensor content from specified {@link android.graphics.Bitmap},
     * normalized with specified in parameters mean and std to specified {@link java.nio.FloatBuffer}
     * with specified offset.
     *
     * @param bitmap      {@link android.graphics.Bitmap} as a source for Tensor data
     * @param x           - x coordinate of top left corner of bitmap's area
     * @param y           - y coordinate of top left corner of bitmap's area
     * @param width       - width of bitmap's area
     * @param height      - height of bitmap's area
     * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
     * @param normStdRGB  standard deviation for RGB channels normalization, length must equal 3, RGB order
     */
    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int x,
            final int y,
            final int width,
            final int height,
            final float[] normMeanRGB,
            final float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset) {
        checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);
        checkNormMeanArg(normMeanRGB);
        checkNormStdArg(normStdRGB);

        final int pixelsCount = height * width;
        final int[] pixels = new int[pixelsCount];
        bitmap.getPixels(pixels, 0, width, x, y, width, height);
        final int offset_g = pixelsCount;
        final int offset_b = 2 * pixelsCount;
        for (int i = 0; i < pixelsCount; i++) {
            final int c = pixels[i];
            float r = ((c >> 16) & 0xff);
            float g = ((c >> 8) & 0xff);
            float b = ((c) & 0xff);
            float rF = (r - normMeanRGB[0]) / normStdRGB[0];
//            Log.i("rf","-----------rf---------------"+rF);
            float gF = (g - normMeanRGB[1]) / normStdRGB[1];
            float bF = (b - normMeanRGB[2]) / normStdRGB[2];
            outBuffer.put(outBufferOffset + i, rF);
            outBuffer.put(outBufferOffset + offset_g + i, gF);
            outBuffer.put(outBufferOffset + offset_b + i, bF);
        }
    }

    /**
     * Creates new {@link org.pytorch.Tensor} from specified area of {@link android.graphics.Bitmap},
     * normalized with specified in parameters mean and std.
     *
     * @param bitmap      {@link android.graphics.Bitmap} as a source for Tensor data
     * @param x           - x coordinate of top left corner of bitmap's area
     * @param y           - y coordinate of top left corner of bitmap's area
     * @param width       - width of bitmap's area
     * @param height      - height of bitmap's area
     * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
     * @param normStdRGB  standard deviation for RGB channels normalization, length must equal 3, RGB order
     */
    public static Tensor bitmapToFloat32Tensor(
            final Bitmap bitmap,
            int x,
            int y,
            int width,
            int height,
            float[] normMeanRGB,
            float[] normStdRGB) {
            checkNormMeanArg(normMeanRGB);
            checkNormStdArg(normStdRGB);

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
        bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0);
        return Tensor.fromBlob(floatBuffer, new long[]{1, 3, height, width});
    }
    private static void checkNormStdArg(float[] normStdRGB) {
        if (normStdRGB.length != 3) {
            throw new IllegalArgumentException("normStdRGB length must be 3");
        }
    }

    private static void checkNormMeanArg(float[] normMeanRGB) {
        if (normMeanRGB.length != 3) {
            throw new IllegalArgumentException("normMeanRGB length must be 3");
        }
    }
    private static void checkOutBufferCapacity(FloatBuffer outBuffer, int outBufferOffset, int tensorWidth, int tensorHeight) {
        if (outBufferOffset + 3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
            throw new IllegalStateException("Buffer underflow");
        }
    }
}
