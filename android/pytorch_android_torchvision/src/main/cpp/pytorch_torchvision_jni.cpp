#include <cassert>
#include <cmath>
#include <vector>

#include <libyuv.h>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(__ANDROID__)

#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-vision-jni", __VA_ARGS__)

#endif

namespace pytorch_vision_jni {
class PytorchVisionJni : public facebook::jni::JavaClass<PytorchVisionJni> {
 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/torchvision/PyTorchVision;";

  static int clamp(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
  }

  static void nativeImageYUV420CenterCropToFloatBuffer(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight,
      const int rotateCWDegrees,
      const int tensorWidth,
      const int tensorHeight,
      facebook::jni::alias_ref<jfloatArray> jnormMeanRGB,
      facebook::jni::alias_ref<jfloatArray> jnormStdRGB,
      facebook::jni::alias_ref<facebook::jni::JBuffer> outBuffer,
      const int outOffset) {
    JNIEnv* jni = facebook::jni::Environment::current();
    const auto dataCapacity = jni->GetDirectBufferCapacity(outBuffer.get());
    float* outData = (float*)jni->GetDirectBufferAddress(outBuffer.get());

    const auto normMeanRGB = jnormMeanRGB->getRegion(0, 3);
    const auto normStdRGB = jnormStdRGB->getRegion(0, 3);

    const int widthBeforeRotation = imageWidth;
    const int heightBeforeRotation = imageHeight;

    int widthAfterRotation = widthBeforeRotation;
    int heightAfterRotation = heightBeforeRotation;
    if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
      widthAfterRotation = heightBeforeRotation;
      heightAfterRotation = widthBeforeRotation;
    }

    int centerCropWidthAfterRotation = widthAfterRotation;
    int centerCropHeightAfterRotation = heightAfterRotation;

    if (tensorWidth * heightAfterRotation <=
        tensorHeight * widthAfterRotation) {
      centerCropWidthAfterRotation =
          std::floor(tensorWidth * heightAfterRotation / tensorHeight);
    } else {
      centerCropHeightAfterRotation =
          std::floor(tensorHeight * widthAfterRotation / tensorWidth);
    }

    int centerCropWidthBeforeRotation = centerCropWidthAfterRotation;
    int centerCropHeightBeforeRotation = centerCropHeightAfterRotation;
    if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
      centerCropHeightBeforeRotation = centerCropWidthAfterRotation;
      centerCropWidthBeforeRotation = centerCropHeightAfterRotation;
    }

    int offsetX =
        std::floor((widthBeforeRotation - centerCropWidthBeforeRotation) / 2.f);
    int offsetY = std::floor(
        (heightBeforeRotation - centerCropHeightBeforeRotation) / 2.f);

    uint8_t* yData = yBuffer->getDirectBytes();
    uint8_t* uData = uBuffer->getDirectBytes();
    uint8_t* vData = vBuffer->getDirectBytes();

    float scale = centerCropWidthAfterRotation / tensorWidth;
    int uvRowStride = uRowStride >> 1;

    int channelSize = tensorHeight * tensorWidth;
    int tensorInputOffsetG = channelSize;
    int tensorInputOffsetB = 2 * channelSize;

    for (int y = 0; y < tensorHeight; y++) {
      for (int x = 0; x < tensorWidth; x++) {
        int centerCropXAfterRotation = std::floor(x * scale);
        int centerCropYAfterRotation = std::floor(y * scale);

        int xBeforeRotation = offsetX + centerCropXAfterRotation;
        int yBeforeRotation = offsetY + centerCropYAfterRotation;
        if (rotateCWDegrees == 90) {
          xBeforeRotation = offsetX + centerCropYAfterRotation;
          yBeforeRotation = offsetY + (centerCropHeightBeforeRotation - 1) -
              centerCropXAfterRotation;
        } else if (rotateCWDegrees == 180) {
          xBeforeRotation = offsetX + (centerCropWidthBeforeRotation - 1) -
              centerCropXAfterRotation;
          yBeforeRotation = offsetY + (centerCropHeightBeforeRotation - 1) -
              centerCropYAfterRotation;
        } else if (rotateCWDegrees == 270) {
          xBeforeRotation = offsetX + (centerCropWidthBeforeRotation - 1) -
              centerCropYAfterRotation;
          yBeforeRotation = offsetY + centerCropXAfterRotation;
        }

        int yIdx =
            yBeforeRotation * yRowStride + xBeforeRotation * yPixelStride;
        int uvIdx = (yBeforeRotation >> 1) * uvRowStride +
            xBeforeRotation * uvPixelStride;

        const int yi = yData[yIdx];
        const int ui = uData[uvIdx];
        const int vi = vData[uvIdx];

        const int a0 = 1192 * (yi - 16);
        const int a1 = 1634 * (vi - 128);
        const int a2 = 832 * (vi - 128);
        const int a3 = 400 * (ui - 128);
        const int a4 = 2066 * (ui - 128);

        const int r = clamp((a0 + a1) >> 10, 0, 255);
        const int g = clamp((a0 - a2 - a3) >> 10, 0, 255);
        const int b = clamp((a0 + a4) >> 10, 0, 255);

        const float rf = ((r / 255.f) - normMeanRGB[0]) / normStdRGB[0];
        const float gf = ((g / 255.f) - normMeanRGB[1]) / normStdRGB[1];
        const float bf = ((b / 255.f) - normMeanRGB[2]) / normStdRGB[2];

        int offset = y * tensorWidth + x;
        outData[outOffset + offset] = rf;
        outData[outOffset + offset + tensorInputOffsetG] = gf;
        outData[outOffset + offset + tensorInputOffsetB] = bf;
      }
    }
  }

  static void putYuvImage(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uvRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight) {
    ALOGI(
        "JJJ putYuvImage(yRowStride:%d yPixelStride:%d uvRowStride:%d uvPixelStride:%d imageWidth:%d imageHeight:%d)",
        yRowStride,
        yPixelStride,
        uvRowStride,
        uvPixelStride,
        imageWidth,
        imageHeight);

    int halfImageWidth = (imageWidth + 1) / 2;
    int halfImageHeight = (imageHeight + 1) / 2;

    const uint32_t yuvSize =
        imageHeight * imageWidth + 2 * halfImageHeight * halfImageWidth;

    std::vector<uint8_t> yuvData;
    std::vector<uint8_t> rgbData;

    if (yuvData.size() != yuvSize) {
      yuvData.resize(yuvSize);
    }

    const uint32_t argbSize = 4 * imageHeight * imageWidth;
    if (rgbData.size() != argbSize) {
      rgbData.resize(argbSize);
    }

    const auto ret = libyuv::Android420ToI420(
        yBuffer->getDirectBytes(),
        yRowStride,
        uBuffer->getDirectBytes(),
        uvRowStride,
        vBuffer->getDirectBytes(),
        uvRowStride,
        uvPixelStride,
        yuvData.data(),
        imageWidth,
        yuvData.data() + imageHeight * imageWidth,
        halfImageWidth,
        yuvData.data() + imageHeight * imageWidth +
            halfImageHeight * halfImageWidth,
        halfImageHeight,
        imageWidth,
        imageHeight);
    ALOGI("JJJ libyuv::Android420ToI420 ret %d", ret);
    assert(ret == 0);
    const auto cvtRet = libyuv::I420ToARGB(
        yuvData.data(),
        imageWidth,
        yuvData.data() + imageHeight * imageWidth,
        halfImageWidth,
        yuvData.data() + imageHeight * imageWidth +
            halfImageHeight * halfImageWidth,
        halfImageWidth,
        rgbData.data(),
        4 * imageWidth,
        imageWidth,
        imageHeight);
    assert(cvtRet == 0);

    int cSize = imageHeight * imageWidth;

    for (int y = 0; y < std::min(5, imageHeight); y++) {
      for (int x = 0; x < imageWidth; x++) {
        auto idx = y * imageHeight + x;
        auto r = rgbData[cSize + idx];
        auto g = rgbData[2 * cSize + idx];
        auto b = rgbData[3 * cSize + idx];
        ALOGI("JJJ rgb(x: %d, y: %d) = (%d, %d, %d)", x, y, r, g, b);
      }
    }
  }

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod("nativePutYuvImage", PytorchVisionJni::putYuvImage),
        makeNativeMethod(
            "nativeImageYUV420CenterCropToFloatBuffer",
            PytorchVisionJni::nativeImageYUV420CenterCropToFloatBuffer),
    });
  }

  static void cameraOutputToAtTensor(facebook::jni::alias_ref<jclass>) {
    ALOGI("XXX cameraOutputToAtTensor()");
  }
};
} // namespace pytorch_vision_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { pytorch_vision_jni::PytorchVisionJni::registerNatives(); });
}