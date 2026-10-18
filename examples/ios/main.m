/* SPDX-License-Identifier: Apache-2.0 */
/*
 * A small iOS app that runs UAIRT on the device: the reference backend, then the CoreML backend linked in statically (an iOS app cannot
 * load plugins) on a model bundled in the app, once per compute-unit setting. Everything is printed to stdout, prefixed "UAIRT:", and
 * shown on screen; with the launch argument --exit-after the app quits when it is done, so `devicectl ... --console` returns.
 */
#import <AVFoundation/AVFoundation.h>
#import <Accelerate/Accelerate.h>
#import <PhotosUI/PhotosUI.h>
#import <UIKit/UIKit.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <uairt/uairt.h>
#include <uairt/uairt_static.h>

#include "detector.h"

static NSMutableString* g_log;

static void say(const char* fmt, ...) {
  char line[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(line, sizeof(line), fmt, args);
  va_end(args);
  printf("UAIRT: %s\n", line);
  fflush(stdout);
  @synchronized(g_log) {
    [g_log appendFormat:@"%s\n", line];
  }
}

static double now_ms(void) {
  return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6;
}

#define TRY(call)                                                                              \
  do {                                                                                         \
    uairt_status status_ = (call);                                                             \
    if (status_ != UAIRT_OK) {                                                                 \
      say("FAILED %s: %s (%s)", #call, uairt_status_string(status_), uairt_last_error());       \
      return 0;                                                                                \
    }                                                                                          \
  } while (0)

static int run_reference(void) {
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  TRY(uairt_engine_create("reference", NULL, 0, &engine));
  uairt_model_source source = {.struct_size = sizeof(source), .path = "unused"};
  TRY(uairt_model_load(engine, &source, &model));
  uairt_tensor input, output;
  TRY(uairt_model_input_info(model, 0, &input));
  TRY(uairt_model_output_info(model, 0, &output));
  float in_data[4] = {1.f, 2.f, 3.f, 4.f}, out_data[4] = {0};
  input.data = in_data;
  input.nbytes = sizeof(in_data);
  output.data = out_data;
  output.nbytes = sizeof(out_data);
  TRY(uairt_model_run(model, &input, 1, &output, 1));
  say("reference backend output: %g %g %g %g", out_data[0], out_data[1], out_data[2], out_data[3]);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
  return out_data[3] == 4.f;
}

/* uairt_model_input_info describes dtype and shape; the size in bytes is the caller's to work out. */
static size_t tensor_bytes(const uairt_tensor* tensor) {
  return (size_t)uairt_tensor_num_elements(tensor) * uairt_dtype_size(tensor->dtype);
}

/* The same deterministic input is generated on the Mac (tests/compare script in examples/ios/README.md), so outputs can be compared. */
static void fill_input(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    x[i] = (float)((i * 37) % 256) / 255.0f - 0.5f;
  }
}

static int run_coreml(const char* units, const char* model_path) {
  uairt_option option = {"compute_units", units};
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  double t0 = now_ms();
  TRY(uairt_engine_create("coreml", &option, 1, &engine));
  uairt_model_source source = {.struct_size = sizeof(source), .path = model_path};
  TRY(uairt_model_load(engine, &source, &model));
  double load_ms = now_ms() - t0;
  uairt_tensor input, output;
  TRY(uairt_model_input_info(model, 0, &input));
  TRY(uairt_model_output_info(model, 0, &output));
  input.nbytes = tensor_bytes(&input);
  output.nbytes = tensor_bytes(&output);
  size_t n_in = input.nbytes / sizeof(float), n_out = output.nbytes / sizeof(float);
  float in_data[12288], out_data[64];
  if (n_in > 12288 || n_out > 64) {
    say("unexpected model size");
    return 0;
  }
  fill_input(in_data, n_in);
  input.data = in_data;
  output.data = out_data;
  TRY(uairt_model_run(model, &input, 1, &output, 1));  /* first run: includes CoreML's own warm-up */
  const int runs = 50;
  double t1 = now_ms();
  for (int i = 0; i < runs; ++i) {
    TRY(uairt_model_run(model, &input, 1, &output, 1));
  }
  double per_run = (now_ms() - t1) / runs;
  char values[256] = "";
  for (size_t i = 0; i < n_out; ++i) {
    snprintf(values + strlen(values), sizeof(values) - strlen(values), "%s%.9g", i ? " " : "", out_data[i]);
  }
  say("coreml[%s] load %.1f ms, %.3f ms/run over %d runs", units, load_ms, per_run, runs);
  say("coreml[%s] output %s", units, values);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
  return 1;
}

/* An optional second model, bundled by the build: a detector's network stage (image in, class scores and boxes out) with one input image.
 * The raw outputs are saved to Documents/yolo_<units>_<n>.bin so they can be copied off the device and compared. */
static int run_detector(const char* units, const char* model_path, const char* input_path, NSString* documents) {
  uairt_option option = {"compute_units", units};
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  double t0 = now_ms();
  TRY(uairt_engine_create("coreml", &option, 1, &engine));
  uairt_model_source source = {.struct_size = sizeof(source), .path = model_path};
  TRY(uairt_model_load(engine, &source, &model));
  double load_ms = now_ms() - t0;
  uairt_tensor input, outputs[2];
  TRY(uairt_model_input_info(model, 0, &input));
  TRY(uairt_model_output_info(model, 0, &outputs[0]));
  TRY(uairt_model_output_info(model, 1, &outputs[1]));
  input.nbytes = tensor_bytes(&input);
  outputs[0].nbytes = tensor_bytes(&outputs[0]);
  outputs[1].nbytes = tensor_bytes(&outputs[1]);
  NSData* image = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:input_path]];
  if (image.length != input.nbytes) {
    say("FAILED: input image is %lu bytes, the model wants %zu", (unsigned long)image.length, input.nbytes);
    return 0;
  }
  float* scores = malloc(outputs[0].nbytes);
  float* boxes = malloc(outputs[1].nbytes);
  input.data = (void*)image.bytes;
  outputs[0].data = scores;
  outputs[1].data = boxes;
  TRY(uairt_model_run(model, &input, 1, outputs, 2));
  const int runs = 20;
  double t1 = now_ms();
  for (int i = 0; i < runs; ++i) {
    TRY(uairt_model_run(model, &input, 1, outputs, 2));
  }
  double per_run = (now_ms() - t1) / runs;
  size_t anchors = outputs[1].nbytes / (4 * sizeof(float)), classes = outputs[0].nbytes / sizeof(float) / anchors, above = 0;
  float best = 0;
  for (size_t a = 0; a < anchors; ++a) {
    float top = 0;
    for (size_t c = 0; c < classes; ++c) {
      top = scores[a * classes + c] > top ? scores[a * classes + c] : top;
    }
    above += top > 0.25f;
    best = top > best ? top : best;
  }
  say("detector[%s] load %.1f ms, %.2f ms/run; %zu anchors above 0.25, best score %.4f", units, load_ms, per_run, above, best);
  [[NSData dataWithBytes:scores length:outputs[0].nbytes] writeToFile:[documents stringByAppendingFormat:@"/yolo_%s_0.bin", units] atomically:YES];
  [[NSData dataWithBytes:boxes length:outputs[1].nbytes] writeToFile:[documents stringByAppendingFormat:@"/yolo_%s_1.bin", units] atomically:YES];
  free(scores);
  free(boxes);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
  return 1;
}

static void run_all(void) {
  say("UAIRT %s on %s %s", uairt_version_string(), UIDevice.currentDevice.model.UTF8String,
      UIDevice.currentDevice.systemVersion.UTF8String);
  int ok = run_reference();
  if (uairt_register_backend(uairt_backend_get_api_coreml) != UAIRT_OK) {
    say("FAILED to register the CoreML backend: %s", uairt_last_error());
    ok = 0;
  } else {
    NSString* model = [NSBundle.mainBundle pathForResource:@"convnet" ofType:@"mlmodelc"];
    if (!model) {
      say("FAILED: convnet.mlmodelc is not in the app bundle");
      ok = 0;
    } else {
      const char* units[] = {"cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"};
      for (size_t i = 0; i < 4; ++i) {
        ok = run_coreml(units[i], model.fileSystemRepresentation) && ok;
      }
    }
    NSString* detector = [NSBundle.mainBundle pathForResource:@"detector" ofType:@"mlmodelc"];
    NSString* image = [NSBundle.mainBundle pathForResource:@"detector_input" ofType:@"raw"];
    if (detector && image) {
      NSString* documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
      const char* units[] = {"cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"};
      for (size_t i = 0; i < 4; ++i) {
        ok = run_detector(units[i], detector.fileSystemRepresentation, image.fileSystemRepresentation, documents) && ok;
      }
    }
  }
  say("%s", ok ? "all checks passed" : "SOME CHECKS FAILED");
}

/* ---- live camera -------------------------------------------------------------------------------------------------------------- */

static const char* const kUnits[] = {"all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"};
enum { kInputSide = 640, kMaxDetections = 20 };

@interface OverlayView : UIView
@property(nonatomic, copy) NSArray<NSDictionary*>* items; /* @{@"rect": NSValue(CGRect), @"text": NSString} in view coordinates */
@end

@implementation OverlayView
- (void)drawRect:(CGRect)rect {
  CGContextRef ctx = UIGraphicsGetCurrentContext();
  NSDictionary* attributes = @{
    NSFontAttributeName : [UIFont boldSystemFontOfSize:13],
    NSForegroundColorAttributeName : UIColor.blackColor,
    NSBackgroundColorAttributeName : [UIColor.greenColor colorWithAlphaComponent:0.85]
  };
  CGContextSetStrokeColorWithColor(ctx, UIColor.greenColor.CGColor);
  CGContextSetLineWidth(ctx, 2);
  for (NSDictionary* item in self.items) {
    CGRect box = [item[@"rect"] CGRectValue];
    CGContextStrokeRect(ctx, box);
    [item[@"text"] drawAtPoint:CGPointMake(box.origin.x + 2, MAX(0, box.origin.y - 16)) withAttributes:attributes];
  }
}
@end

@interface PreviewView : UIView
@end
@implementation PreviewView
+ (Class)layerClass {
  return AVCaptureVideoPreviewLayer.class;
}
- (AVCaptureVideoPreviewLayer*)previewLayer {
  return (AVCaptureVideoPreviewLayer*)self.layer;
}
@end

/* Runs the detector on camera frames. Frames arrive on one serial queue, and the UAIRT engine and model are only touched there. */
@interface CameraDetector : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(readonly) AVCaptureSession* session;
@property(copy) void (^onResult)(NSArray<NSDictionary*>* items, double inference_ms, double fps);
@property(atomic) int unitsIndex;
@property(copy) NSString* modelPath;
- (instancetype)initWithModelPath:(NSString*)path;
- (void)start;
- (void)stop;
@end

@implementation CameraDetector {
  AVCaptureSession* _session;
  dispatch_queue_t _queue;
  uairt_engine* _engine;
  uairt_model* _model;
  int _loadedUnits;
  uint8_t* _input;
  float *_scores, *_boxes;
  uairt_tensor _in_info, _out_info[2];
  double _last_frame_ms, _fps;
}

- (instancetype)initWithModelPath:(NSString*)path {
  if ((self = [super init])) {
    _modelPath = path;
    _queue = dispatch_queue_create("uairt.camera", DISPATCH_QUEUE_SERIAL);
    _loadedUnits = -1;
  }
  return self;
}

- (AVCaptureSession*)session {
  if (!_session) {
    _session = [AVCaptureSession new];
    _session.sessionPreset = AVCaptureSessionPreset640x480;
    AVCaptureDevice* camera = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera
                                                                 mediaType:AVMediaTypeVideo
                                                                  position:AVCaptureDevicePositionBack];
    AVCaptureDeviceInput* input = camera ? [AVCaptureDeviceInput deviceInputWithDevice:camera error:nil] : nil;
    if (input && [_session canAddInput:input]) {
      [_session addInput:input];
    }
    AVCaptureVideoDataOutput* output = [AVCaptureVideoDataOutput new];
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
    output.alwaysDiscardsLateVideoFrames = YES;
    [output setSampleBufferDelegate:self queue:_queue];
    if ([_session canAddOutput:output]) {
      [_session addOutput:output];
      AVCaptureConnection* connection = [output connectionWithMediaType:AVMediaTypeVideo];
      if (@available(iOS 17.0, *)) {
        connection.videoRotationAngle = 90; /* portrait */
      } else {
        connection.videoOrientation = AVCaptureVideoOrientationPortrait;
      }
    }
  }
  return _session;
}

- (void)start {
  [self.session startRunning];
}

- (void)stop {
  [_session stopRunning];
}

/* Called on _queue. */
- (BOOL)ensureModel {
  int wanted = self.unitsIndex;
  if (_model && _loadedUnits == wanted) {
    return YES;
  }
  if (_model) {
    uairt_model_destroy(_model);
    uairt_engine_destroy(_engine);
    _model = NULL;
    _engine = NULL;
  }
  uairt_option option = {"compute_units", kUnits[wanted]};
  uairt_model_source source = {.struct_size = sizeof(source), .path = _modelPath.fileSystemRepresentation};
  if (uairt_engine_create("coreml", &option, 1, &_engine) != UAIRT_OK || uairt_model_load(_engine, &source, &_model) != UAIRT_OK) {
    say("camera: could not load the detector: %s", uairt_last_error());
    return NO;
  }
  uairt_model_input_info(_model, 0, &_in_info);
  uairt_model_output_info(_model, 0, &_out_info[0]);
  uairt_model_output_info(_model, 1, &_out_info[1]);
  _in_info.nbytes = tensor_bytes(&_in_info);
  _out_info[0].nbytes = tensor_bytes(&_out_info[0]);
  _out_info[1].nbytes = tensor_bytes(&_out_info[1]);
  free(_input);
  free(_scores);
  free(_boxes);
  _input = malloc(_in_info.nbytes);
  _scores = malloc(_out_info[0].nbytes);
  _boxes = malloc(_out_info[1].nbytes);
  _in_info.data = _input;
  _out_info[0].data = _scores;
  _out_info[1].data = _boxes;
  _loadedUnits = wanted;
  say("camera: detector loaded, compute_units=%s", kUnits[wanted]);
  return YES;
}

- (void)captureOutput:(AVCaptureOutput*)output didOutputSampleBuffer:(CMSampleBufferRef)buffer fromConnection:(AVCaptureConnection*)connection {
  if (![self ensureModel] || _in_info.nbytes != (size_t)kInputSide * kInputSide * 4) {
    return;
  }
  CVPixelBufferRef pixels = CMSampleBufferGetImageBuffer(buffer);
  CVPixelBufferLockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);
  /* The model takes a 640 x 640 BGRA image: the camera frame (BGRA, already upright) is stretched to it. */
  vImage_Buffer from = {CVPixelBufferGetBaseAddress(pixels), CVPixelBufferGetHeight(pixels), CVPixelBufferGetWidth(pixels),
                        CVPixelBufferGetBytesPerRow(pixels)};
  vImage_Buffer to = {_input, kInputSide, kInputSide, kInputSide * 4};
  vImageScale_ARGB8888(&from, &to, NULL, kvImageNoFlags);
  CVPixelBufferUnlockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);

  double inference_ms = 0;
  NSArray* items = [self inferWithMs:&inference_ms];
  if (!items) {
    return;
  }
  double now = now_ms();
  _fps = _last_frame_ms > 0 ? 0.8 * _fps + 0.2 * 1000.0 / (now - _last_frame_ms) : 0;
  _last_frame_ms = now;
  if (self.onResult) {
    self.onResult(items, inference_ms, _fps);
  }
}

/* Runs the model on what is in _input (a 640 x 640 BGRA image) and returns the detections as normalised boxes, or nil on an error.
 * Called on _queue. */
- (NSArray<NSDictionary*>*)inferWithMs:(double*)inference_ms {
  double t0 = now_ms();
  uairt_status status = uairt_model_run(_model, &_in_info, 1, _out_info, 2);
  *inference_ms = now_ms() - t0;
  if (status != UAIRT_OK) {
    say("detector: run failed: %s (%s)", uairt_status_string(status), uairt_last_error());
    return nil;
  }
  size_t anchors = _out_info[1].nbytes / (4 * sizeof(float)), classes = _out_info[0].nbytes / sizeof(float) / anchors;
  detection found[kMaxDetections];
  size_t n = detector_decode(_scores, _boxes, anchors, classes, 0.25f, 0.45f, found, kMaxDetections);
  NSMutableArray* items = [NSMutableArray arrayWithCapacity:n];
  for (size_t i = 0; i < n; ++i) { /* normalised boxes; the view maps them onto the displayed picture */
    [items addObject:@{
      @"box" : [NSValue valueWithCGRect:CGRectMake(found[i].x0, found[i].y0, found[i].x1 - found[i].x0, found[i].y1 - found[i].y0)],
      @"text" : [NSString stringWithFormat:@"%s %.0f%%", kCocoNames[found[i].class_id], found[i].score * 100],
      @"score" : @(found[i].score)
    }];
  }
  return items;
}

/* Detects objects in a still picture: it is drawn upright into the 640 x 640 input (stretched, like the camera frames). */
- (void)detectImage:(UIImage*)image completion:(void (^)(NSArray<NSDictionary*>*, double))completion {
  dispatch_async(_queue, ^{
    if (![self ensureModel] || self->_in_info.nbytes != (size_t)kInputSide * kInputSide * 4) {
      completion(nil, 0);
      return;
    }
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(self->_input, kInputSide, kInputSide, 8, kInputSide * 4, space,
                                                 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst); /* BGRA */
    CGColorSpaceRelease(space);
    CGContextTranslateCTM(context, 0, kInputSide);
    CGContextScaleCTM(context, 1, -1);
    UIGraphicsPushContext(context);
    [image drawInRect:CGRectMake(0, 0, kInputSide, kInputSide)];
    UIGraphicsPopContext();
    CGContextRelease(context);
    double ms = 0;
    NSArray* items = [self inferWithMs:&ms];
    completion(items, ms);
  });
}

- (void)dealloc {
  if (_model) {
    uairt_model_destroy(_model);
    uairt_engine_destroy(_engine);
  }
  free(_input);
  free(_scores);
  free(_boxes);
}
@end

/* ---- app ----------------------------------------------------------------------------------------------------------------------- */

@interface AppDelegate : UIResponder <UIApplicationDelegate, PHPickerViewControllerDelegate>
@property(strong) UIWindow* window;
@end

@implementation AppDelegate {
  UITextView* _log;
  PreviewView* _preview;
  OverlayView* _overlay;
  UILabel* _status;
  UIButton *_camera_button, *_units_button, *_choose_button, *_sample_button;
  UIImageView* _image_view;
  NSString* _sample_path;
  CameraDetector* _detector;
  BOOL _running, _self_test_done, _exit_after;
  NSString* _detector_path;
  /* summary for --camera-seconds, written from the camera queue */
  NSMutableDictionary<NSString*, NSNumber*>* _label_counts;
  int _frames;
  double _inference_total;
}

- (UIButton*)buttonWithTitle:(NSString*)title action:(SEL)action {
  UIButton* button = [UIButton buttonWithType:UIButtonTypeSystem];
  [button setTitle:title forState:UIControlStateNormal];
  button.titleLabel.font = [UIFont boldSystemFontOfSize:17];
  button.backgroundColor = [UIColor.systemBlueColor colorWithAlphaComponent:0.9];
  [button setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
  [button setTitleColor:[UIColor.whiteColor colorWithAlphaComponent:0.4] forState:UIControlStateDisabled];
  button.layer.cornerRadius = 10;
  [button addTarget:self action:action forControlEvents:UIControlEventTouchUpInside];
  button.translatesAutoresizingMaskIntoConstraints = NO;
  return button;
}

- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)options {
  g_log = [NSMutableString string];
  NSArray* arguments = NSProcessInfo.processInfo.arguments;
  _exit_after = [arguments containsObject:@"--exit-after"];
  _detector_path = [NSBundle.mainBundle pathForResource:@"detector" ofType:@"mlmodelc"];
  NSUInteger camera_seconds_at = [arguments indexOfObject:@"--camera-seconds"];

  self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
  UIViewController* controller = [UIViewController new];
  UIView* root = controller.view;
  root.backgroundColor = UIColor.blackColor;

  _log = [UITextView new];
  _log.editable = NO;
  _log.font = [UIFont monospacedSystemFontOfSize:11 weight:UIFontWeightRegular];
  _log.text = @"running the self-test...";
  _preview = [PreviewView new];
  _preview.hidden = YES;
  _preview.previewLayer.videoGravity = AVLayerVideoGravityResizeAspect;
  _overlay = [OverlayView new];
  _overlay.backgroundColor = UIColor.clearColor;
  _overlay.userInteractionEnabled = NO;
  _overlay.hidden = YES;
  _status = [UILabel new];
  _status.font = [UIFont monospacedSystemFontOfSize:12 weight:UIFontWeightMedium];
  _status.textColor = UIColor.whiteColor;
  _status.backgroundColor = [UIColor.blackColor colorWithAlphaComponent:0.6];
  _status.numberOfLines = 2;
  _camera_button = [self buttonWithTitle:@"Start camera" action:@selector(toggleCamera)];
  _units_button = [self buttonWithTitle:@"Compute: all" action:@selector(cycleUnits)];
  _choose_button = [self buttonWithTitle:@"Choose image" action:@selector(chooseImage)];
  _sample_button = [self buttonWithTitle:@"Sample image" action:@selector(showSample)];
  _camera_button.enabled = NO;
  _units_button.enabled = NO;
  _choose_button.enabled = NO;
  _sample_button.enabled = NO;
  _sample_path = [NSBundle.mainBundle pathForResource:@"sample" ofType:@"jpg"];
  _image_view = [UIImageView new];
  _image_view.contentMode = UIViewContentModeScaleAspectFit;
  _image_view.hidden = YES;
  for (UIView* view in @[ _log, _preview, _image_view, _overlay, _status, _camera_button, _units_button, _choose_button, _sample_button ]) {
    view.translatesAutoresizingMaskIntoConstraints = NO;
    [root addSubview:view];
  }
  UILayoutGuide* safe = root.safeAreaLayoutGuide;
  NSMutableArray* constraints = [NSMutableArray array];
  for (UIView* fill in @[ _log, _preview, _image_view, _overlay ]) {
    [constraints addObjectsFromArray:@[
      [fill.topAnchor constraintEqualToAnchor:safe.topAnchor],
      [fill.leadingAnchor constraintEqualToAnchor:root.leadingAnchor],
      [fill.trailingAnchor constraintEqualToAnchor:root.trailingAnchor],
      [fill.bottomAnchor constraintEqualToAnchor:_camera_button.topAnchor constant:-12]
    ]];
  }
  [constraints addObjectsFromArray:@[
    [_status.topAnchor constraintEqualToAnchor:safe.topAnchor],
    [_status.leadingAnchor constraintEqualToAnchor:root.leadingAnchor],
    [_status.trailingAnchor constraintEqualToAnchor:root.trailingAnchor],
    [_camera_button.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:16],
    [_camera_button.bottomAnchor constraintEqualToAnchor:_choose_button.topAnchor constant:-8],
    [_choose_button.leadingAnchor constraintEqualToAnchor:_camera_button.leadingAnchor],
    [_choose_button.bottomAnchor constraintEqualToAnchor:safe.bottomAnchor constant:-12],
    [_choose_button.heightAnchor constraintEqualToAnchor:_camera_button.heightAnchor],
    [_choose_button.widthAnchor constraintEqualToAnchor:_camera_button.widthAnchor],
    [_sample_button.leadingAnchor constraintEqualToAnchor:_units_button.leadingAnchor],
    [_sample_button.bottomAnchor constraintEqualToAnchor:_choose_button.bottomAnchor],
    [_sample_button.heightAnchor constraintEqualToAnchor:_camera_button.heightAnchor],
    [_sample_button.widthAnchor constraintEqualToAnchor:_camera_button.widthAnchor],
    [_camera_button.heightAnchor constraintEqualToConstant:48],
    [_units_button.leadingAnchor constraintEqualToAnchor:_camera_button.trailingAnchor constant:12],
    [_units_button.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-16],
    [_units_button.bottomAnchor constraintEqualToAnchor:_camera_button.bottomAnchor],
    [_units_button.heightAnchor constraintEqualToAnchor:_camera_button.heightAnchor],
    [_units_button.widthAnchor constraintEqualToAnchor:_camera_button.widthAnchor]
  ]];
  [NSLayoutConstraint activateConstraints:constraints];
  self.window.rootViewController = controller;
  [self.window makeKeyAndVisible];

  if (_detector_path) {
    __weak AppDelegate* weak = self;
    _detector = [[CameraDetector alloc] initWithModelPath:_detector_path];
    _detector.onResult = ^(NSArray<NSDictionary*>* items, double inference_ms, double fps) {
      [weak handleItems:items inference:inference_ms fps:fps];
    };
    _preview.previewLayer.session = _detector.session;
  }

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    run_all();
    dispatch_async(dispatch_get_main_queue(), ^{
      @synchronized(g_log) {
        self->_log.text = g_log;
      }
      self->_self_test_done = YES;
      self->_camera_button.enabled = self->_detector != nil;
      self->_units_button.enabled = self->_detector != nil;
      self->_choose_button.enabled = self->_detector != nil;
      self->_sample_button.enabled = self->_detector != nil && self->_sample_path != nil;
      if (!self->_detector) {
        self->_status.text = @"No detector model in this build: see examples/ios/README.md";
      }
      if (camera_seconds_at != NSNotFound && camera_seconds_at + 1 < arguments.count && self->_detector) {
        [self runCameraForSeconds:[arguments[camera_seconds_at + 1] doubleValue]];
      } else if ([arguments containsObject:@"--image-test"] && self->_detector && self->_sample_path) {
        [self runImageTest];
      } else if ([arguments containsObject:@"--show-sample"] && self->_detector && self->_sample_path) {
        [self showSample];
      } else if (self->_exit_after) {
        exit(0);
      }
    });
  });
  return YES;
}

/* Maps normalised boxes onto `picture` (the displayed video or image, in the overlay's coordinates) and redraws. */
- (void)drawItems:(NSArray<NSDictionary*>*)items inRect:(CGRect)picture {
  NSMutableArray* drawn = [NSMutableArray arrayWithCapacity:items.count];
  for (NSDictionary* item in items) {
    CGRect n = [item[@"box"] CGRectValue];
    CGRect box = CGRectMake(picture.origin.x + n.origin.x * picture.size.width, picture.origin.y + n.origin.y * picture.size.height,
                            n.size.width * picture.size.width, n.size.height * picture.size.height);
    [drawn addObject:@{@"rect" : [NSValue valueWithCGRect:box], @"text" : item[@"text"]}];
  }
  _overlay.items = drawn;
  [_overlay setNeedsDisplay];
}

- (void)handleItems:(NSArray<NSDictionary*>*)items inference:(double)ms fps:(double)fps {
  dispatch_async(dispatch_get_main_queue(), ^{
    [self drawItems:items inRect:[self->_preview.previewLayer rectForMetadataOutputRectOfInterest:CGRectMake(0, 0, 1, 1)]];
    for (NSDictionary* item in items) {
      NSString* name = [item[@"text"] substringToIndex:[item[@"text"] rangeOfString:@" " options:NSBackwardsSearch].location];
      self->_label_counts[name] = @(self->_label_counts[name].intValue + 1);
    }
    self->_frames += 1;
    self->_inference_total += ms;
    self->_status.text = [NSString stringWithFormat:@" %s | inference %.1f ms | %.0f fps | %lu objects", kUnits[self->_detector.unitsIndex], ms,
                                                    fps, (unsigned long)items.count];
  });
}

/* ---- still images ---- */

- (void)showPicture:(UIImage*)image title:(NSString*)title {
  if (_running) {
    [self setRunning:NO];
  }
  _log.hidden = YES;
  _image_view.image = image;
  _image_view.hidden = NO;
  _overlay.hidden = NO;
  _overlay.items = nil;
  [_overlay setNeedsDisplay];
  _status.text = [NSString stringWithFormat:@" %@: detecting...", title];
  [_detector detectImage:image completion:^(NSArray<NSDictionary*>* items, double ms) {
    dispatch_async(dispatch_get_main_queue(), ^{
      if (!items) {
        self->_status.text = @" detection failed (see the console)";
        return;
      }
      [self->_overlay layoutIfNeeded];
      CGRect picture = AVMakeRectWithAspectRatioInsideRect(image.size, self->_image_view.bounds);
      [self drawItems:items inRect:picture];
      self->_status.text = [NSString stringWithFormat:@" %@ | %s | inference %.1f ms | %lu objects", title, kUnits[self->_detector.unitsIndex],
                                                      ms, (unsigned long)items.count];
    });
  }];
}

- (void)showSample {
  UIImage* image = _sample_path ? [UIImage imageWithContentsOfFile:_sample_path] : nil;
  if (image) {
    [self showPicture:image title:@"sample image"];
  }
}

- (void)chooseImage {
  PHPickerConfiguration* configuration = [PHPickerConfiguration new];
  configuration.filter = PHPickerFilter.imagesFilter;
  configuration.selectionLimit = 1;
  PHPickerViewController* picker = [[PHPickerViewController alloc] initWithConfiguration:configuration];
  picker.delegate = self;
  [self.window.rootViewController presentViewController:picker animated:YES completion:nil];
}

- (void)picker:(PHPickerViewController*)picker didFinishPicking:(NSArray<PHPickerResult*>*)results {
  [picker dismissViewControllerAnimated:YES completion:nil];
  NSItemProvider* provider = results.firstObject.itemProvider;
  if (![provider canLoadObjectOfClass:UIImage.class]) {
    return;
  }
  [provider loadObjectOfClass:UIImage.class completionHandler:^(UIImage* image, NSError* error) {
    if (image) {
      dispatch_async(dispatch_get_main_queue(), ^{ [self showPicture:image title:@"chosen image"]; });
    }
  }];
}

/* Test mode: detect on the bundled sample and print the detections to stdout. */
- (void)runImageTest {
  UIImage* image = [UIImage imageWithContentsOfFile:_sample_path];
  [_detector detectImage:image completion:^(NSArray<NSDictionary*>* items, double ms) {
    say("image: %lu detections, %.1f ms, compute_units=%s", (unsigned long)items.count, ms, kUnits[self->_detector.unitsIndex]);
    for (NSDictionary* item in items) {
      CGRect box = [item[@"box"] CGRectValue];
      say("image: %s at [%.0f, %.0f, %.0f, %.0f] of %.0fx%.0f", [item[@"text"] UTF8String], box.origin.x * image.size.width,
          box.origin.y * image.size.height, (box.origin.x + box.size.width) * image.size.width,
          (box.origin.y + box.size.height) * image.size.height, image.size.width, image.size.height);
    }
    say("%s", items ? "image check passed" : "image check FAILED");
    if (self->_exit_after) {
      exit(items ? 0 : 1);
    }
  }];
}

- (void)setRunning:(BOOL)running {
  _running = running;
  _preview.hidden = !running;
  _overlay.hidden = !running;
  if (running) {
    _image_view.hidden = YES;
    _log.hidden = YES;
  } else if (_image_view.hidden) {
    _log.hidden = NO;
  }
  [_camera_button setTitle:running ? @"Stop camera" : @"Start camera" forState:UIControlStateNormal];
  if (running) {
    _label_counts = [NSMutableDictionary dictionary];
    _frames = 0;
    _inference_total = 0;
    [_detector start];
  } else {
    [_detector stop];
    _overlay.items = nil;
    [_overlay setNeedsDisplay];
  }
}

- (void)toggleCamera {
  if (_running) {
    [self setRunning:NO];
    return;
  }
  [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
    dispatch_async(dispatch_get_main_queue(), ^{
      if (granted) {
        [self setRunning:YES];
      } else {
        self->_status.text = @" Camera access is off: Settings > Privacy > Camera";
        say("camera: access was not granted");
      }
    });
  }];
}

- (void)cycleUnits {
  _detector.unitsIndex = (_detector.unitsIndex + 1) % 4;
  [_units_button setTitle:[NSString stringWithFormat:@"Compute: %s", kUnits[_detector.unitsIndex]] forState:UIControlStateNormal];
}

/* Test mode: run the camera for a while, print a summary to stdout, and quit if asked to. */
- (void)runCameraForSeconds:(double)seconds {
  [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL granted) {
    dispatch_async(dispatch_get_main_queue(), ^{
      if (!granted) {
        say("camera: access was not granted (allow it on the phone once, then run again)");
        if (self->_exit_after) exit(1);
        return;
      }
      [self setRunning:YES];
      dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        int frames = self->_frames;
        say("camera: %d frames in %.0f s, %.2f ms average inference, compute_units=%s", frames, seconds,
            frames ? self->_inference_total / frames : 0.0, kUnits[self->_detector.unitsIndex]);
        for (NSString* name in [self->_label_counts.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
          say("camera: saw %s in %d detections", name.UTF8String, self->_label_counts[name].intValue);
        }
        say("%s", frames > 0 ? "camera check passed" : "camera check FAILED: no frames");
        [self setRunning:NO];
        if (self->_exit_after) {
          dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 2), dispatch_get_main_queue(), ^{ exit(frames > 0 ? 0 : 1); });
        }
      });
    });
  }];
}
@end

int main(int argc, char* argv[]) {
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
  }
}
