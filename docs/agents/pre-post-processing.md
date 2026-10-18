<!-- SPDX-License-Identifier: Apache-2.0 -->
# Pre-processing, post-processing, tokenizers and decoding

Goal: the model loads and `model.inputs` / `model.outputs` are known, and you must turn a real image, sound or sentence into the
tensors it takes, and its tensors into an answer a person can use. UAIRT runs the tensors and nothing else: everything before and
after is the application's. This page is the order to work in. The rules in `AGENTS.md` apply; running and checking a model is
[run-model.md](run-model.md), and the program that wraps all this is [write-app.md](write-app.md).

The loop: **inspect** the model, **identify** what kind it is from its inputs and outputs, **find the reference pipeline** that
produced it, **implement** the pre- and post-processing, then **verify** against that reference on the same input. Do not skip the
last step: a wrong mean, a swapped channel or a stale tokenizer still produces plausible-looking output.

What was run, so you know how far to trust each recipe: the helper code, NMS (against `torchvision.ops.nms` on 150 random
cases), the ImageNet resize/crop/normalize recipe (against torchvision on 8 image shapes), classification through UAIRT against
PyTorch, a BERT tokenizer feeding a UAIRT model, a YOLO-shaped head decoded back to image coordinates, GGUF tokenization against
Hugging Face's tokenizer, the chat template, stop tokens, seeded sampling, and an ffmpeg-converted recording through Whisper
all ran on an Apple M5. Not run here: log-mel feature extraction, CTC with a real speech model, segmentation, depth, quantized
TFLite or QNN models, a trained detector. Those sections say "pattern".

## 1. Principles

- UAIRT never converts. An input must match the model's dtype and shape exactly or the call fails with a message naming the tensor
  (`input 'input' needs float32 (1, 3, 224, 224), got ...`). Shapes are static, so resize, crop, pad or truncate to exactly the shape.
- The numbers (size, mean, std, scale, colour order, thresholds, labels, tokenizer, template) belong to the model, not to this
  repository. Take them from its model card or its original framework, never from a default in this page.
- Compare raw outputs numerically against the reference. Matching top-1 or top-5 is a weak check (section 8 shows a wrong
  preprocessing step that keeps the top-5 identical).
- Say what you could not verify. A recipe you only wrote, without a reference to compare against, is "not verified", not "done".

## 2. Inspect, then find the reference pipeline

```bash
python examples/python/inspect_model.py model.onnx                       # inputs, outputs, and a heuristic guess
python examples/python/inspect_model.py yolov8n_ncnn_model --option input_shapes=3x640x640   # NCNN needs the shapes
python examples/python/inspect_model.py ggml-tiny.en.bin --backend whispercpp --option n_samples=176000   # a .bin goes to QNN unless forced
build/run_model --info <plugin> <backend> <model>                                  # the same table from C, without Python
```

The Python script needs the binding set up (a shared build of the library, `PYTHONPATH`, `UAIRT_LIBRARY`, NumPy); if it is not, the script says so and prints the exact
commands, using an existing shared build when it finds one. `run_model --info` needs none of that.

The table (name, dtype, shape, scale and zero point when quantized) is what the backend reports for the loaded model. The "Looks like"
lines are a heuristic from names and shapes (section 3): use them to know what to look for, not as an answer. A GGUF or a server model
prints no tensors; that means a language model (sections 6 and 7). `vocab_size` is a property, not a method.

Then find the pipeline that made the model, in this order:

1. The model card or README where the file came from (Hugging Face model page, torchvision/timm docs, the ultralytics export).
2. Machine-readable config next to the weights: `preprocessor_config.json` (image size, mean, std, resample), `config.json`
   (labels in `id2label`, number of classes), `tokenizer.json` and `tokenizer_config.json`, `generation_config.json`,
   `special_tokens_map.json`. For torchvision, `weights.transforms()` prints the exact recipe. For ultralytics, the exported file's
   metadata has `imgsz` and `names`.
3. The file itself: an ONNX file's `metadata_props` (`onnx.load(path).metadata_props`), a GGUF's key/value metadata
   (`gguf.GGUFReader`, section 7), a TFLite file's metadata.
4. The original code that trained or exported it.

If none of these gives a number you need, say so and ask, or try the plausible alternatives and report which matched the reference.
Never invent class labels or normalization constants.

## 3. What the inputs and outputs tell you

This is what `inspect_model.py` encodes; confirm each row with the model card.

| Signature | Probably | Read |
|---|---|---|
| input float/uint8 `[1,3,H,W]` (or `[1,H,W,3]`) | an image model; the layout is where the 3 is | section 4.1 |
| output `[1,K]` after an image input | a classifier with K classes (or an embedding) | 5.1 |
| output `[1,4+C,A]` or `[1,A,5+C]`, A in the thousands | a YOLO-style detector head (features x anchors) | 5.2 |
| outputs named boxes, scores, classes | a detector with the decoding and NMS built in (SSD style) | 5.2 |
| output `[1,C,H',W']` | segmentation (C classes), depth (C=1) or heatmaps | 5.3 |
| input int `input_ids` `[1,L]` (+ `attention_mask`) | a text model; L is the fixed length | 6.2 |
| output `[1,L,H]` / `[1,K]` after text input | token embeddings or token labels / sequence logits or a pooled vector | 5.4 |
| `start_logits`, `end_logits` | extractive question answering | 5.4 |
| inputs named `past_key_values.*` | a decoder with a KV cache: dynamic shapes, not supported by UAIRT | 7.7 |
| input float `[N]` or `[1,N]`, N in the thousands | a waveform of N samples | 4.3 |
| input float `[1,80,3000]` (or 64/128 mel bins) | log-mel features, computed by the model's own feature extractor | 4.3 |
| output `[1,T,V]` after audio | CTC logits (speech recognition) | 5.5 |
| output uint8 `text` + int32 `text_length` | the whispercpp convention: UTF-8 bytes and their length | `docs/backends/whispercpp.md` |
| no tensors at all | a GGUF language model or a server model | 6.1, 7 |
| integer inputs/outputs with a scale and zero point | a quantized model | 4.2, 5.6 |

The helpers named below are in `examples/python/prepost.py` (NumPy, Pillow for image files; tests in
`bindings/python/tests/test_prepost.py`). Use them with `sys.path.insert(0, "examples/python"); import prepost`.

## 4. Pre-processing

### 4.1 Images (run here)

Decide each of these from the reference pipeline, in this order:

| Step | The choices | If you get it wrong |
|---|---|---|
| Decode | Pillow gives RGB, OpenCV gives BGR | colours swapped (`bgr=True` fixes a BGR model) |
| Resize | straight to H x W; or short side then centre crop (the ImageNet recipe, 256 then 224); or letterbox (keep the aspect ratio, pad) | stretched or cropped content, shifted boxes |
| Range | 0-255 as is (integer models), 0-1, or -1..1 | confident but wrong classes |
| Normalize | `(x - mean) / std` per channel, in the 0-1 range, or none | same |
| Layout | NCHW (channels first) or NHWC | UAIRT refuses a wrong shape |
| Type | float32, float16, uint8/int8 (quantized: section 4.2) | UAIRT refuses a wrong dtype |
| Batch | a leading dimension of 1 | wrong shape |

```python
import sys; sys.path.insert(0, "examples/python")
import numpy as np, uairt, prepost

image = prepost.load_rgb("photo.jpg")
x = prepost.to_tensor(prepost.resize_center_crop(image, 224),
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))   # the model card's numbers, not a default
with uairt.AutoModel.from_file("resnet18.onnx", device="cpu") as model:
    info = model.inputs[0]
    assert x.dtype == info.dtype and x.shape == info.shape, (x.dtype, x.shape, info)
    (logits,) = model.run(x)
```

`resize_center_crop` follows torchvision's rounding exactly (it matches `Resize(256) + CenterCrop(224) + ToTensor + Normalize` to
7e-7 on 8 image shapes); another framework may round differently. For a detector use `prepost.letterbox(image, 640)`, which returns
the canvas and `(scale, pad_x, pad_y)` to keep for section 5.2, then `to_tensor(canvas)`.

### 4.2 Quantized inputs (pattern)

A model with integer inputs and a non-zero scale takes `round(x / scale) + zero_point`, clipped to the dtype. Preprocess to the
real-valued tensor first (section 4.1), then quantize with the printed numbers:

```python
info = model.inputs[0]                                          # info.quant_scale, info.quant_zero_point, info.dtype
q = prepost.quantize(x, info.quant_scale, info.quant_zero_point, info.dtype)
```

Some quantized image models take raw uint8 pixels with scale 1/255 and zero point 0, which makes `q` equal the pixels; check
the numbers, do not assume. Quantized models are less forgiving: compare accuracy on several inputs, not one.

### 4.3 Audio (run here for the waveform; the mel step is a pattern)

A waveform input wants mono samples as float32 in [-1, 1] at the model's sample rate (usually 16 kHz), exactly N samples:

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -f f32le recording.f32        # any format to 16 kHz mono float32 (run here: jfk.mp3 -> 176000 samples)
```

```python
pcm = np.fromfile("recording.f32", dtype=np.float32)
n = model.inputs[0].shape[-1]
pcm = np.pad(pcm, (0, max(0, n - len(pcm))))[:n]                # pad with silence or trim to the static length
```

For the whispercpp backend `n_samples` is an engine option, so give the real length (`len(pcm)`) and avoid decoding padding. Recordings
that are already 16 kHz mono 16-bit WAV can be read with the standard `wave` module and divided by 32768 (`uairt_backends_tour.ipynb`
does this). Log-mel inputs (`[1,80,3000]` and similar) are computed by the model's own feature extractor (for Whisper,
`transformers.WhisperFeatureExtractor`); reimplementing the mel filterbank by hand is a common source of silent errors.

## 5. Post-processing

### 5.1 Classification (run here)

```python
probs = prepost.softmax(logits[0])           # skip when the model already ends in a softmax (outputs in [0,1] summing to 1)
indices, values = prepost.topk(probs, 5)
labels = ...                                  # id2label from the model's config.json, or the dataset's class list; never guess
```

### 5.2 Detection (run on a crafted head; a trained YOLO was run only through TensorRT, `detect_tensorrt.py`)

YOLOv8/v11 exports give `[1, 4 + classes, anchors]` with no objectness; YOLOv5 exports give `[1, anchors, 5 + classes]` with it.
Rows are centre x, centre y, width, height in the pixels of the *model input*. Exports with NMS built in have a different layout
(then there is nothing to decode); check the output names and shapes first.

```python
canvas, scale, pad_x, pad_y = prepost.letterbox(image, 640)
(out,) = model.run(prepost.to_tensor(canvas))
boxes, scores, classes = prepost.decode_yolo(out, conf=0.25, iou=0.45)          # objectness=True for v5; channels_first= if ambiguous
boxes = prepost.undo_letterbox(boxes, scale, pad_x, pad_y, image.width, image.height)   # back to the original image
```

The thresholds are choices, not properties of the model: report the ones you used. On a crafted head with one known box, a duplicate and a
weak candidate, this returned exactly the expected box in the original image (`[80, 40, 120, 60]`) and suppressed the other two.
`decode_yolo` raises a clear error for a shape it cannot read instead of returning nonsense.

### 5.3 Dense maps (pattern)

Segmentation: `mask = out[0].argmax(axis=0)` gives a class id per pixel at the output resolution; resize it to the image with
nearest-neighbour (never bilinear on class ids). Depth (one channel): resize bilinearly and say the values are relative unless the
card says metric. Heatmaps: `argmax` per keypoint channel, then scale the coordinates back by the stride.

### 5.4 Text-model outputs (run for the shapes; pattern for the heads)

- Sentence embedding from token features `[1,L,H]`: `prepost.l2_normalize(prepost.mean_pool(hidden, attention_mask))`. If the model
  has a pooled output (`pooler_output`), use it only if the card says that is the embedding.
- Sequence classification `[1,K]`: `softmax` then `argmax`; the labels are `id2label` in `config.json`.
- Token classification `[1,L,K]`: `argmax` over the last axis, ignoring padded positions (`attention_mask == 0`) and the special tokens.
- Question answering: `start = start_logits.argmax()`, `end = end_logits.argmax()`; check `end >= start`, then map the token span back
  to characters with the tokenizer's offsets.

### 5.5 Speech recognition with CTC (pattern; the decoder is unit tested)

`ids = prepost.ctc_greedy(logits[0], blank=0)` merges repeats and drops the blank (the blank index is the model's, often 0). Turn the ids
into text with the model's vocabulary (`vocab.json` or the tokenizer), and replace the word-delimiter symbol with a space.

### 5.6 Quantized outputs (pattern)

`real = model.outputs[0].dequantize(q)` (or `prepost.dequantize(q, scale, zero_point)`), then continue as above. Apply it before softmax.

## 6. Tokenizers

The tokenizer is part of the model: the same vocabulary, normalization and special tokens it was trained with. Use the file published
with the model, at the same revision. A different tokenizer, or the same one with a different `add_special_tokens`, gives
confident garbage.

### 6.1 GGUF models (llamacpp), run here

```python
with uairt.AutoModel.from_file("model.gguf", device="cpu") as model:
    ids = model.tokenize("Hello, world!")                      # numpy int32; add_special=True by default (adds BOS when the model wants one)
    text = model.detokenize(ids)                               # str; detokenize_bytes(ids) for raw bytes
    n_vocab = model.vocab_size                                 # a property
```

On SmolLM2-135M-Instruct the ids equalled Hugging Face's `tokenizers` output on four strings (plain, Unicode, odd whitespace, and a full
chat prompt with special tokens), and `detokenize(tokenize(x)) == x` held. `tokenize` parses special-token text such as `<|im_end|>`
into control tokens, so do not pass untrusted user text through a prompt template without thinking about that.

### 6.2 Encoder models on ONNX and other tensor backends, run here (with a synthetic model)

The model takes integers of a fixed length L, so pad and truncate to exactly L and send the dtype the model declares (usually int64):

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")                    # from the model's repository
length = model.inputs[0].shape[1]
tok.enable_truncation(max_length=length); tok.enable_padding(length=length)
enc = tok.encode("Unified runtimes are useful.")
ids  = np.array([enc.ids], dtype=np.int64)
mask = np.array([enc.attention_mask], dtype=np.int64)          # 1 for real tokens, 0 for padding
outputs = model.run(ids, mask)                                 # add token_type_ids if the model lists it
```

A real BERT tokenizer (`bert-base-uncased`) padded a 9-token sentence to 16 and truncated a long one to 16, and both ran. The tokenizer
adds `[CLS]` first and `[SEP]` last itself; do not add them again. Inputs appear in the order `model.inputs` lists them.

## 7. Decoding and language models

### 7.1 The chat template (run here)

UAIRT tokenizes text but does not apply a chat template. A chat model expects its own format around each turn. The template is stored in
the GGUF, so read it instead of remembering it:

```python
from gguf import GGUFReader                                    # pip install gguf
field = GGUFReader("model.gguf").fields["tokenizer.chat_template"]
template = bytes(field.parts[field.data[0]]).decode()          # a Jinja2 template; the same keys exist for the BOS/EOS ids
import jinja2
prompt = jinja2.Template(template).render(messages=[{"role": "user", "content": "Capital of France?"}], add_generation_prompt=True)
```

For SmolLM2-135M-Instruct this renders ChatML, including a default system message, and a prompt built by hand with the same layout was
identical to the rendered one:

```text
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
Capital of France?<|im_end|>
<|im_start|>assistant
```

Render with the template, or compare your hand-built prompt with it. Other models differ (Llama 3 uses `<|start_header_id|>`, Gemma
`<start_of_turn>`), and a prompt in the wrong format still gets an answer, just a worse one. Also read `tokenizer.ggml.add_bos_token`
and the BOS/EOS ids from the same metadata.

### 7.2 Stop tokens (run here)

`generate` does not stop at the model's end-of-turn token by itself. Get its id from the tokenizer and pass it:

```python
stop = model.tokenize("<|im_end|>", add_special=False).tolist()           # [2] for SmolLM2: exactly one id
reply = "".join(uairt.generate(model, prompt, max_tokens=200, stop_tokens=stop))
```

With the stop token the reply was `The capital of France is Paris.`. Without it the same call ran on into `<|im_end|>`, an empty system turn and
more control tokens until `max_tokens`. Always set both `stop_tokens` and `max_tokens`.

### 7.3 Sampling (run here)

`uairt.generate(model, prompt, max_tokens=64, temperature=0.0, top_k=0, top_p=1.0, seed=None, stop_tokens=())`. Temperature 0 is greedy and
deterministic on one device. Otherwise the logits are divided by the temperature, cut to the top-k (0 means no limit), then to the top-p
nucleus, and one token is drawn. The same `seed` gave the same text twice and a different seed gave different text. Greedy output can differ
between the CPU and the GPU (floating-point), so compare like with like. Use low temperature for facts and extraction, higher with
`top_p` around 0.9 for creative text; say which you used.

### 7.4 Context

The context window is `n_ctx` (default 2048, capped by the model); a generation that reaches it fails mid-stream. Count prompt tokens with
`len(model.tokenize(prompt))`, leave room for `max_tokens`, and set `n_ctx` on the session when you need more
(`model.session({"n_ctx": "8192"})` passed as `session=` to `generate`). Chat history grows each turn: keep a session and append only new
tokens, or trim old turns.

### 7.5 Your own decoding loop (pattern)

When you need constrained output, a custom sampler, a repetition penalty or the log-probabilities, drive the session yourself:

```python
with model.session(n_ctx=2048) as session:
    session.append(model.tokenize(prompt))
    for _ in range(max_tokens):
        logits = session.logits()                              # next-token scores, numpy float32, length vocab_size
        token = int(your_sampler(logits))                     # yours: greedy is np.argmax(logits); bindings/python/uairt/_generate.py has the library's, as NumPy
        if token in stop:
            break
        session.append([token])
```

### 7.6 Servers

The llmserver backend sends text to a running Ollama, LM Studio or SGLang server over the OpenAI chat completions API and streams text back
(`uairt_model_complete`); there are no tensors, tokens or sessions to handle, so none of 6.1 to 7.5 applies. Because it uses the chat
endpoint the server applies the model's chat template, and the `system` option sets the system message. See
`docs/backends/llmserver.md` for the options.

### 7.7 What UAIRT does not do

- ONNX or other tensor-level decoders with a **KV cache as inputs** (`past_key_values.*`): their cache shapes grow, and UAIRT needs
  static shapes. Use a GGUF of the model with the llamacpp backend.
- **Speech recognition decoding** for Whisper is inside whisper.cpp (greedy; options `language`, `translate`); you only convert the audio
  (4.3) and read the text bytes (`text[:int(length[0])].decode()`).
- Beam search and speculative decoding are not offered; a session loop (7.5) is the way to build your own.

## 8. Verify against the reference

Run the original pipeline on the same input and compare *raw outputs*, then the decoded result.

| Kind | Reference | Compare |
|---|---|---|
| image classifier | torchvision/timm/ONNX Runtime in Python | logits within about 1e-4 (float32); the top-5 as a second check, never the only one |
| detector | the framework's own predict (for example ultralytics) | the same boxes, classes and scores within a few pixels and a few percent, with the same thresholds |
| text encoder | `transformers` on the same text | the embeddings or logits, and the token ids from the tokenizer |
| language model | llama.cpp's own tool (`tests/compare_with_llamacpp.py`), the same prompt, greedy | the same tokens, on the same device setting; plus Hugging Face's tokenizer ids |
| speech | the model's own recogniser (`whisper-cli`, the Python pipeline) | the same text, ignoring case and spacing |
| quantized | the float model | accuracy on several inputs, not one |

A real example of why numbers beat top-k: ResNet-18 through UAIRT against PyTorch on one photo. The correct recipe differed from PyTorch by
2e-6 in the logits. Dropping the mean/std changed the logits by 1.84 and the top-5; reading the colours as BGR changed them by 0.84 yet the
top-5 **stayed identical**, and feeding 0-255 pixels changed them by 154 with the top-5 also identical. Only the raw comparison caught all
three. (Random weights, so the classes mean nothing here; the point is the check.)

Report what you checked, on what input, and what you did not: the labels, the thresholds, the device, and any step you could not compare.

## 9. When it looks wrong

| Symptom | Likely cause |
|---|---|
| `needs float32 (1, 3, 224, 224), got ...` | layout, dtype or batch dimension; section 4.1 |
| plausible but wrong classes | range (0-255 vs 0-1), mean/std, BGR/RGB, or resize policy |
| detector boxes shifted or too big | letterbox padding or scale not undone; coordinates normalized vs pixels |
| no or hundreds of detections | threshold too high or low, NMS missing, class scores already sigmoid-ed or not |
| text model output is noise | wrong tokenizer file, special tokens added twice or missing, padding side, no attention mask |
| language model rambles or repeats turns | no stop token, wrong chat template, no `max_tokens` |
| language model answers badly but fluently | prompt not in the chat format; system prompt missing |
| output differs between CPU and GPU | floating-point; compare each with the reference on the same device |
| speech output empty or hallucinated | wrong sample rate, stereo not mixed down, padding decoded as speech, clip shorter than 100 ms |
| `needs ... got float64` | NumPy defaulted to float64; `astype(info.dtype)` |

Related: [export-model.md](export-model.md) (getting a model into a loadable format), [troubleshooting.md](troubleshooting.md) (load and
build errors), `examples/python/` (`detect_tensorrt.py` is a complete detector, `llm_generate.py` a complete language model).
