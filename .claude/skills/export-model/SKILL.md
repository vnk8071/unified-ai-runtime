---
name: export-model
description: >-
  Convert a PyTorch module or an ONNX file to a format a UAIRT backend loads (ONNX Runtime, CoreML, ExecuTorch, OpenVINO, NCNN, MNN)
  and check the result through UAIRT against the original. Use when the user has a model that no backend loads yet, asks which
  format to use for a device, or says "export", "convert" or "quantize" a model for UAIRT. It converts with the user's own packages and
  never accepts a model licence or downloads an SDK.
---

# Export a model for a backend

1. Read `AGENTS.md`, then `docs/agents/export-model.md` (the routes, which were run and which were not).
2. Ask first (section 1 of that page): where the model comes from and its licence (a gated model needs the user to accept it on
   the publisher's page), which device, whether a ready-made file in the target format exists, the accuracy or speed needed, and the
   **fixed input shape** to export with (UAIRT needs static shapes).
3. Convert with the tool, in a throw-away virtual environment (converter packages replace others, for example `MNN` replaced `protobuf`):

   ```bash
   scripts/export_model.py torchvision:resnet18 --shape 1,3,224,224 --to onnx openvino      # a PyTorch source
   scripts/export_model.py model.onnx --to ncnn mnn                                          # an ONNX file with fixed shapes
   ```

   It exports each target in its own process (ONNX Runtime, OpenVINO and PyTorch can deadlock in one) and runs the result through
   UAIRT. Run `--help` for the installed version first.
4. **Trust the comparison, not the export.** The tool compares raw outputs through UAIRT with PyTorch or ONNX Runtime on the same
   random input. Known precision traps: OpenVINO defaults to f16 on Arm (set `inference_precision=f32`), pnnx writes fp16 weights,
   NCNN defaults to fp16, MNN on a GPU computes in fp16. A difference of 1e-4 on logits of size 5 is rounding; 1e-2 means half
   precision somewhere; constant or NaN output means the export or the comparison is broken.
5. Report the file you made, the tool and version that made it, the largest difference from the original, and what you did not check
   (task accuracy, other shapes, other devices).

Stop and ask before: accepting a licence, using the user's access token, installing a vendor SDK, or quantizing (it changes the model).
