# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's whisper.cpp backend with whisper.cpp's own `whisper-cli` on the same model and audio.

Both run on the CPU with greedy decoding and whisper's default timestamp tokens (`-nt` would change the decoding). The audio is a 16 kHz mono 16-bit WAV file (whisper.cpp's samples/jfk.wav is one).
UAIRT gets the samples as float32 and an `n_samples` equal to the clip length; the comparison ignores whitespace and case.

    python tests/compare_with_whisper_cli.py build/run_model build/libuairt_backend_whispercpp.so \
        whisper-cli ggml-tiny.en.bin jfk.wav
"""
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


def normalise(text):
    text = re.sub(r"\[\d\d:\d\d:\d\d\.\d+ --> \d\d:\d\d:\d\d\.\d+\]", " ", text)  # whisper-cli's segment timestamps
    return re.sub(r"\s+", " ", text).strip().lower()


def main(run_model, plugin, cli, model, audio):
    with wave.open(audio, "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2), "need 16 kHz mono 16-bit WAV"
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    # "en" fixes the language; "auto" makes whisper.cpp detect it and then transcribe (a build that only detected it returned no text)
    for language in ("en", "auto"):
        expected = subprocess.run([cli, "-m", model, "-f", audio, "-ng", "-l", language, "-bs", "1"], check=True,
                                  capture_output=True, text=True).stdout
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples.tofile(tmp / "in0.bin")
            subprocess.run([run_model, "--option", f"n_samples={samples.size}", "--option", "use_gpu=false",
                            "--option", f"language={language}", plugin, "whispercpp", model, str(tmp / "out"), str(tmp / "in0.bin")],
                           check=True)
            length = int(np.fromfile(tmp / "out1.bin", dtype=np.int32)[0])
            got = bytes(np.fromfile(tmp / "out0.bin", dtype=np.uint8)[:length]).decode("utf-8")

        assert normalise(expected), "whisper-cli produced no text; the comparison proves nothing"
        assert normalise(got) == normalise(expected), f"language={language}\nUAIRT: {got!r}\nwhisper-cli: {expected!r}"
        print(f"language={language}: transcript ({length} bytes) matches whisper-cli: {normalise(got)!r}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
