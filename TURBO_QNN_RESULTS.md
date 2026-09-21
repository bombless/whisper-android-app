# Whisper-Large-V3-Turbo on QNN: migration result

Status: **PASS on RMX3800 / SM8650 / HTP v75**, both for a fixed WAV and for the live
microphone loop in the app.

This records the move from Whisper-Tiny to Whisper-Large-V3-Turbo on the Qualcomm QNN
HTP backend. The Tiny path was kept working and is used as a cross-check.

## What was missing

The downloaded `qnn_context_binary` asset was byte-complete, but it could not run: a raw
QAIRT context binary is not an ONNX model, and three pieces of the contract were absent.

| Piece | Before | Now |
|---|---|---|
| EPContext ONNX wrappers | only Tiny's existed | generated for Turbo by `tools/build_turbo_epcontext.py` |
| Large-V3 tokenizer | only the Tiny 51865-entry vocab existed | `vocab.json` / `merges.txt` / `added_tokens.json` fetched from `openai/whisper-large-v3-turbo` |
| 128-bin mel frontend | `WhisperTurboFeatureExtractor` could only do a full 30 s chunk | added an incremental cache for the live loop |
| Runner shapes | hard-coded to Tiny (80 mel / 6 heads / 51865 vocab) | driven by `WhisperVariant` |

The context binaries themselves were verified against `TURBO_CONTRACT.md` before any code
change:

```text
encoder.bin  1,755,942,912 bytes  sha256 4F8CE2C1...3A6195   (matches contract)
decoder.bin    452,481,024 bytes  sha256 4043029B...8A893E   (matches contract)
source zip   2,018,863,899 bytes  sha256 CC2358A0...6734CC   (matches contract)
```

## Variant contract

`WhisperVariant` is now the single place that describes a model. Everything else is
shared, because both assets expose the same graph topology: 4 decoder layers, a 200-wide
attention mask, 199 self-KV slots and 1500 cross-KV positions.

| | Tiny | Large-V3-Turbo |
|---|---|---|
| Asset dir | `models/whisper` | `models/whisper_large_v3_turbo` |
| `input_features` | `[1,80,3000]` | `[1,128,3000]` |
| KV heads | 6 | 20 |
| Vocab | 51865 | 51866 |
| Encoder binary | 19,832,832 B | 1,755,942,912 B |
| Decoder binary | 97,452,032 B | 452,481,024 B |

### The special-token trap

Large-V3 inserts `<|yue|>`, which shifts two tokens that the old code hard-coded:

| Token | Tiny | Large-V3-Turbo |
|---|---|---|
| `<|transcribe|>` | 50359 | **50360** |
| `<|notimestamps|>` | 50363 | **50364** |

The runner used to carry `50359`/`50363` as constants. Feeding those to Turbo would have
put the wrong prompt tokens in the decoder, so the prompt and EOS are now resolved from
the loaded tokenizer by name (`WhisperTokenizer.idFor`). `<|endoftext|>`, `<|startoftranscript|>`,
`<|en|>` and `<|zh|>` are stable across both variants and are still asserted.

The *text* vocabulary is shared: scanning the Large-V3 `vocab.json` for Traditional-Chinese
tokens yields the same 216 token IDs as Tiny (`shared=216 only-new=0 only-legacy=0`), so
`TraditionalChineseBlocklist` is reused unchanged instead of duplicated.

## Device evidence

Deterministic run: `sample.wav` (16 kHz mono PCM16, 6.7 s) pushed to the app's external
files dir, then

```powershell
adb shell am start -n com.example.whisperapp/.qnn.M35RealAudioDecoderActivity --es variant turbo
```

```text
M3.5 status: PASS
variant: LARGE_V3_TURBO
features: [1,128,3000] FP32
encoder: HTP SUCCESS
decoder: HTP SUCCESS
forced_prompt: [50258, 50260, 50360, 50364]
eos_reached: true
encoder_ms: 1390.873229
encoder_context_binary_bytes: 1755942912
decoder_context_binary_bytes: 452481024
cpu_fallback: disabled=true
dsp_crash: false
decoded_text: 你好呀 你会说话吗 你应该不会说话吧你说几句来听听呀
```

`encoder_ms` in that report was taken before the HTP performance vote was added; the same fixed WAV
now reports 604.7 / 593.8 / 587.1 ms (see
[Encoder latency](#encoder-latency-the-missing-htp-performance-vote)). Decoded text, token IDs and
`forced_prompt` are unchanged.

Running the same WAV through Tiny with the same request produces a **byte-identical token
sequence and text**, while reporting its own prompt IDs:

```text
variant: TINY
features: [1,80,3000] FP32
forced_prompt: [50258, 50260, 50359, 50363]
token_ids: 26410,15348,10930,12949,8090,21596,14769,10930,44297,44646,47928,8090,21596,6062,42405,6336,254,34592,6912,31022,31022,15348,50257
decoded_text: 你好呀 你会说话吗 你应该不会说话吧你说几句来听听呀
```

That the two independently compiled context binaries agree exactly while consuming
different token IDs is what cross-validates the migration: the 128-bin mel, the 20-head KV
shapes, the mask convention and the by-name prompt resolution are all consistent, and the
Tiny path was not disturbed.

Live microphone loop (STT tab), 1-second updates with the 128-bin incremental cache:

```text
QNN_START #1 variant=LARGE_V3_TURBO samples=16000 cachedMel=true melSize=384000
RESULT #1 text="你好你好"
RESULT #3 text="你好你好你好"
RESULT #4 text="你好你好你好吗"
```

Per-update wall time was ~1.6 s at the time, dominated by the 1.39 s encoder (see the next
section: that encoder number was 2.3x too high); decoder steps cost ~20 ms each. For 30 s of
audio that is an encoder RTF of roughly 0.05, so the model is far from the bottleneck.

## Encoder latency: the missing HTP performance vote

The 1390.9 ms encoder above was not a property of the Turbo asset. The app never set
`htp_performance_mode`, and ONNX Runtime's QNN EP treats that as `"default"`, which makes **no
`QnnHtpPerfInfrastructure` call at all**: no HVX/HMX clock floor and no NoC/DDR bus vote, so the
HTP simply stays in whatever DCVS state the platform picked. Qualcomm AI Hub profiles every
`qnn_context_binary` with burst, which is what its published 466.4 ms encoder number reflects.

Measured on RMX3800 / SM8650 / HTP v75 with the 1.75 GB `encoder.bin` and the fixed 30 s
`[1,128,3000]` window (`encoder_run_ms` is the per-execution duration with the QNN accelerator
busy time confirmed independently through `profiling_level=optrace`):

| `htp_performance_mode` | Encoder execution | Notes |
|---|---:|---|
| unset (ORT default) | 1343.8, 1304.6, 1365.0 ms | three consecutive executions in one session — not a first-run artifact |
| `sustained_high_performance` | 604.7, 593.8, 587.1 ms | shipped default (thermally sustainable) |
| `burst` | 630.0 ms | AI Hub's profiling profile |
| `high_performance` | 644.6 ms | |
| `rpc_control_latency=100` | 829.9 ms | slower; leave at ORT's default |
| `qnn_context_priority=high` | 588.0 ms | no effect |
| `htp_graph_finalization_optimization_mode=3` | 1386.9 ms | no effect |
| `vtcm_mb=8` | 610.4 ms | no effect |

The QNN optrace profile explains why: with the default profile the encoder is limited by
accelerator busy time (1323.8 ms of `Accelerator (execute) time`, 3.89e9 cycles — 41.6 % conv2d,
32.9 % softmax, 13.2 % matmul), while under a performance vote the same graph finishes in ~590 ms.
The decoder never showed the gap (HTP execute ~9-11 ms vs AI Hub's 7.96 ms), because it is a short
matmul-dominated graph rather than a long clock-bound conv/softmax one — so this was never a
generic "the app is slow" problem, and it was never mel either: the live loop feeds precomputed
incremental mel into `transcribeChunk`, and `encoder_ms` is measured around `session.run` only.

Why the other knobs do nothing here: `vtcm_mb`, `htp_graph_finalization_optimization_mode` and
`enable_htp_fp16_precision` are turned into QNN *graph* configs by ORT only on its JIT compile
path; with a precompiled context binary ORT loads the `.bin` directly and never applies them.
Those would have to be re-linked offline through AI Hub (`default_graph_htp_vtcm_size=...`,
`default_graph_htp_optimizations=O=3`).

`QnnWhisperRealAudioRunner` now votes `htp_performance_mode=sustained_high_performance` on both
sessions and also passes the run-level `qnn.perf_mode` (ORT gates its DSPQ polling on it), and it
exposes the diagnostic knobs used for the sweep above through
`configureHtp(...)` / `M35RealAudioDecoderActivity` extras.

## Latency

| Stage | Tiny | Turbo (before) | Turbo (now) |
|---|---:|---:|---:|
| Encoder (30 s window) | 239.8 ms | 1390.9 ms | ~590 ms |
| Decoder step | ~20 ms | ~20 ms | ~20 ms |

AI Hub's own figure for this asset and device is 466.4 ms encoder / 7.96 ms decoder; the residual
gap is methodology (their number is a minimum over many hosted, non-throttled iterations in burst)
rather than a missing option.

## Files

Added:

- `app/src/main/java/com/example/whisperapp/asr/WhisperVariant.kt`
- `app/src/main/java/com/example/whisperapp/audio/WhisperMelBackend.kt`
- `app/src/test/java/com/example/whisperapp/TurboQnnContractTest.kt`
- `tools/build_turbo_epcontext.py`, `tools/verify_turbo_epcontext.py`, `tools/gen_blocklist.py`
- `tools/fetch_turbo_assets.py`
- `app/src/main/assets/models/whisper_large_v3_turbo/{encoder_ctx.onnx,decoder_ctx.onnx,tokenizer/}`

Changed:

- `QnnWhisperRealAudioRunner.kt`: variant-parameterised; per-variant cache directory;
  `openFd`-based asset length check; votes `htp_performance_mode` plus the run-level
  `qnn.perf_mode` on every execution and exposes `configureHtp(...)` for the diagnostic sweep
- `WhisperTokenizer.kt`: `expectedVocabSize`, `idFor`, `vocabularySize`; the
  transcribe/notimestamps assertions are no longer hard-coded
- `WhisperFeatureExtractor.kt` / `WhisperTurboFeatureExtractor.kt`: implement the
  frontend/cache interfaces; Turbo gains a 128-bin incremental cache
- `MainScreen.kt`: uses `WhisperVariant.LARGE_V3_TURBO`
- `app/build.gradle.kts`: `unitTests.isReturnDefaultValues` (the incremental-cache tests
  previously failed with `Method i in android.util.Log not mocked`), and a comment
  explaining why assets stay uncompressed
- `M35RealAudioDecoderActivity.kt`: accepts `--es variant tiny|turbo`, plus the diagnostic
  `perf`, `runopt`, `rpc`, `prio`, `vtcm`, `fin`, `extra_key`/`extra_value` and `repeats` extras

## Tests

`gradle :app:testDebugUnitTest` — 37 tests, 0 failures. New coverage:

- the Large-V3 tokenizer loads at 51866 entries and reports the shifted IDs
- `WhisperVariant` shapes match `metadata.json`
- the 128-bin incremental mel cache reproduces the full extraction byte-for-byte

`tools/verify_turbo_epcontext.py` asserts both generated wrappers match `metadata.json`
in tensor order, dtype and shape.

## Rebuilding the payload

The weights are **not** committed: `encoder.bin` is 1.75 GB, far over GitHub's 100 MB
per-file limit, and the Turbo set is a build artifact rather than source. A fresh clone has
the runner code but not the payload, so reconstruct it with:

```powershell
python tools/fetch_turbo_assets.py          # downloads + verifies the AI Hub zip, extracts
                                            # the binaries, fetches the Large-V3 tokenizer
python tools/build_turbo_epcontext.py       # generates encoder_ctx.onnx / decoder_ctx.onnx
python tools/verify_turbo_epcontext.py      # checks the wrappers against metadata.json
```

`fetch_turbo_assets.py --verify-only` re-checks an existing tree without downloading. It
validates the archive and both binaries against the SHA-256 values in `TURBO_CONTRACT.md`,
and asserts the tokenizer merges to 51866 entries with the Large-V3 token IDs (`<|yue|>`
50358, `<|transcribe|>` 50360, `<|notimestamps|>` 50364).

The resulting layout the app expects:

```text
app/src/main/assets/models/whisper_large_v3_turbo/
├── encoder.bin  decoder.bin  metadata.json      Qualcomm AI Hub asset (QAIRT 2.45.0)
├── encoder_ctx.onnx  decoder_ctx.onnx           generated EPContext wrappers
└── tokenizer/{vocab.json,merges.txt,added_tokens.json}
```

## Known limitations

- The APK is ~3.06 GB because both variants' context binaries ship as uncompressed
  assets, and the app copies the active variant's binaries (2.2 GB for Turbo) into
  `cacheDir` on first use. `cacheDir` is system-evictable; `filesDir` would be safer.
- Session creation for the 1.75 GB encoder binary is slow, and the first STT tab switch
  pays the asset copy, so the first start takes noticeably longer than later ones.
- The live loop stops reading `AudioRecord` while a chunk is transcribing, so audio is
  dropped during inference (pre-existing design).
- Only the `zh` forced-language prompt is wired up.
