# Whisper-Large-V3-Turbo Contract

Status: verified from the Qualcomm `qnn_context_binary` asset's `metadata.json`.

## Asset

- Model: `Whisper-Large-V3-Turbo`
- Runtime: `qnn_context_binary`
- Precision family: `float` (the metadata declares FP16 tensors)
- Target: Snapdragon 8 Gen 3 / SM8650
- QAIRT: `2.45.0.260326154327`
- HTP version: `75`
- SoC model: `57`
- Reference device: `Samsung Galaxy S24 (Family)`
- Android asset directory: `app/src/main/assets/models/whisper_large_v3_turbo/`
- Files: `encoder.bin`, `decoder.bin`, `metadata.json`

## Encoder contract

The metadata declares one input:

| Name | dtype | shape |
|---|---|---|
| `input_features` | `float16` | `[1, 128, 3000]` |

The encoder emits four cross-attention K/V pairs. For each layer index `0..3`:

| Name pattern | dtype | shape |
|---|---|---|
| `k_cache_cross_N` | `float16` | `[20, 1, 64, 1500]` |
| `v_cache_cross_N` | `float16` | `[20, 1, 1500, 64]` |

## Decoder contract

Inputs include:

| Name | dtype | shape |
|---|---|---|
| `input_ids` | `int32` | `[1, 1]` |
| `position_ids` | `int32` | `[1]` |
| `attention_mask` | `float16` | `[1, 1, 1, 200]` |

For each layer index `0..3`, the decoder consumes:

| Name pattern | dtype | shape |
|---|---|---|
| `k_cache_self_N_in` | `float16` | `[20, 1, 64, 199]` |
| `v_cache_self_N_in` | `float16` | `[20, 1, 199, 64]` |
| `k_cache_cross_N` | `float16` | `[20, 1, 64, 1500]` |
| `v_cache_cross_N` | `float16` | `[20, 1, 1500, 64]` |

For each layer index `0..3`, the decoder emits:

| Name pattern | dtype | shape |
|---|---|---|
| `k_cache_self_N_out` | `float16` | `[20, 1, 64, 199]` |
| `v_cache_self_N_out` | `float16` | `[20, 1, 199, 64]` |

It also emits:

| Name | dtype | shape |
|---|---|---|
| `logits` | `float16` | `[1, 51866, 1, 1]` |

## Important migration consequences

1. This Qualcomm asset is **QNN Context Binary**, not an ONNX model. The contract inspector therefore reads `metadata.json` rather than trying to open `.bin` with ONNX Runtime.
2. The mel feature contract is **128 bins**, not the existing Tiny 80-bin contract.
3. The decoder vocabulary/logits dimension is **51866**, not the Tiny runner's 51865 assumption.
4. The decoder has four layers and uses the metadata-declared cache layouts. These values are now documented from the actual asset, rather than copied from the Tiny runner.
5. The maximum decoded sequence length exposed by the decoder attention mask is **200**.

## Binary integrity

- `encoder.bin`: 1,755,942,912 bytes; SHA-256 `4F8CE2C1BED4360260F6B6C089B5ED3BE24D39559D1B6797FB1CF3A58A3A6195`
- `decoder.bin`: 452,481,024 bytes; SHA-256 `4043029B973A24BAC07317033BF18F6608ABFCA1B7F4A02B2BF78087A78A893E`
- Source ZIP: 2,018,863,899 bytes; SHA-256 `CC2358A0BC1365CDD1CAE047E599679B25CA6C1C0AA9592127C88440667734CC`

## Execution path

The raw context binaries are consumed through generated `EPContext` ONNX wrappers
(`encoder_ctx.onnx` / `decoder_ctx.onnx`, `embed_mode=0`, `ep_cache_context` pointing at the
sibling `.bin`). Regenerate and re-verify them with:

```powershell
python tools/build_turbo_epcontext.py
python tools/verify_turbo_epcontext.py
```

`WhisperVariant` selects Tiny or Turbo for one shared execution path, because the graph
topology is identical and only the mel bins (80/128), KV heads (6/20) and vocabulary
(51865/51866) differ. See `TURBO_QNN_RESULTS.md` for the device results and for the
Large-V3 special-token shift (`<|transcribe|>` 50360, `<|notimestamps|>` 50364) that the
runner resolves by name instead of hard-coding.