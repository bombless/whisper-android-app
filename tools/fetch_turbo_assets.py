#!/usr/bin/env python3
"""Reconstruct the Whisper-Large-V3-Turbo QNN payload that is deliberately not in git.

The context binaries are far too large to commit (encoder.bin alone is 1.75 GB, well over
GitHub's 100 MB per-file limit), so a fresh clone has the runner code but not the weights.
This script fetches them into exactly the layout the app expects:

    app/src/main/assets/models/whisper_large_v3_turbo/
        encoder.bin  decoder.bin  metadata.json               <- Qualcomm AI Hub asset
        tokenizer/{vocab.json,merges.txt,added_tokens.json}   <- openai/whisper-large-v3-turbo

Afterwards generate and check the EPContext wrappers that ONNX Runtime actually loads:

    python tools/build_turbo_epcontext.py
    python tools/verify_turbo_epcontext.py

Usage:
    python tools/fetch_turbo_assets.py                 # download + extract + verify
    python tools/fetch_turbo_assets.py --verify-only   # check what is already present
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile

ASSET_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/"
    "whisper_large_v3_turbo/releases/v0.62.2/"
    "whisper_large_v3_turbo-qnn_context_binary-float-qualcomm_snapdragon_8gen3.zip"
)
ZIP_NAME = "whisper_large_v3_turbo.zip"
ZIP_SIZE = 2018863899
ZIP_SHA256 = "CC2358A0BC1365CDD1CAE047E599679B25CA6C1C0AA9592127C88440667734CC"

# From TURBO_CONTRACT.md, verified from the Qualcomm asset's metadata.json.
BINARIES = {
    "encoder.bin": (1755942912, "4F8CE2C1BED4360260F6B6C089B5ED3BE24D39559D1B6797FB1CF3A58A3A6195"),
    "decoder.bin": (452481024, "4043029B973A24BAC07317033BF18F6608ABFCA1B7F4A02B2BF78087A78A893E"),
}

TOKENIZER_BASE = "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/"
TOKENIZER_FILES = ("vocab.json", "merges.txt", "added_tokens.json")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(
    REPO_ROOT, "app", "src", "main", "assets", "models", "whisper_large_v3_turbo"
)
DEFAULT_CACHE = os.path.join(REPO_ROOT, "tools", "whisper_large_v3_turbo_asset")


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def matches(path: str, size: int, sha256: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) == size and sha256_of(path) == sha256


def download(url: str, dest: str, expected_size: int | None = None, expected_sha256: str | None = None) -> None:
    """Download with HTTP Range resume; the asset is ~2 GB so restarts must be cheap."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if expected_size is not None and have > expected_size:
        os.remove(part)
        have = 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, open(part, mode) as out:
        total = expected_size
        if total is None:
            length = response.headers.get("Content-Length")
            total = (have + int(length)) if length else None
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            out.write(block)
            have += len(block)
            if total:
                print(f"\r  {have / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    os.replace(part, dest)
    if expected_sha256 and sha256_of(dest) != expected_sha256:
        raise SystemExit(f"checksum mismatch after download: {dest}")


def ensure_zip(cache_dir: str, verify_only: bool) -> str:
    zip_path = os.path.join(cache_dir, ZIP_NAME)
    if matches(zip_path, ZIP_SIZE, ZIP_SHA256):
        print(f"OK   source zip present and verified: {zip_path}")
        return zip_path
    if verify_only:
        raise SystemExit(f"missing or incomplete source zip: {zip_path}")
    print(f"downloading Qualcomm AI Hub asset -> {zip_path}")
    download(ASSET_URL, zip_path, ZIP_SIZE, ZIP_SHA256)
    print(f"OK   source zip verified: {zip_path}")
    return zip_path


def extract_assets(zip_path: str, dest: str, verify_only: bool) -> None:
    """Pull encoder.bin / decoder.bin / metadata.json out of the AI Hub zip."""
    need_bins = {n: s for n, s in BINARIES.items() if not matches(os.path.join(dest, n), *s)}
    need_meta = not os.path.isfile(os.path.join(dest, "metadata.json"))
    if not need_bins and not need_meta:
        for name in BINARIES:
            print(f"OK   {name} present and verified")
        return
    if verify_only:
        raise SystemExit(
            f"missing or invalid payload: binaries={sorted(need_bins)} metadata_missing={need_meta}"
        )
    os.makedirs(dest, exist_ok=True)
    staging = os.path.join(dest, "_staging")
    wanted = set(need_bins) | ({"metadata.json"} if need_meta else set())
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            entry for entry in archive.namelist()
            if not entry.endswith("/") and os.path.basename(entry) in wanted
        ]
        missing = wanted - {os.path.basename(m) for m in members}
        if missing:
            raise SystemExit(f"source zip does not contain {sorted(missing)}")
        archive.extractall(staging, members=members)
    for member in members:
        base = os.path.basename(member)
        print(f"extracting {base}")
        os.replace(os.path.join(staging, member), os.path.join(dest, base))
    shutil.rmtree(staging, ignore_errors=True)
    for name, spec in BINARIES.items():
        if not matches(os.path.join(dest, name), *spec):
            raise SystemExit(f"{name} does not match the recorded checksum")
        print(f"OK   {name} extracted and verified")


def ensure_metadata(dest: str, verify_only: bool) -> None:
    path = os.path.join(dest, "metadata.json")
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("model_id") != "whisper_large_v3_turbo":
        raise SystemExit(f"unexpected model_id in {path}: {metadata.get('model_id')}")
    print(f"OK   metadata.json model_id={metadata['model_id']} qairt={metadata['tool_versions']['qairt']}")


def ensure_tokenizer(dest: str, verify_only: bool) -> None:
    tokenizer_dir = os.path.join(dest, "tokenizer")
    missing = [n for n in TOKENIZER_FILES if not os.path.isfile(os.path.join(tokenizer_dir, n))]
    if missing:
        if verify_only:
            raise SystemExit(f"missing tokenizer files: {missing}")
        os.makedirs(tokenizer_dir, exist_ok=True)
        for name in missing:
            print(f"downloading {name} from openai/whisper-large-v3-turbo")
            download(TOKENIZER_BASE + name, os.path.join(tokenizer_dir, name))
    vocab = json.load(open(os.path.join(tokenizer_dir, "vocab.json"), encoding="utf-8"))
    added = json.load(open(os.path.join(tokenizer_dir, "added_tokens.json"), encoding="utf-8"))
    merged = dict(vocab)
    merged.update(added)
    if len(merged) != 51866:
        raise SystemExit(f"tokenizer must merge to 51866 entries, got {len(merged)}")
    # Large-V3 inserts <|yue|>, shifting these by one relative to Tiny.
    if merged.get("<|yue|>") != 50358 or merged.get("<|transcribe|>") != 50360 or merged.get("<|notimestamps|>") != 50364:
        raise SystemExit("tokenizer does not match the expected Large-V3 special-token table")
    print(f"OK   tokenizer merged vocabulary={len(merged)} yue=50358 transcribe=50360 notimestamps=50364")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--verify-only", action="store_true",
                        help="fail instead of downloading; checks an existing tree")
    args = parser.parse_args()

    print(f"target: {args.dest}")
    zip_path = ensure_zip(args.cache_dir, args.verify_only)
    extract_assets(zip_path, args.dest, args.verify_only)
    ensure_metadata(args.dest, args.verify_only)
    ensure_tokenizer(args.dest, args.verify_only)
    print()
    print("next:")
    print("  python tools/build_turbo_epcontext.py")
    print("  python tools/verify_turbo_epcontext.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
