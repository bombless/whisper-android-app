"""Download the tested SDXL assets and deploy them to the debug Android app."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from build_sdxl_epcontext import build_wrapper

import json
import onnx


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "downloads"
MODEL_DIR = CACHE / "model"
DEVICE_DIR = "/sdcard/Android/data/io.github.bombless.htp_playpen/files/sdxl"
MODEL_REVISION = "ead90f4635e21e7412b8200a5efd220b0193beeb"
CLIP_REVISION = "16aee6ccb10aee640bb37c7072763ad4d090e4ca"
ARCHIVE_SHA = "426e36987fd3b84dd05255cb12bc5463c427c8b55598bd3b2486a72291d6be7f"
ASSETS = {
    "text_encoder/model.onnx": "21267b285b3e3ab33fc6d95610ce629a67e0bca01647c36e65e6cd04b6eaa6e7",
    "text_encoder/model.onnx_data": "4b61db0f4781c335e6bba50ab7471bbfe02aad41d050ff5935dcbb2d7724fda0",
    "text_encoder_2/model.onnx": "0cf6265108359805f27b16141209c83b9452d4ac66f7b626c6c6f046f648f484",
    "text_encoder_2/model.onnx_data": "81330f8daecb663019d641658449d3d76205d90c2207356b9cccf1a427214c12",
}
EXTRACTED = {
    "unet.bin": "1fa6f33f6e3aaa4797250b76b2c7c8217cda8c64c8768f2307748aba5598c50b",
    "tokenizer.json": "5f120510b5c01ccdb9a7d29d2beca07cac667f76472cab123ffb1c2f0b27cdeb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo: str, revision: str, filename: str, target: Path, expected: str, proxy: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == expected:
        print(f"Verified {target}", flush=True)
        return
    urls = [(f"https://hf-mirror.com/{repo}/resolve/{revision}/{filename}", None)]
    if proxy:
        urls.append((f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", proxy))
    for url, proxy_url in urls:
        command = ["curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "30",
                   "--speed-limit", "1024", "--speed-time", "30", "-C", "-", "-o", str(target)]
        if proxy_url:
            command += ["--proxy", proxy_url]
        command.append(url)
        print(f"Downloading {filename} from {url.split('/')[2]}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode == 0 and target.is_file() and sha256(target) == expected:
            return
    raise RuntimeError(f"Download failed or SHA-256 mismatch: {target}; remove the incomplete file and retry")


def prepare(proxy: str | None) -> None:
    archive = CACHE / "sdxl_base_qnn2.28_8gen3.zip"
    download("xororz/sdxl-qnn", MODEL_REVISION, archive.name, archive, ARCHIVE_SHA, proxy)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for filename, expected in EXTRACTED.items():
            target = MODEL_DIR / filename
            if target.is_file() and sha256(target) == expected:
                continue
            temporary = target.with_suffix(target.suffix + ".partial")
            with bundle.open(f"output/qnn_models_sdxl_8gen3/{filename}") as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            if sha256(temporary) != expected:
                raise RuntimeError(f"Extracted asset checksum mismatch: {temporary}")
            temporary.replace(target)
    for name, expected in ASSETS.items():
        download("Cyronius/sdxl-lightning-4step-onnx-web-int8", CLIP_REVISION, name,
                 MODEL_DIR / name, expected, proxy)
    vae = ROOT / "taesdxl_decoder.onnx"
    if sha256(vae) != "a2911f6ecd3fb71bd5bc8656dce9d1bff402a20b81f59cf2b39450b810c2ca98":
        raise RuntimeError(f"Unexpected VAE: {vae}")
    with (ROOT / "unet-metadata.json").open(encoding="utf-8") as handle:
        wrapper = build_wrapper(json.load(handle), "unet.bin")
    onnx.save(wrapper, MODEL_DIR / "unet_ctx.onnx")


def deploy() -> None:
    pkg = "io.github.bombless.htp_playpen"
    subprocess.run(["adb", "shell", "am", "force-stop", pkg], check=True)
    subprocess.run(["adb", "shell", "run-as", pkg, "rm", "-f",
                    "files/sdxl-prepare-result.txt"], check=True)
    subprocess.run(["adb", "shell", "am", "start", "-n", f"{pkg}/com.example.whisperapp.qnn.QnnSmokeActivity",
                    "--ez", "sdxl_prepare", "true"], check=True)
    for _ in range(30):
        time.sleep(1)
        report = subprocess.run(["adb", "shell", "run-as", pkg, "cat",
                                 "files/sdxl-prepare-result.txt"], capture_output=True, text=True)
        if report.returncode == 0 and report.stdout.startswith("READY:"):
            break
        if report.returncode == 0 and report.stdout.startswith("FAIL:"):
            raise RuntimeError(report.stdout.strip())
    else:
        raise RuntimeError("The app could not prepare its SDXL directory; install the debug APK first")
    files = {name: MODEL_DIR / name for name in (*EXTRACTED, *ASSETS, "unet_ctx.onnx")}
    files["vae.onnx"] = ROOT / "taesdxl_decoder.onnx"
    for name, source in files.items():
        expected = sha256(source)
        remote = f"{DEVICE_DIR}/{name}"
        current = subprocess.run(["adb", "shell", "sha256sum", remote], capture_output=True, text=True)
        if current.returncode == 0 and current.stdout.split()[0:1] == [expected]:
            print(f"Already verified on device: {name}", flush=True)
            continue
        print(f"Deploying {name} ({source.stat().st_size} bytes)", flush=True)
        subprocess.run(["adb", "push", str(source), remote], check=True)
        actual = subprocess.run(["adb", "shell", "sha256sum", remote], capture_output=True, text=True, check=True)
        if actual.stdout.split()[0:1] != [expected]:
            raise RuntimeError(f"Device checksum mismatch: {remote}")
    print(f"Ready: {DEVICE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Proxy for fallback to huggingface.co, e.g. http://127.0.0.1:7897")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    prepare(args.proxy)
    if not args.download_only:
        deploy()


if __name__ == "__main__":
    main()
