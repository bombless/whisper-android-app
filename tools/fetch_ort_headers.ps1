# Extracts the ONNX Runtime C API headers and the link stub for the native decoder (P1)
# from the onnxruntime-android AAR that Gradle already downloaded.
#
# The AAR ships `headers/` (onnxruntime_cxx_api.h & friends) and `jni/<abi>/libonnxruntime.so`.
# The native decoder needs both to compile and link; at run time the real library comes from
# the APK, where the same AAR already packages it. Only the headers are committed — the
# 28 MB .so is a build input and is gitignored.
#
# Usage:  pwsh -File tools/fetch_ort_headers.ps1 [-Version 1.27.0] [-Abi arm64-v8a]
# Keep -Version in step with the `onnxruntime-android` dependency in app/build.gradle.kts.

param(
    [string]$Version = "1.27.0",
    [string]$Abi = "arm64-v8a"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $repoRoot "app/src/main/cpp/third_party/onnxruntime"
$gradleCache = Join-Path $env:USERPROFILE ".gradle/caches/modules-2/files-2.1/com.microsoft.onnxruntime/onnxruntime-android/$Version"

$aar = Get-ChildItem -Path $gradleCache -Recurse -Filter "*.aar" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $aar) {
    throw "onnxruntime-android $Version AAR not found under $gradleCache. Run a Gradle build once so it is downloaded."
}

Write-Host "Using AAR: $($aar.FullName)"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ort_extract_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Copy-Item $aar.FullName (Join-Path $tmp "ort.zip")
    Expand-Archive -Path (Join-Path $tmp "ort.zip") -DestinationPath $tmp -Force

    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Copy-Item (Join-Path $tmp "headers/*") $dest -Force
    Write-Host "Headers -> $dest"

    $libDest = Join-Path $dest "lib/$Abi"
    New-Item -ItemType Directory -Path $libDest -Force | Out-Null
    $so = Join-Path $tmp "jni/$Abi/libonnxruntime.so"
    if (-not (Test-Path $so)) { throw "libonnxruntime.so for $Abi not present in the AAR" }
    Copy-Item $so (Join-Path $libDest "libonnxruntime.so") -Force
    Write-Host "Link stub -> $libDest/libonnxruntime.so"
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "done."
