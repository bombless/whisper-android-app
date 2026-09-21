import os, time, urllib.request, shutil
URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/whisper_large_v3_turbo/releases/v0.62.2/whisper_large_v3_turbo-qnn_context_binary-float-qualcomm_snapdragon_8gen3.zip"
OUT = r"D:\mcp-agent-workspace\whisper-app\tools\whisper_large_v3_turbo_asset"
TOTAL = 2018863899
CHUNK = 67108864
MISSING = [0, 13, 18]

for i in MISSING:
    start = i * CHUNK
    end = min(TOTAL - 1, (i + 1) * CHUNK - 1)
    want = end - start + 1
    target = os.path.join(OUT, f"part_{i:02d}")
    tmp = target + ".repairing"
    for attempt in range(1, 9):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "whisper-project-bot/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                cr = resp.headers.get("Content-Range", "")
                cl = int(resp.headers.get("Content-Length", "-1"))
                print(f"part={i:02d} attempt={attempt} status={getattr(resp,'status',None)} content-range={cr} content-length={cl}", flush=True)
                if cl != want:
                    raise RuntimeError(f"unexpected content-length {cl}, want {want}")
                if not cr.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"unexpected content-range {cr}")
                with open(tmp, "wb") as f:
                    n = 0
                    while n < want:
                        b = resp.read(min(1024*1024, want-n))
                        if not b:
                            raise RuntimeError(f"early EOF at {n}/{want}")
                        f.write(b)
                        n += len(b)
            if os.path.getsize(tmp) != want:
                raise RuntimeError("temp size mismatch")
            os.replace(tmp, target)
            print(f"part={i:02d} DONE size={want}", flush=True)
            break
        except Exception as e:
            print(f"part={i:02d} attempt={attempt} ERROR {e}", flush=True)
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            if attempt == 8:
                raise
            time.sleep(min(10, attempt*2))
print("REPAIR_COMPLETE", flush=True)
