package com.example.whisperapp.qnn

import android.util.Log
import java.io.*
import java.net.*
import java.nio.charset.StandardCharsets
import java.util.concurrent.Executors

/**
 * On-device HTTP server exposing the full optrace:
 *  - live app-level span aggregates + per-run breakdown ([OpTrace])
 *  - QNN HTP optrace CSVs parsed into per-op aggregates
 *  - raw file download (QNN CSV / logs / diagnostics)
 *
 * Access from the host with: adb forward tcp:8765 tcp:8765  ->  http://localhost:8765
 */
class QnnProfileServer(private val root: File, private val port: Int = 8765) {
    private val tag = "QNN_PROFILE_HTTP"
    private val pool = Executors.newCachedThreadPool()
    @Volatile private var running = false
    private var server: ServerSocket? = null
    private val bootEpoch = System.nanoTime()

    @Synchronized
    fun start() {
        if (running) return
        root.mkdirs()
        try {
            server = ServerSocket(port, 16, InetAddress.getByName("127.0.0.1"))
        } catch (e: BindException) {
            // The profile dashboard is diagnostic-only. A stale process, another app instance,
            // or an existing adb-forwarded server may already own 8765. Never let that prevent
            // the Whisper QNN session from starting; simply disable this optional endpoint.
            server = null
            running = false
            Log.w(tag, "DISABLED port=$port already in use; STT startup will continue", e)
            return
        }
        running = true
        pool.execute { acceptLoop() }
        Log.i(tag, "LISTEN 127.0.0.1:$port")
    }

    @Synchronized
    fun stop() {
        running = false
        try { server?.close() } catch (_: Throwable) {}
        server = null
    }

    private fun acceptLoop() {
        while (running) {
            try {
                val s = server?.accept() ?: break
                pool.execute { runCatching { handle(s) }.onFailure { Log.w(tag, "HANDLE_FAILED ${it.javaClass.name}: ${it.message}", it) } }
            } catch (_: Throwable) {}
        }
    }

    private fun handle(s: Socket) {
        s.use {
            val r = BufferedReader(InputStreamReader(it.getInputStream(), StandardCharsets.UTF_8))
            val w = BufferedWriter(OutputStreamWriter(it.getOutputStream(), StandardCharsets.UTF_8))
            val req = r.readLine() ?: return
            val p = req.split(" ")
            var contentLength = 0
            while (true) {
                val x = r.readLine() ?: break
                if (x.isEmpty()) break
                val lc = x.lowercase()
                if (lc.startsWith("content-length:")) contentLength = lc.substringAfter(':').trim().toIntOrNull() ?: 0
            }
            if (contentLength > 0) { val body = CharArray(contentLength); var off = 0; while (off < contentLength) { val n = r.read(body, off, contentLength - off); if (n <= 0) break; off += n } }
            if (p.size < 2) { respond(w, 400, "application/json", "{\"error\":\"bad request\"}"); return }
            val method = p[0]
            val rawTarget = p[1]
            val path = URLDecoder.decode(rawTarget.substringBefore("?"), "UTF-8")
            val query = URLDecoder.decode(rawTarget.substringAfter("?", ""), "UTF-8")
            // Every branch is wrapped: an exception used to escape handle() and kill the
            // socket mid-response, which the client sees as a bare disconnect with no clue
            // what failed (the CSV endpoints parse files on the request thread).
            try {
                when {
                    path == "/" && method == "GET" -> respond(w, 200, "text/html; charset=utf-8", DASHBOARD_HTML)
                    path == "/api/status.json" && method == "GET" -> respond(w, 200, "application/json", statusJson())
                    path == "/api/profile.json" && method == "GET" -> respond(w, 200, "application/json", profileJson())
                    path == "/api/trace.json" && method == "GET" -> respond(w, 200, "application/json", traceJson())
                    path == "/api/events.json" && method == "GET" -> respond(w, 200, "application/json", eventsJson())
                    path == "/api/config" -> respond(w, 200, "application/json", configJson(query))
                    path == "/api/trace/clear" && (method == "POST" || method == "GET") -> {
                        OpTrace.clear()
                        respond(w, 200, "application/json", "{\"cleared\":true}")
                    }
                    path.startsWith("/download/") && method == "GET" -> download(w, path.removePrefix("/download/"))
                    else -> respond(w, 404, "application/json", "{\"error\":\"not found\"}")
                }
            } catch (t: Throwable) {
                Log.w(tag, "REQUEST_FAILED path=$path: ${t.javaClass.name}: ${t.message}", t)
                runCatching {
                    respond(w, 500, "application/json", "{\"error\":${OpTrace.jsonEscape(t.javaClass.name + ": " + t.message).let { "\"$it\"" }}}")
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // JSON endpoints
    // ------------------------------------------------------------------

    private fun statusJson(): String {
        val fs = root.listFiles()?.sortedBy { it.name }.orEmpty()
        return "{\"running\":$running,\"port\":$port,\"trace_enabled\":${OpTrace.snapshot().enabled}," +
            "\"uptime_ms\":${OpTrace.fmt((System.nanoTime() - bootEpoch) / 1_000_000.0)}," +
            "\"files\":[${fs.joinToString(",") { fileJson(it) }}]}"
    }

    private fun profileJson(): String {
        val fs = root.listFiles()?.sortedBy { it.name }.orEmpty()
        val csv = fs.filter { it.extension.equals("csv", true) }
        val logs = fs.filter { it.name.endsWith("_qnn.log", true) }
        val ev = csv.flatMap { parseCsvRows(it) }
        return "{\"schema\":\"whisper-qnn-profile-v1\",\"profiling_level\":\"optrace\",\"qnn_ep\":\"QNNExecutionProvider\"," +
            "\"files\":[${fs.joinToString(",") { fileJson(it) }}]," +
            "\"csv_events\":[${ev.joinToString(",")}]," +
            "\"qnn_logs\":[${logs.joinToString(",") { OpTrace.jsonEscape(it.name).let { n -> "\"$n\"" } }}]}"
    }

    private fun traceJson(): String {
        val snap = OpTrace.snapshot()
        val fs = root.listFiles()?.sortedBy { it.name }.orEmpty()
        val bootWallMs = System.currentTimeMillis() - (System.nanoTime() - bootEpoch) / 1_000_000L
        val csvFiles = fs.filter { it.extension.equals("csv", true) && it.lastModified() >= bootWallMs }
        // aggregate the newest encoder/decoder csvs (one inference each after stop())
        val newest = csvFiles.takeLast(2)
        val ops = newest.flatMap { qnnOps(it) }
        val recentRuns = snap.runs.takeLast(16).reversed()
        return buildString {
            append("{\"schema\":\"whisper-optrace-v2\",\"enabled\":${snap.enabled},")
            append("\"uptime_ms\":${OpTrace.fmt((System.nanoTime() - bootEpoch) / 1_000_000.0)},")
            append("\"stats\":[${snap.eventStats.joinToString(",") { OpTrace.statsJson(it) }}],")
            append("\"run_stats\":[${snap.runStats.joinToString(",") { OpTrace.statsJson(it) }}],")
            append("\"runs\":[${recentRuns.joinToString(",") { OpTrace.runJson(it) }}],")
            append("\"qnn_ops\":[${ops.joinToString(",") { OpTrace.opJson(it) }}],")
            append("\"files\":[${fs.joinToString(",") { fileJson(it) }}]}")
        }
    }

    /**
     * Optrace CSVs are tens of MB and parsing one takes seconds, while the dashboard polls
     * every 2 s. Parsing on the request thread used to stall the poll into the minutes when
     * a fresh multi-MB CSV appeared. Results are cached by (path, size, mtime), so a CSV is
     * parsed once and then served from memory until the file actually changes.
     */
    private val qnnOpsCache = java.util.concurrent.ConcurrentHashMap<String, Pair<Long, List<OpTrace.OpAggregate>>>()

    private fun qnnOps(f: File): List<OpTrace.OpAggregate> {
        val stamp = f.length() xor (f.lastModified() shl 8)
        qnnOpsCache[f.absolutePath]?.let { (cachedStamp, ops) -> if (cachedStamp == stamp) return ops }
        val ops = OpTrace.parseQnnOptraceCsv(f)
        qnnOpsCache[f.absolutePath] = stamp to ops
        return ops
    }

    private fun eventsJson(): String {
        val snap = OpTrace.snapshot()
        val events = snap.events.takeLast(512)
        return "{\"events\":[${events.joinToString(",") { OpTrace.eventJson(it) }}]}"
    }

    /** GET/POST /api/config?debug=0|1&profiling=0|1 — toggles runner behaviour remotely. */
    private fun configJson(query: String): String {
        fun flag(name: String): Boolean? {
            val m = Regex("(?:^|&)" + name + "=(0|1|true|false)(?:&|$)").find(query) ?: return null
            return m.groupValues[1] == "1" || m.groupValues[1] == "true"
        }
        val debug = flag("debug")
        val profiling = flag("profiling")
        QnnWhisperRealAudioRunner.configure(profiling = profiling, debug = debug)
        return "{\"debug\":${QnnWhisperRealAudioRunner.debugDiagnostics}," +
            "\"profiling\":${QnnWhisperRealAudioRunner.profilingEnabled}," +
            "\"note\":\"sessions rebuilt on next start()\"}"
    }

    private fun parseCsvRows(f: File): List<String> {
        if (!f.isFile || f.length() > 33554432) return emptyList()
        val ls = f.readLines(Charsets.UTF_8)
        if (ls.isEmpty()) return emptyList()
        val h = splitCsv(ls.first())
        return ls.drop(1).filter { it.isNotBlank() }.map { line ->
            val v = splitCsv(line)
            "{" + h.indices.joinToString(",") { i -> OpTrace.jsonEscape(h[i]) + ":" + OpTrace.jsonEscape(v.getOrNull(i).orEmpty()).let { "\"$it\"" } } + ",\"source_file\":" + "\"${OpTrace.jsonEscape(f.name)}\"" + "}"
        }
    }

    private fun splitCsv(s: String): List<String> {
        val o = mutableListOf<String>()
        val c = StringBuilder()
        var qd = false
        var i = 0
        while (i < s.length) {
            val x = s[i]
            if (x == '"') {
                if (qd && i + 1 < s.length && s[i + 1] == '"') { c.append('"'); i++ } else qd = !qd
            } else if (x == ',' && !qd) { o += c.toString(); c.setLength(0) } else c.append(x)
            i++
        }
        o += c.toString()
        return o
    }

    private fun download(w: BufferedWriter, n: String) {
        val f = File(root, n)
        if (!safe(f) || !f.isFile) { respond(w, 404, "application/json", "{\"error\":\"file not found\"}"); return }
        respond(w, 200, when {
            n.endsWith(".json", true) -> "application/json"
            n.endsWith(".csv", true) -> "text/csv"
            else -> "text/plain"
        }, f.readText(Charsets.UTF_8), "attachment; filename=\"" + f.name + "\"")
    }

    private fun safe(f: File) = runCatching { f.canonicalFile.toPath().startsWith(root.canonicalFile.toPath()) }.getOrDefault(false)

    private fun fileJson(f: File) = "{\"name\":\"${OpTrace.jsonEscape(f.name)}\",\"size\":${f.length()},\"modified\":${f.lastModified()},\"url\":\"/download/${OpTrace.jsonEscape(f.name)}\"}"

    private fun respond(w: BufferedWriter, c: Int, m: String, b: String, d: String? = null) {
        val a = b.toByteArray(StandardCharsets.UTF_8)
        w.write("HTTP/1.1 " + c + " " + (if (c == 200) "OK" else "Error") + "\r\n")
        w.write("Content-Type: " + m + "\r\n")
        w.write("Content-Length: " + a.size + "\r\n")
        w.write("Cache-Control: no-store\r\n")
        if (d != null) w.write("Content-Disposition: " + d + "\r\n")
        w.write("Connection: close\r\n\r\n")
        w.write(b)
        w.flush()
    }

    companion object {
        private val DASHBOARD_HTML: String = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Whisper QNN OpTrace</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--dim:#8b949e;--acc:#58a6ff;--green:#3fb950;--orange:#d29922;--red:#f85149;--purple:#bc8cff}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:16px;max-width:1200px;margin:0 auto}
h1{font-size:20px;margin-bottom:4px}
h2{font-size:15px;margin:22px 0 10px;color:var(--acc)}
.sub{color:var(--dim);font-size:12px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:6px}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:700}
.kpi .l{font-size:11px;color:var(--dim);margin-top:2px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden;font-size:12.5px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600;background:#1c2128;position:sticky;top:0}
tr:last-child td{border-bottom:none}
.bar{height:8px;border-radius:4px;background:var(--acc);display:inline-block;vertical-align:middle;margin-right:6px;min-width:2px}
.wrap{max-height:420px;overflow:auto;border-radius:10px}
.dim{color:var(--dim)}
.tag{display:inline-block;padding:1px 7px;border-radius:8px;font-size:11px;background:#21262d;border:1px solid var(--border)}
.tag.device{color:var(--green)} .tag.host{color:var(--orange)} .tag.pre{color:var(--purple)} .tag.pipeline{color:var(--acc)}
#steps{display:flex;gap:3px;align-items:flex-end;height:120px;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px;overflow-x:auto}
#steps .col{flex:0 0 auto;width:26px;background:var(--acc);border-radius:3px 3px 0 0;position:relative}
#steps .col:hover{background:var(--green)}
#steps .col .tip{position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:#21262d;border:1px solid var(--border);padding:4px 8px;border-radius:6px;font-size:11px;display:none;white-space:nowrap;z-index:5}
#steps .col:hover .tip{display:block}
.err{color:var(--red)}
a{color:var(--acc);text-decoration:none}
button{background:#21262d;color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:5px 12px;cursor:pointer;font-size:12px}
button:hover{border-color:var(--acc)}
.row{display:flex;gap:10px;align-items:center;margin:10px 0}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
</style>
</head>
<body>
<h1>Whisper QNN · OpTrace 看板</h1>
<div class="sub">Snapdragon Voice Lab · HTP optrace + 全链路阶段计时 · <span id="uptime" class="dim">-</span> · 自动刷新 2s</div>

<div class="kpis">
  <div class="kpi"><div class="v" id="kRuns">-</div><div class="l">已完成 runs</div></div>
  <div class="kpi"><div class="v" id="kLast">-</div><div class="l">最近 run 总耗时</div></div>
  <div class="kpi"><div class="v" id="kRtf">-</div><div class="l">RTF(实时倍率,越小越快)</div></div>
  <div class="kpi"><div class="v" id="kStep">-</div><div class="l">decoder 步均耗时</div></div>
  <div class="kpi"><div class="v" id="kTok">-</div><div class="l">tokens/s</div></div>
  <div class="kpi"><div class="v" id="kHost">-</div><div class="l">host 侧时间占比</div></div>
</div>

<h2>阶段耗时(全部 runs 聚合)</h2>
<div class="wrap"><table id="stats"><thead><tr><th>阶段</th><th>类别</th><th>次数</th><th>合计 ms</th><th>均值</th><th>P50</th><th>P95</th><th>最大</th><th>占比</th></tr></thead><tbody></tbody></table></div>

<h2>Decoder 步时间线(最近 run)</h2>
<div id="steps"></div>

<h2>QNN HTP 算子明细(optrace CSV,最近一轮)</h2>
<div class="wrap"><table id="ops"><thead><tr><th>算子</th><th>来源</th><th>次数</th><th>合计 µs</th><th>均值 µs</th><th>最大 µs</th><th>占比</th></tr></thead><tbody></tbody></table></div>

<h2>运行时开关</h2>
<div class="row">
  <button onclick="toggleDbg()">调试诊断: <span id="cfgDbg">-</span></button>
  <button onclick="toggleProf()">HTP optrace: <span id="cfgProf">-</span></button>
  <span class="dim">profiling 切换后下一次转录生效</span>
</div>

<h2>原始文件</h2>
<div class="row"><button onclick="fetch('/api/trace/clear',{method:'POST'}).then(load)">清空 trace</button><span class="dim" id="files"></span></div>

<script>
let dbg=false, prof=false;
function applyCfg(d,p){ dbg=d; prof=p; document.getElementById('cfgDbg').textContent=d?'开':'关'; document.getElementById('cfgProf').textContent=p?'开':'关'; }
async function toggleDbg(){ await fetch('/api/config?debug='+(!dbg?1:0)); applyCfg(!dbg,prof); }
async function toggleProf(){ await fetch('/api/config?profiling='+(!prof?1:0)); applyCfg(dbg,!prof); }
fetch('/api/config').then(r=>r.json()).then(c=>applyCfg(c.debug,c.profiling)).catch(()=>{});
</script>

<script>
const fmt=(v,d=2)=>(v==null||isNaN(v))?'-':Number(v).toFixed(d);
const catClass=c=>({device:'device',host:'host',preprocess:'pre',pipeline:'pipeline',postprocess:'pre',init:'pre'}[c]||'');
async function load(){
  let t;
  try{ t=await (await fetch('/api/trace.json')).json(); }catch(e){ return; }
  document.getElementById('uptime').textContent='uptime '+fmt(t.uptime_ms/1000,0)+'s';
  const runs=t.runs||[];
  document.getElementById('kRuns').textContent=runs.length;
  const last=runs[0];
  if(last){
    document.getElementById('kLast').textContent=fmt(last.total_ms,1)+' ms';
    document.getElementById('kRtf').textContent=last.rtf!=null?fmt(last.rtf,4):'-';
    const steps=(last.spans||[]).filter(s=>s.name==='decoder.run');
    const gen=(last.spans||[]).filter(s=>s.name==='decoder.step');
    const stepMean=steps.length?steps.reduce((a,s)=>a+s.duration_ms,0)/steps.length:null;
    document.getElementById('kStep').textContent=stepMean!=null?fmt(stepMean,2)+' ms':'-';
    const audio=last.audio_seconds||0;
    const nTok=gen.length?Math.max(0,gen.length-4):0;
    document.getElementById('kTok').textContent=(audio>0&&last.total_ms>0)?fmt(nTok/(last.total_ms/1000),1):'-';
    const host=(last.spans||[]).filter(s=>s.category==='host').reduce((a,s)=>a+s.duration_ms,0);
    document.getElementById('kHost').textContent=fmt(host/last.total_ms*100,1)+'%';
    // step timeline
    const dev=steps.map(s=>s.duration_ms);
    const mx=Math.max(...dev,1);
    document.getElementById('steps').innerHTML=dev.map((v,i)=>{
      const h=Math.max(4,v/mx*100);
      const hostMs=(last.spans||[]).filter(s=>s.name==='decoder.host_copy'&&s.attrs&&s.attrs.step==String(i)).reduce((a,s)=>a+s.duration_ms,0);
      return `<div class="col" style="height:${'$'}{h}%"><div class="tip">step ${'$'}{i}<br>device ${'$'}{fmt(v,2)} ms<br>host_copy ${'$'}{fmt(hostMs,2)} ms</div></div>`;
    }).join('');
  }
  // stats table
  const totalAll=(t.run_stats||[]).reduce((a,s)=>a+s.total_ms,0)||1;
  document.querySelector('#stats tbody').innerHTML=(t.run_stats||[]).map(s=>{
    const pct=s.total_ms/totalAll*100;
    return `<tr><td>${'$'}{s.name}</td><td><span class="tag ${'$'}{catClass(s.category)}">${'$'}{s.category}</span></td><td>${'$'}{s.count}</td><td>${'$'}{fmt(s.total_ms,1)}</td><td>${'$'}{fmt(s.mean_ms,2)}</td><td>${'$'}{fmt(s.p50_ms,2)}</td><td>${'$'}{fmt(s.p95_ms,2)}</td><td>${'$'}{fmt(s.max_ms,2)}</td><td><span class="bar" style="width:${'$'}{Math.min(120,pct*3)}px"></span>${'$'}{fmt(pct,1)}%</td></tr>`;
  }).join('');
  // qnn ops
  const opTotal=(t.qnn_ops||[]).reduce((a,o)=>a+o.total_us,0)||1;
  document.querySelector('#ops tbody').innerHTML=(t.qnn_ops||[]).map(o=>
    `<tr><td class="mono">${'$'}{o.op}</td><td class="dim">${'$'}{o.file}</td><td>${'$'}{o.count}</td><td>${'$'}{fmt(o.total_us,0)}</td><td>${'$'}{fmt(o.mean_us,1)}</td><td>${'$'}{fmt(o.max_us,1)}</td><td><span class="bar" style="width:${'$'}{Math.min(120,o.share_pct*3)}px"></span>${'$'}{fmt(o.share_pct,1)}%</td></tr>`
  ).join('')||'<tr><td colspan="7" class="dim">尚无 optrace CSV —— 跑一次转录后自动出现</td></tr>';
  // files
  document.getElementById('files').innerHTML=(t.files||[]).map(f=>`<a href="${'$'}{f.url}">${'$'}{f.name}</a>(${'$'}{(f.size/1024).toFixed(0)}K)`).join(' · ');
}
load(); setInterval(load,2000);
</script>
</body>
</html>
""".trimIndent()
    }
}
