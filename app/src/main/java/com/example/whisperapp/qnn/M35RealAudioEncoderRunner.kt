package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtEpDevice
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import com.example.whisperapp.audio.WhisperFeatureExtractor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import kotlin.math.max
import kotlin.math.min

object M35RealAudioEncoderRunner {
    private const val TAG = "M35_REAL_ENCODER"
    private const val EP_NAME = "QNNExecutionProvider"
    private val outputs = listOf("k_cache_cross_0","v_cache_cross_0","k_cache_cross_1","v_cache_cross_1","k_cache_cross_2","v_cache_cross_2","k_cache_cross_3","v_cache_cross_3")
    data class Result(val passed:Boolean,val report:String)

    fun run(context:Context,wav:File):Result {
        var session:OrtSession?=null; var options:OrtSession.SessionOptions?=null; var input:OnnxTensor?=null
        return try {
            val pcm=readWav(wav); check(pcm.rate==16000 && pcm.channels==1 && pcm.bits==16 && pcm.format==1) { "WAV contract invalid: rate=${pcm.rate} channels=${pcm.channels} bits=${pcm.bits} format=${pcm.format}" }
            Log.i(TAG,"PCM16 samples=${pcm.samples.size} rate=${pcm.rate} channels=${pcm.channels} bits=16 durationMs=${pcm.samples.size*1000L/pcm.rate}")
            Log.i(TAG,"MEL_START")
            val melStart=System.nanoTime()
            val heartbeat=Thread({
                while (!Thread.currentThread().isInterrupted) {
                    try { Thread.sleep(10000L) } catch (_:InterruptedException) { break }
                    val elapsed=(System.nanoTime()-melStart)/1_000_000L
                    Log.i(TAG,"STILL_RUNNING phase=MEL elapsedMs=$elapsed")
                }
            },"m35-mel-heartbeat").apply { isDaemon=true; start() }
            val features=try { WhisperFeatureExtractor().extract(pcm.samples,pcm.rate) } finally { heartbeat.interrupt() }
            val melMs=(System.nanoTime()-melStart)/1_000_000L
            Log.i(TAG,"MEL_END elapsedMs=$melMs")
            check(features.shape.contentEquals(longArrayOf(1,80,3000)))
            val fs=stats(features.data); check(fs.nan==0 && fs.inf==0 && fs.nonZero>0); Log.i(TAG,"MEL shape=${features.shape.contentToString()} dtype=FP32 min=${fs.min} max=${fs.max} mean=${fs.mean} finite=${fs.finite} nan=${fs.nan} inf=${fs.inf}")
            val model=copyAsset(context,"models/whisper/encoder_ctx.onnx"); val bin=copyAsset(context,"models/whisper/encoder.bin")
            val env=OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,TAG); System.loadLibrary("onnxruntime_providers_qnn"); env.registerExecutionProviderLibrary(EP_NAME,"libonnxruntime_providers_qnn.so")
            val device=env.epDevices.firstOrNull{it.epName==EP_NAME} ?: error("QNN EP device not exposed"); val backend=File(context.applicationInfo.nativeLibraryDir,"libQnnHtp.so"); check(backend.isFile)
            options=OrtSession.SessionOptions().apply { addConfigEntry("session.disable_cpu_ep_fallback","1"); setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO); addExecutionProvider(listOf(device as OrtEpDevice),mapOf("backend_path" to backend.absolutePath,"soc_model" to "57","htp_arch" to "75","offload_graph_io_quantization" to "0")) }
            Log.i(TAG,"ENCODER_ASSET bytes=${bin.length()}"); session=env.createSession(model.absolutePath,options); check(session.inputNames==setOf("input_features")); check(session.outputNames==outputs.toSet())
            val ii=session.inputInfo["input_features"]!!.info as TensorInfo; check(ii.shape.contentEquals(longArrayOf(1,80,3000))); check(ii.type==OnnxJavaType.FLOAT16)
            for(i in 0 until 4){assertInfo(session.outputInfo,"k_cache_cross_$i",longArrayOf(6,1,64,1500));assertInfo(session.outputInfo,"v_cache_cross_$i",longArrayOf(6,1,1500,64))}
            val fp16=ShortArray(features.data.size){f16(features.data[it])}; input=OnnxTensor.createTensor(env,ShortBuffer.wrap(fp16),longArrayOf(1,80,3000),OnnxJavaType.FLOAT16)
            val t0=System.nanoTime(); session.run(mapOf("input_features" to input)).use{r->check(r.size()==8); for((i,n) in session.outputNames.withIndex()){val s=stats16((r[i] as OnnxTensor).getShortBuffer());check(s.nan==0&&s.inf==0&&s.nonZero>0){"$n invalid $s"};Log.i(TAG,"$n dtype=FP16 min=${s.min} max=${s.max} mean=${s.mean} finite=${s.finite} nan=${s.nan} inf=${s.inf}")}}
            val ms=(System.nanoTime()-t0)/1_000_000.0; Log.i(TAG,"CROSS_KV_COMPUTED_ONCE=true executionMs=$ms"); Result(true,"Encoder Contract PASS\nReal PCM16 -> Mel -> FP16 -> HTP Encoder -> 8 Cross-KV\nExecution ms=$ms\nCross-KV computed once=true")
        } catch(t:Throwable){Log.e(TAG,"Encoder Contract FAIL: ${t.javaClass.name}: ${t.message}",t);Result(false,"Encoder Contract FAIL\n${t.javaClass.name}: ${t.message}")} finally {try{input?.close()}catch(_:Throwable){};try{session?.close()}catch(_:Throwable){};try{options?.close()}catch(_:Throwable){}}
    }
    private data class Wav(val rate:Int,val channels:Int,val bits:Int,val format:Int,val samples:ShortArray)
    private fun readWav(f:File):Wav {
        val b=f.readBytes()
        fun u16(o:Int)=(b[o].toInt() and 255) or ((b[o+1].toInt() and 255) shl 8)
        fun i32(o:Int)=u16(o) or (u16(o+2) shl 16)
        fun tag(o:Int)=String(b,o,4,Charsets.US_ASCII)
        check(tag(0)=="RIFF" && tag(8)=="WAVE")
        var p=12; var rate=0; var ch=0; var bits=0; var fmt=0; var ds=-1; var dn=0
        while(p+8<=b.size){
            val id=tag(p); val n=i32(p+4); val st=p+8
            if(id=="fmt "){fmt=u16(st); ch=u16(st+2); rate=i32(st+4); bits=u16(st+14)}
            if(id=="data"){ds=st; dn=min(n,b.size-st); break}
            p=st+n+(n and 1)
        }
        check(ds>=0)
        val sb=ByteBuffer.wrap(b,ds,dn).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val a=ShortArray(sb.remaining()); sb.get(a); return Wav(rate,ch,bits,fmt,a)
    }
    private data class S(val min:Float,val max:Float,val mean:Double,val finite:Int,val nan:Int,val inf:Int,val nonZero:Int)
    private fun stats(a:FloatArray):S{var lo=Float.POSITIVE_INFINITY;var hi=Float.NEGATIVE_INFINITY;var sum=0.0;var f=0;var n=0;var i=0;var z=0;for(v in a){when{v.isNaN()->n++;v.isInfinite()->i++;else->{f++;lo=min(lo,v);hi=max(hi,v);sum+=v;if(v!=0f)z++}}};return S(lo,hi,sum/f,f,n,i,z)}
    private fun stats16(b:ShortBuffer):S{val x=b.duplicate();var lo=Float.POSITIVE_INFINITY;var hi=Float.NEGATIVE_INFINITY;var sum=0.0;var f=0;var n=0;var i=0;var z=0;while(x.hasRemaining()){val v=h2f(x.get().toInt() and 65535);when{v.isNaN()->n++;v.isInfinite()->i++;else->{f++;lo=min(lo,v);hi=max(hi,v);sum+=v;if(v!=0f)z++}}};return S(lo,hi,if(f==0)Double.NaN else sum/f,f,n,i,z)}
    private fun h2f(x:Int):Float{val s=(x ushr 15) and 1;val e=(x ushr 10) and 31;val m=x and 1023;val v=when(e){0->m/1024f*Math.pow(2.0,-14.0).toFloat();31->if(m==0)Float.POSITIVE_INFINITY else Float.NaN;else->(1f+m/1024f)*Math.pow(2.0,(e-15).toDouble()).toFloat()};return if(s==0)v else -v}
    private fun f16(v:Float):Short{val b=v.toRawBits();val s=(b ushr 16) and 32768;val e=((b ushr 23) and 255)-127+15;val m=b and 8388607;return when{e<=0->if(e < -10)s.toShort() else(s or ((m or 8388608) shr (1-e+13))).toShort();e>=31->(s or 31744).toShort();else->(s or (e shl 10) or (m shr 13)).toShort()}}
    private fun assertInfo(m:Map<String,ai.onnxruntime.NodeInfo>,n:String,shape:LongArray){val i=m[n]?.info as? TensorInfo?:error("Missing output $n");check(i.shape.contentEquals(shape));check(i.type==OnnxJavaType.FLOAT16)}
    private fun copyAsset(c:Context,a:String):File{val f=File(c.cacheDir,a.substringAfterLast('/'));c.assets.open(a).use{input->f.outputStream().use{out->input.copyTo(out)}};return f}
}
