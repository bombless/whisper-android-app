package io.github.bombless.htp_playpen

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import io.github.bombless.htp_playpen.ui.App
import io.github.bombless.htp_playpen.ui.ManagedModel
import io.github.bombless.htp_playpen.ui.ModelManager
import android.util.Log

class MainActivity : ComponentActivity() {
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        val lfm = ModelManager.status(this, ManagedModel.LFM3)
        Log.i("MODEL_STATUS_BOOT", "ready=${lfm.ready} root=${ModelManager.modelDir(this, ManagedModel.LFM3).absolutePath} missing=${lfm.missingFiles} details=${lfm.files.joinToString(";") { "${it.localFile.absolutePath}|exists=${it.localFile.exists()}|readable=${it.localFile.canRead()}|len=${it.localFile.length()}" }}")
        setContent { App() }
    }
}
