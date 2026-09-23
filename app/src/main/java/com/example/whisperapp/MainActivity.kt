package com.example.whisperapp

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import com.example.whisperapp.ui.App
import com.example.whisperapp.ui.ManagedModel
import com.example.whisperapp.ui.ModelManager
import android.util.Log

class MainActivity : ComponentActivity() {
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        val lfm2 = ModelManager.status(this, ManagedModel.LFM2)
        Log.i("MODEL_STATUS_BOOT", "ready=${lfm2.ready} root=${ModelManager.modelDir(this, ManagedModel.LFM2).absolutePath} missing=${lfm2.missingFiles} details=${lfm2.files.joinToString(";") { "${it.localFile.absolutePath}|exists=${it.localFile.exists()}|readable=${it.localFile.canRead()}|len=${it.localFile.length()}" }}")
        setContent { App() }
    }
}
