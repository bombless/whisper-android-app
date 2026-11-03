plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "io.github.bombless.htp_playpen"
    compileSdk = 35
    ndkVersion = "25.2.9519653"

    packaging {
        jniLibs.useLegacyPackaging = true
        jniLibs.pickFirsts += setOf(
            "**/libsherpa-onnx-c-api.so",
            "**/libsherpa-onnx-cxx-api.so",
            "**/libsherpa-onnx-jni.so",
            // Keep the app's newer 1.27.0 ORT runtime for QNN and Java OrtEnvironment.
            "**/libonnxruntime.so",
        )
    }

    androidResources {
        // Keep assets stored uncompressed. The QNN runner opens each context binary with
        // AssetManager.openFd() to learn its real length and skip re-copying multi-GB
        // binaries into cacheDir on every chunk; openFd is only available uncompressed.
        noCompress += listOf("")
    }

    defaultConfig {
        applicationId = "io.github.bombless.htp_playpen"
        minSdk = 29
        targetSdk = 35
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        versionCode = 2
        versionName = "0.2.0"

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=c++_static"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }

    testOptions {
        // The mel caches log progress via android.util.Log, which the stub android.jar
        // does not implement; without this every incremental-cache unit test throws
        // "Method i in android.util.Log not mocked".
        unitTests.isReturnDefaultValues = true
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(file("../lib/assets"))
            jniLibs.srcDir(file("../lib/runtime/jniLibs"))
        }
    }
}

dependencies {
    implementation(project(":shared-tools"))
    implementation("org.mozilla:rhino:1.9.1")
    implementation(fileTree(mapOf("dir" to "../lib/runtime", "include" to listOf("*.jar"))))
    implementation(files("../lib/runtime/sherpa-onnx-1.13.8-no-ort.aar"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.27.0")
    // Kokoro ONNX/G2P runtime; model weights are downloaded into app-private storage.
    implementation("audio.soniqo:speech:0.0.20")
    implementation("com.qualcomm.qti:onnxruntime-android-qnn:2.6.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")
    implementation(platform("androidx.compose:compose-bom:2025.02.00"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.1")
    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation(kotlin("test"))
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
