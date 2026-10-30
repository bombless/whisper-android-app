package io.github.bombless.htp_playpen.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

data class ImageGenHistoryItem(
    val id: String = UUID.randomUUID().toString(),
    val prompt: String,
    val negativePrompt: String,
    val timestamp: Long = System.currentTimeMillis(),
    val imageFileName: String,
)

object ImageGenHistoryManager {
    private const val TAG = "ImageGenHistoryManager"
    private const val HISTORY_DIR_NAME = "image_gen_history"
    private const val IMAGES_DIR_NAME = "images"
    private const val INDEX_FILE_NAME = "history.json"

    private fun getBaseDir(context: Context): File {
        return File(context.filesDir, HISTORY_DIR_NAME).apply {
            if (!exists()) mkdirs()
        }
    }

    private fun getImagesDir(context: Context): File {
        return File(getBaseDir(context), IMAGES_DIR_NAME).apply {
            if (!exists()) mkdirs()
        }
    }

    private fun getIndexFile(context: Context): File {
        return File(getBaseDir(context), INDEX_FILE_NAME)
    }

    @Synchronized
    fun loadHistory(context: Context): List<ImageGenHistoryItem> {
        val file = getIndexFile(context)
        if (!file.exists()) return emptyList()
        return try {
            val jsonStr = file.readText()
            val array = JSONArray(jsonStr)
            val list = mutableListOf<ImageGenHistoryItem>()
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                list.add(
                    ImageGenHistoryItem(
                        id = obj.optString("id", UUID.randomUUID().toString()),
                        prompt = obj.getString("prompt"),
                        negativePrompt = obj.optString("negativePrompt", ""),
                        timestamp = obj.optLong("timestamp", System.currentTimeMillis()),
                        imageFileName = obj.getString("imageFileName")
                    )
                )
            }
            list
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load history", e)
            emptyList()
        }
    }

    @Synchronized
    fun saveItem(
        context: Context,
        prompt: String,
        negativePrompt: String,
        bitmap: Bitmap
    ): ImageGenHistoryItem? {
        return try {
            val id = UUID.randomUUID().toString()
            val imageFileName = "${id}.png"
            val imageFile = File(getImagesDir(context), imageFileName)
            FileOutputStream(imageFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            val newItem = ImageGenHistoryItem(
                id = id,
                prompt = prompt,
                negativePrompt = negativePrompt,
                timestamp = System.currentTimeMillis(),
                imageFileName = imageFileName
            )

            val currentList = loadHistory(context).toMutableList()
            currentList.add(0, newItem) // Latest first

            val array = JSONArray()
            for (item in currentList) {
                val obj = JSONObject().apply {
                    put("id", item.id)
                    put("prompt", item.prompt)
                    put("negativePrompt", item.negativePrompt)
                    put("timestamp", item.timestamp)
                    put("imageFileName", item.imageFileName)
                }
                array.put(obj)
            }
            getIndexFile(context).writeText(array.toString())
            newItem
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save history item", e)
            null
        }
    }

    @Synchronized
    fun deleteItem(context: Context, id: String) {
        try {
            val currentList = loadHistory(context).toMutableList()
            val target = currentList.find { it.id == id }
            if (target != null) {
                currentList.remove(target)
                val imgFile = File(getImagesDir(context), target.imageFileName)
                if (imgFile.exists()) imgFile.delete()

                val array = JSONArray()
                for (item in currentList) {
                    val obj = JSONObject().apply {
                        put("id", item.id)
                        put("prompt", item.prompt)
                        put("negativePrompt", item.negativePrompt)
                        put("timestamp", item.timestamp)
                        put("imageFileName", item.imageFileName)
                    }
                    array.put(obj)
                }
                getIndexFile(context).writeText(array.toString())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete history item", e)
        }
    }

    fun loadBitmap(context: Context, fileName: String): Bitmap? {
        val file = File(getImagesDir(context), fileName)
        if (!file.exists()) return null
        return try {
            BitmapFactory.decodeFile(file.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decode bitmap $fileName", e)
            null
        }
    }
}
