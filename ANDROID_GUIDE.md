# Android Adaptation Guide 🤖📱

This guide explains how to adapt this Python implementation for Android.

## Overview

The current implementation is Python-based for rapid prototyping. For Android deployment, you'll need to:

1. Convert the ML model to TensorFlow Lite
2. Implement Android app with Kotlin/Java
3. Integrate with Android photo gallery
4. Handle background processing

## Step 1: Convert Model to TensorFlow Lite

### Export TFLite Model

```python
# export_model.py
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

# Load model
model = EfficientNetB0(weights='imagenet', include_top=True)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('efficientnet_b0.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Step 2: Android Project Setup

### Gradle Dependencies (app/build.gradle)

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    
    // Image loading
    implementation 'com.github.bumptech.glide:glide:4.15.1'
    
    // Coroutines for background processing
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
    
    // WorkManager for background tasks
    implementation 'androidx.work:work-runtime-ktx:2.8.1'
}
```

### Permissions (AndroidManifest.xml)

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" 
                     android:maxSdkVersion="28" />
    
    <application>
        <!-- Your app components -->
    </application>
</manifest>
```

## Step 3: Kotlin Implementation

### ImageClassifier.kt

```kotlin
import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class HomeworkImageClassifier(context: Context) {
    private var interpreter: Interpreter? = null
    private val homeworkKeywords = listOf(
        "notebook", "book", "paper", "pen", "whiteboard", "laptop"
    )
    
    init {
        val model = FileUtil.loadMappedFile(context, "efficientnet_b0.tflite")
        interpreter = Interpreter(model)
    }
    
    fun isHomeworkRelated(bitmap: Bitmap): Boolean {
        val inputBuffer = preprocessImage(bitmap)
        val outputBuffer = Array(1) { FloatArray(1000) } // ImageNet classes
        
        interpreter?.run(inputBuffer, outputBuffer)
        
        // Check top predictions
        val predictions = outputBuffer[0]
        val topIndices = predictions.indices.sortedByDescending { predictions[it] }
        
        // Check if any top prediction matches homework keywords
        // (You'll need to load ImageNet labels separately)
        return checkHomeworkKeywords(topIndices.take(5))
    }
    
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputSize = 224
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        buffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(inputSize * inputSize)
        scaled.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (pixel in pixels) {
            // EfficientNet preprocessing
            buffer.putFloat(((pixel shr 16 and 0xFF) - 127.0f) / 128.0f)
            buffer.putFloat(((pixel shr 8 and 0xFF) - 127.0f) / 128.0f)
            buffer.putFloat(((pixel and 0xFF) - 127.0f) / 128.0f)
        }
        
        return buffer
    }
    
    fun close() {
        interpreter?.close()
    }
}
```

### PhotoOrganizer.kt

```kotlin
import android.content.ContentValues
import android.content.Context
import android.provider.MediaStore
import kotlinx.coroutines.*

class PhotoOrganizer(private val context: Context) {
    private val classifier = HomeworkImageClassifier(context)
    
    suspend fun organizePhotos() = withContext(Dispatchers.IO) {
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATA
        )
        
        val cursor = context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            "${MediaStore.Images.Media.DATE_ADDED} DESC"
        )
        
        cursor?.use {
            while (it.moveToNext()) {
                val id = it.getLong(it.getColumnIndexOrThrow(MediaStore.Images.Media._ID))
                val uri = ContentUris.withAppendedId(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    id
                )
                
                // Load bitmap and classify
                val bitmap = MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                val isHomework = classifier.isHomeworkRelated(bitmap)
                
                if (isHomework) {
                    // Update album/bucket
                    updateImageAlbum(uri, "School")
                }
            }
        }
    }
    
    private fun updateImageAlbum(uri: Uri, albumName: String) {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.BUCKET_DISPLAY_NAME, albumName)
        }
        context.contentResolver.update(uri, values, null, null)
    }
}
```

### BackgroundWorker.kt

```kotlin
import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters

class PhotoOrganizerWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        return try {
            val organizer = PhotoOrganizer(applicationContext)
            organizer.organizePhotos()
            Result.success()
        } catch (e: Exception) {
            Result.retry()
        }
    }
}
```

## Step 4: Create Hidden Album

### Option 1: Custom Album with MediaStore

Create a new album in MediaStore and mark photos:

```kotlin
fun createSchoolAlbum(context: Context) {
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "School")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/School")
    }
    // Use this for organizing photos
}
```

### Option 2: Use .nomedia File

Create `.nomedia` file in a folder to hide it from gallery:

```kotlin
fun createHiddenFolder(context: Context) {
    val folder = File(context.getExternalFilesDir(null), "school_photos")
    folder.mkdirs()
    File(folder, ".nomedia").createNewFile()
}
```

## Step 5: Schedule Background Processing

```kotlin
import androidx.work.*
import java.util.concurrent.TimeUnit

fun schedulePhotoOrganization(context: Context) {
    val constraints = Constraints.Builder()
        .setRequiresCharging(true)
        .setRequiresDeviceIdle(true)
        .build()
    
    val workRequest = PeriodicWorkRequestBuilder<PhotoOrganizerWorker>(
        24, TimeUnit.HOURS
    ).setConstraints(constraints)
     .build()
    
    WorkManager.getInstance(context).enqueueUniquePeriodicWork(
        "photo_organization",
        ExistingPeriodicWorkPolicy.KEEP,
        workRequest
    )
}
```

## Step 6: UI (Optional)

### Simple Activity

```kotlin
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        findViewById<Button>(R.id.organizeButton).setOnClickListener {
            lifecycleScope.launch {
                val organizer = PhotoOrganizer(this@MainActivity)
                organizer.organizePhotos()
                Toast.makeText(this@MainActivity, "Photos organized!", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Schedule background work
        schedulePhotoOrganization(this)
    }
}
```

## Performance Optimization

### 1. Use GPU Delegate

```kotlin
val options = Interpreter.Options()
val gpuDelegate = GpuDelegate()
options.addDelegate(gpuDelegate)
interpreter = Interpreter(model, options)
```

### 2. Quantized Model

Use quantized TFLite model for faster inference:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### 3. Batch Processing

Process multiple images at once:

```kotlin
val batchSize = 5
val inputBuffer = Array(batchSize) { preprocessImage(bitmaps[it]) }
```

## Testing

### Test on Emulator

1. Create Android Virtual Device (AVD)
2. Load test images into emulator gallery
3. Run the app and verify classification

### Test on Device

1. Enable USB debugging
2. Install APK: `adb install app-debug.apk`
3. Grant permissions
4. Monitor logs: `adb logcat | grep PhotoOrganizer`

## Alternative: Use ML Kit

Google's ML Kit provides pre-built solutions:

```gradle
implementation 'com.google.mlkit:image-labeling:17.0.7'
```

```kotlin
val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)
labeler.process(inputImage)
    .addOnSuccessListener { labels ->
        val isHomework = labels.any { 
            it.text.contains("book", ignoreCase = true) && it.confidence > 0.7
        }
    }
```

## Resources

- [TensorFlow Lite Android Guide](https://www.tensorflow.org/lite/android)
- [MediaStore API](https://developer.android.com/training/data-storage/shared/media)
- [WorkManager Guide](https://developer.android.com/topic/libraries/architecture/workmanager)
- [ML Kit Documentation](https://developers.google.com/ml-kit)

## Notes

- Android 10+ requires scoped storage
- Request proper permissions at runtime
- Consider battery usage for background tasks
- Test with various image sizes and formats
- Handle edge cases (corrupted images, etc.)

---

Happy coding! 🚀
