import keras
import tensorflow as tf
import tempfile
import shutil


def h5_to_tflite(h5_path="classifier_final.h5", output_path="final.tflite"):
    model = keras.models.load_model(h5_path)

    temp_dir = tempfile.mkdtemp()
    saved_model_path = f"{temp_dir}/saved_model"

    model.export(saved_model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()
    print(f"Conversion successful, model size: {len(tflite_model)} bytes")

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    shutil.rmtree(temp_dir)

    print(f"TFLite model saved to {output_path}")

if __name__ == "__main__":
    h5_to_tflite()