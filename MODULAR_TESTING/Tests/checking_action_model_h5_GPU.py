import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("✅ TensorFlow is using the GPU! 🚀")
else:
    print("❌ TensorFlow is NOT using the GPU!")
