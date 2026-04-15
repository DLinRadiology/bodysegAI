"""
One-time script: convert Keras HDF5 weights to ONNX format.
Run this once, then the app only needs onnxruntime (no TensorFlow).

Usage:
    pip install -r requirements-convert.txt
    python convert_model.py
"""
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

act = "tanh"

def get_unet():
    inputs = Input((512, 512, 1))
    conv1 = Conv2D(16, (3, 3), activation=act, padding="same")(inputs)
    conv1 = Conv2D(16, (3, 3), activation=act, padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation=act, padding="same")(pool1)
    conv2 = Conv2D(32, (3, 3), activation=act, padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation=act, padding="same")(pool2)
    conv3 = Conv2D(64, (3, 3), activation=act, padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation=act, padding="same")(pool3)
    conv4 = Conv2D(128, (3, 3), activation=act, padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv45 = Conv2D(256, (3, 3), activation=act, padding="same")(pool4)
    conv45 = Conv2D(256, (3, 3), activation=act, padding="same")(conv45)
    pool45 = MaxPooling2D(pool_size=(2, 2))(conv45)

    conv55 = Conv2D(512, (3, 3), activation=act, padding="same")(pool45)
    conv55 = Conv2D(512, (3, 3), activation=act, padding="same")(conv55)
    pool55 = MaxPooling2D(pool_size=(2, 2))(conv55)

    conv56 = Conv2D(1024, (3, 3), activation=act, padding="same")(pool55)
    conv56 = Conv2D(1024, (3, 3), activation=act, padding="same")(conv56)

    up54 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(conv56), conv55], axis=3)
    conv57 = Conv2D(512, (3, 3), activation=act, padding="same")(up54)
    conv57 = Conv2D(512, (3, 3), activation=act, padding="same")(conv57)

    up55 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv57), conv45], axis=3)
    conv58 = Conv2D(256, (3, 3), activation=act, padding="same")(up55)
    conv58 = Conv2D(256, (3, 3), activation=act, padding="same")(conv58)

    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv58), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation=act, padding="same")(up6)
    conv6 = Conv2D(128, (3, 3), activation=act, padding="same")(conv6)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation=act, padding="same")(up7)
    conv7 = Conv2D(64, (3, 3), activation=act, padding="same")(conv7)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation=act, padding="same")(up8)
    conv8 = Conv2D(32, (3, 3), activation=act, padding="same")(conv8)

    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation=act, padding="same")(up9)
    conv9 = Conv2D(16, (3, 3), activation=act, padding="same")(conv9)

    conv10 = Conv2D(3, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model


if __name__ == "__main__":
    print("Building model...")
    model = get_unet()

    weights_path = os.path.join(ROOT_DIR, "xRobotstuffx", "weights.h5")
    print(f"Loading weights from {weights_path}")
    model.load_weights(weights_path)

    # Save as SavedModel format (Keras 3 uses .export())
    saved_model_dir = os.path.join(ROOT_DIR, "xRobotstuffx", "saved_model")
    print(f"Exporting SavedModel to {saved_model_dir}")
    model.export(saved_model_dir)

    # Convert to ONNX
    onnx_path = os.path.join(ROOT_DIR, "xRobotstuffx", "model.onnx")
    print(f"Converting to ONNX: {onnx_path}")
    import subprocess
    subprocess.run([
        "python", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", onnx_path,
        "--opset", "13",
    ], check=True)

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"\nONNX model verified:")
    print(f"  Input:  {inp.name} {inp.shape} {inp.type}")
    print(f"  Output: {out.name} {out.shape} {out.type}")

    # Test inference
    dummy = np.zeros((1, 512, 512, 1), dtype=np.float32)
    result = sess.run(None, {inp.name: dummy})
    print(f"  Test output shape: {result[0].shape}")
    print("\nConversion complete!")
