import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os, json

# Paths
DATASET_DIR = "dataset"
MODEL_PATH = "model/plant_disease_model.h5"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # fewer epochs since the dataset is smaller

def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # transfer learning freeze

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    if not os.path.exists(DATASET_DIR):
        raise ValueError(f"Dataset folder not found at {DATASET_DIR}")

    # Data preprocessing
    datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} classes:", list(train_gen.class_indices.keys()))

    model = build_model(num_classes)
    model.summary()

    # Training
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    # Save model and classes
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)

    with open("model/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    print("✅ Model and class indices saved!")


if __name__ == "__main__":
    main()
