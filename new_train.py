import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils import class_weight

# -------------------- Parameters --------------------
dataset_path = './medicinal_plants_dataset/'  
weights_path = './mobilenet_v2_weights.h5' 
img_size = (224, 224)
batch_size = 64
num_epochs_top = 60      
num_epochs_finetune = 50  

# -------------------- Filter out rare classes --------------------
class_counts = {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in os.listdir(dataset_path)}
valid_classes = [cls for cls, count in class_counts.items() if count > 1]

if not valid_classes:
    raise ValueError("No classes with more than 1 image found!")

print(f"✅ Classes used for training: {valid_classes}")

# -------------------- Data Generators --------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    classes=valid_classes,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    classes=valid_classes,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(valid_classes)
print(f"✅ Number of classes: {num_classes}")

# -------------------- Compute Class Weights --------------------
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# -------------------- MobileNetV2 Base Model --------------------
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))

if os.path.exists(weights_path):
    base_model.load_weights(weights_path)
    print("✅ MobileNetV2 weights loaded from local file.")
else:
    print("⚠️ Weights not found, training from scratch.")

# Freeze base layers initially
for layer in base_model.layers:
    layer.trainable = False

# -------------------- Custom Top Layers --------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=3e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------- Callbacks --------------------
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_mobilenetv2_model1.h5', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# -------------------- Stage 1: Train Top Layers --------------------
history_top = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs_top,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# -------------------- Stage 2: Fine-tune Base --------------------
# Unfreeze top 30 layers of base
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs_finetune,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# -------------------- Evaluate Model --------------------
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"\n✅ Final Validation Accuracy: {val_acc*100:.2f}%")

# -------------------- Plot Accuracy and Loss --------------------
history = {
    'accuracy': history_top.history['accuracy'] + history_finetune.history['accuracy'],
    'val_accuracy': history_top.history['val_accuracy'] + history_finetune.history['val_accuracy'],
    'loss': history_top.history['loss'] + history_finetune.history['loss'],
    'val_loss': history_top.history['val_loss'] + history_finetune.history['val_loss']
}

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.grid()
plt.show()
