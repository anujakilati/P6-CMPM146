import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
from tensorflow.keras import models

# ============================================================
# 1. RETRAIN AND PLOT: Val accuracy with/without transfer
# ============================================================
from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel

input_shape = (150, 150, 3)
categories_count = 2
epochs = 15

train_dataset, validation_dataset, test_dataset = get_transfer_datasets()

# --- Transferred model ---
print("=" * 60)
print("TRANSFERRED MODEL")
print("=" * 60)
transferred_model = TransferedModel(input_shape, categories_count)
transferred_model.model.summary()
transfer_history = transferred_model.train_model(train_dataset, validation_dataset, epochs)

# --- Random model ---
print("=" * 60)
print("RANDOM MODEL (no transfer)")
print("=" * 60)
random_model = RandomModel(input_shape, categories_count)
random_model.model.summary()
random_history = random_model.train_model(train_dataset, validation_dataset, epochs)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), transfer_history.history['val_accuracy'], 'b-o', label='With Transfer')
plt.plot(range(1, epochs + 1), random_history.history['val_accuracy'], 'r-s', label='Without Transfer')
plt.title('Far Transfer from Facial Recognition to Doom vs Animal Crossing')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('results/transfer_val_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to results/transfer_val_accuracy.png")

# ============================================================
# 2. TEST SET ACCURACY
# ============================================================
print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
transfer_loss, transfer_acc = transferred_model.model.evaluate(test_dataset)
random_loss, random_acc = random_model.model.evaluate(test_dataset)
print(f"\nWith Transfer    — Test Accuracy: {transfer_acc:.4f}")
print(f"Without Transfer — Test Accuracy: {random_acc:.4f}")

# ============================================================
# 3. MODEL SUMMARIES (save to text file for easy copy-paste)
# ============================================================

with open('results/model_summaries.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("TRANSFERRED MODEL SUMMARY\n")
    f.write("=" * 60 + "\n")
    transferred_model.model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n" + "=" * 60 + "\n")
    f.write("RANDOM MODEL SUMMARY (without transfer)\n")
    f.write("=" * 60 + "\n")
    random_model.model.summary(print_fn=lambda x: f.write(x + '\n'))
print("Model summaries saved to results/model_summaries.txt")

# ============================================================
# 4. SAMPLE IMAGES — Facial Recognition classes
# ============================================================
facial_categories = ['neutral', 'happy', 'surprise']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Facial Recognition Classes', fontsize=14)

for i, emotion in enumerate(facial_categories):
    emotion_dir = os.path.join('train', emotion)
    if os.path.exists(emotion_dir):
        imgs = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            img = imread(os.path.join(emotion_dir, imgs[0]))
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(emotion.title())
            axes[i].axis('off')

plt.savefig('results/facial_classes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Facial class images saved to results/facial_classes.png")

# ============================================================
# 5. SAMPLE IMAGES — Doom vs Animal Crossing classes
# ============================================================
transfer_dir = 'kaggle/Doom_vs_AnimalCrossing'
transfer_categories = ['animal_crossing', 'doom']

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Doom vs Animal Crossing Classes', fontsize=14)

# Specific images to use
transfer_images = {
    'animal_crossing': '0czcmw8rmsz41.jpg',
    'doom': None,  # use first available
}

for i, category in enumerate(transfer_categories):
    cat_dir = os.path.join(transfer_dir, category)
    if os.path.exists(cat_dir):
        chosen = transfer_images.get(category)
        if chosen and os.path.exists(os.path.join(cat_dir, chosen)):
            img_file = chosen
        else:
            imgs = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            img_file = imgs[0] if imgs else None
        if img_file:
            img = imread(os.path.join(cat_dir, img_file))
            axes[i].imshow(img)
            axes[i].set_title(category.replace('_', ' ').title())
            axes[i].axis('off')

plt.savefig('results/transfer_classes.png', dpi=150, bbox_inches='tight')
plt.show()
print("Transfer class images saved to results/transfer_classes.png")

print("\n" + "=" * 60)
print("ALL REPORT ASSETS GENERATED IN results/")
print("=" * 60)
print("Files:")
print("  - results/transfer_val_accuracy.png  (val accuracy plot)")
print("  - results/model_summaries.txt        (both model summaries)")
print("  - results/facial_classes.png         (facial recognition classes)")
print("  - results/transfer_classes.png       (doom vs animal crossing classes)")
print(f"\nWith Transfer    — Test Accuracy: {transfer_acc:.4f}")
print(f"Without Transfer — Test Accuracy: {random_acc:.4f}")