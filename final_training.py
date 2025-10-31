import os
import pickle
import tensorflow as tf
from keras import layers
from keras.applications import ResNet50, resnet
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
import math # Import math for splitting dataset

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Configs
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 27 # Use the same seed for reproducibility
OUTPUT_DIR = "TB_Chest_Radiography_Database_split"
PLOTS_DIR = "plots/final_training" # Separate plots for final training
os.makedirs(PLOTS_DIR, exist_ok=True)

# Save plot utility
def save_plot(fig, name):
    fig.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')
    plt.close(fig) # Close the figure after saving

# Load datasets
# We need train_ds, val_ds, and test_ds separately first to combine train+val
# and keep test completely separate.
def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(OUTPUT_DIR, "train"),
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(OUTPUT_DIR, "val"),
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(OUTPUT_DIR, "test"),
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    return train_ds, val_ds, test_ds

# Load datasets (they are initially batched)
train_ds_batched, val_ds_batched, test_ds_batched_initial = load_datasets() # Use temporary names for clarity

# --- Data Preparation for Final Training ---
# Unbatch initial datasets first to work with individual samples
train_ds_unbatched = train_ds_batched.unbatch()
val_ds_unbatched = val_ds_batched.unbatch()
test_ds_unbatched = test_ds_batched_initial.unbatch() # Also unbatch test now for consistent processing

# Combine original train and validation samples (now unbatched)
combined_ds_unbatched = train_ds_unbatched.concatenate(val_ds_unbatched)

# Calculate the size of the combined dataset (now unbatched)
# This loop counts individual samples
print("Calculating combined dataset size...")
combined_ds_unbatched_size = 0
for _ in combined_ds_unbatched:
    combined_ds_unbatched_size += 1
print(f"Combined Train + Val Dataset Size: {combined_ds_unbatched_size} samples")

# Define the split ratios for the final training (based on unbatched samples)
FINAL_TRAIN_RATIO = 0.8
FINAL_VAL_RATIO = 1.0 - FINAL_TRAIN_RATIO # Should be 0.2

final_train_size_unbatched = math.floor(FINAL_TRAIN_RATIO * combined_ds_unbatched_size)
final_val_size_unbatched = combined_ds_unbatched_size - final_train_size_unbatched

print(f"Final Training Set Size (from combined): {final_train_size_unbatched} samples")
print(f"Final Validation Set Size (from combined): {final_val_size_unbatched} samples")
# Alternative way to get actual test set size for printing:
print("Calculating test dataset size for printing...")
test_ds_actual_size = 0
# Iterate over the UNBATCHED test set to count samples
for _ in test_ds_unbatched:
    test_ds_actual_size += 1
print(f"Test Set Size (original unbatched): {test_ds_actual_size} samples")


# Shuffle the combined unbatched dataset
combined_ds_unbatched = combined_ds_unbatched.shuffle(buffer_size=combined_ds_unbatched_size, seed=SEED)

# Split the combined unbatched dataset into final training and final validation sets (still unbatched)
final_train_ds_unbatched = combined_ds_unbatched.take(final_train_size_unbatched)
final_val_ds_unbatched = combined_ds_unbatched.skip(final_train_size_unbatched)

# --- Now Batch, Cache, and Prefetch the Split Datasets ---
# Batch the individual samples into desired batch sizes
final_train_ds = final_train_ds_unbatched.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
final_val_ds = final_val_ds_unbatched.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
# Batch the test set consistently as well
test_ds = test_ds_unbatched.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE) # Use 'test_ds' for final batched dataset


# --- Preprocessing (ResNet specific) ---
# Apply preprocess_input AFTER batching but before prefetching

def preprocess_batched_dataset(dataset):
     # Preprocessing is applied to batches returned by .batch()
     dataset = dataset.map(lambda x, y: (resnet.preprocess_input(x), y))
     return dataset.prefetch(tf.data.AUTOTUNE) # Prefetch after transformations

# Apply preprocessing to the newly batched datasets
final_train_ds = preprocess_batched_dataset(final_train_ds)
final_val_ds = preprocess_batched_dataset(final_val_ds)
test_ds = preprocess_batched_dataset(test_ds) # Apply to the final batched test_ds

# --- End of Corrected Data Preparation Section ---


# --- Load Best Hyperparameters ---
try:
    with open("best_hyperparameters.pkl", "rb") as f:
        best_hps_values = pickle.load(f)
    print("\nLoaded Best Hyperparameters:")
    for param in ['dropout', 'dropout_2', 'dense_units', 'learning_rate']:
        if param in best_hps_values:
             print(f"{param}: {best_hps_values.get(param)}")
        else:
             print(f"{param}: Not found in loaded hyperparameters")

except FileNotFoundError:
    print("Error: best_hyperparameters.pkl not found.")
    print("Please run the hyperparameter tuning script (script 1) first to generate this file.")
    exit()


# --- Build Final Model (consistent with tuning architecture) ---
def build_final_model(hparams):
    # Ensure the base model remains frozen, as it was during tuning
    base_model = ResNet50(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
    base_model.trainable = False # Keep base model frozen

    x = layers.GlobalAveragePooling2D()(base_model.output)
    # Use dropout rates from tuning
    x = layers.Dropout(hparams['dropout'])(x)
    # Use dense units from tuning, NO L2 regularization to match tuning setup
    x = layers.Dense(
        hparams['dense_units'],
        activation='relu'
        # Removed: kernel_regularizer=regularizers.l2(0.001)
    )(x)
    # Use second dropout rate from tuning
    x = layers.Dropout(hparams['dropout_2'])(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Build the final model
final_model = build_final_model(best_hps_values)

# --- Compile Final Model ---
# Use the exact learning rate found by the tuner
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(best_hps_values['learning_rate']), # Use tuner's LR directly
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)



# --- Callbacks for Final Training ---
early_stopping = EarlyStopping(
    patience=8, # Increased patience slightly for training on larger data
    monitor='val_loss', # Monitor the loss on the temporary val set
    mode='min',
    verbose=1,
    restore_best_weights=True # Restore weights from the epoch with best val performance
)

# Save the best model weights during this final training run based on validation AUC
model_checkpoint_cb = ModelCheckpoint(
    os.path.join(PLOTS_DIR, "final_best_model_ResNet50.weights.h5"), # Save only weights
    save_best_only=True,
    monitor='val_auc', # Monitor val_auc to align with tuning objective
    mode='max', # We want to maximize AUC
    verbose=1
)

# --- Train on the Final Combined Dataset ---
# Train for enough epochs to allow Early Stopping to trigger
print("\nStarting final model training on combined dataset...")
history = final_model.fit(
    final_train_ds,
    validation_data=final_val_ds, # Validate on the temporary split
    epochs=100, # Set a sufficiently large number of epochs, EarlyStopping will stop it
    callbacks=[early_stopping, model_checkpoint_cb]
)

print("\nFinal model training finished.")

# --- Plotting Training History ---
print("Generating training history plots...")

# Get history data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
train_auc_history = history.history['auc'] # Renamed
val_auc_history = history.history['val_auc'] # Renamed
epochs_range = range(len(acc))

# Accuracy Plot
fig = plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
save_plot(fig, 'training_validation_accuracy.png')

# Loss Plot
fig = plt.figure(figsize=(8, 6))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
save_plot(fig, 'training_validation_loss.png')

# AUC Plot
fig = plt.figure(figsize=(8, 6))
plt.plot(epochs_range, train_auc_history, label='Training AUC') # Use the new variable name
plt.plot(epochs_range, val_auc_history, label='Validation AUC') # Use the new variable name
plt.legend(loc='lower right')
plt.title('Training and Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
save_plot(fig, 'training_validation_auc.png')

# Load the best weights saved by the checkpoint
final_model.load_weights(os.path.join(PLOTS_DIR, "final_best_model_ResNet50.weights.h5"))
print("\n\nLoaded best weights from final training for evaluation.")


# --- Evaluate on the Test Set ---
print("\nEvaluating final model on the test set...")
y_true, y_pred_classes, y_score = [], [], []

# Iterate directly over the final correctly batched and preprocessed test_ds
for x_batch, y_batch in test_ds: # test_ds is now the final batched dataset from the corrected pipeline
    y_true.extend(y_batch.numpy())
    preds = final_model.predict(x_batch, verbose=0) # Predict on the batch
    y_score.extend(preds.ravel()) # Store probabilities
    y_pred_classes.extend((preds.ravel() > 0.5).astype(int)) # Store predicted classes

y_true = np.array(y_true)
y_pred_classes = np.array(y_pred_classes)
y_score = np.array(y_score)


# --- Calculate and Plot Metrics ---
print("\nCalculating test set metrics...")

# Test Accuracy
test_acc = np.mean(y_pred_classes == y_true)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_score)
# Note: Precision_recall_curve does not return a single score like AUC.
# We can calculate average precision if needed, or just plot the curve.
# Let's calculate F1 score based on the 0.5 threshold
f1 = f1_score(y_true, y_pred_classes)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {roc_auc:.4f}")
print(f"Test F1 Score (threshold 0.5): {f1:.4f}")
print("Confusion Matrix:\n", cm)


# Plotting
print("Generating plots...")

# Confusion matrix plot
fig = plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal (0)', 'TB (1)'])
plt.yticks(tick_marks, ['Normal (0)', 'TB (1)'])
# Add text labels
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
save_plot(fig, 'confusion_matrix.png')

# ROC Curve plot
fig = plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
save_plot(fig, 'roc_curve.png')

# Precision-Recall Curve plot
fig = plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='teal', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower left")
save_plot(fig, 'pr_curve.png')

# Histogram of probabilities
fig = plt.figure(figsize=(8, 5))
plt.hist(y_score, bins=50, edgecolor='black', alpha=0.7)
plt.title('Prediction Probability Distribution on Test Set')
plt.xlabel('Predicted Probability ( closer to 0 is Normal, closer to 1 is TB)')
plt.ylabel('Frequency')
save_plot(fig, 'probability_histogram.png')


# True vs predicted count (on test set)
fig = plt.figure(figsize=(6, 5))
bar_x = np.arange(2)
counts_true = [np.sum(y_true == 0), np.sum(y_true == 1)]
counts_pred = [np.sum(y_pred_classes == 0), np.sum(y_pred_classes == 1)]
plt.bar(bar_x - 0.1, counts_true, width=0.2, label='True')
plt.bar(bar_x + 0.1, counts_pred, width=0.2, label='Predicted')
plt.xticks(bar_x, ['Normal (0)', 'TB (1)'])
plt.ylabel('Count')
plt.title("True vs Predicted Counts on Test Set")
plt.legend()
save_plot(fig, 'true_vs_predicted_counts.png')


print(f"\nEvaluation complete. Plots saved to {PLOTS_DIR}/")

# --- Save the final model for deployment ---
final_model_path = "final_model_for_deployment.h5" # Use .keras format (recommended by TensorFlow)
# Or you could use .h5 format: final_model_path = "final_model_for_deployment.h5"

final_model.save(final_model_path)
print(f"\nFinal model for deployment saved to '{final_model_path}'")