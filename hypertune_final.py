import os
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import ResNet50, resnet
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner.tuners import Hyperband
from keras_tuner import Objective

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Configs
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Increased batch size for stability
SEED = 27
OUTPUT_DIR = "TB_Chest_Radiography_Database_split"
PLOTS_DIR = "plots/tuning"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load datasets
def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(OUTPUT_DIR, "train"), seed=SEED, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="binary")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(OUTPUT_DIR, "val"), seed=SEED, image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, label_mode="binary")
    return train_ds, val_ds

train_ds, val_ds = load_datasets()

# Preprocess
train_ds = train_ds.map(lambda x, y: (resnet.preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (resnet.preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)


# Build model with tunable hyperparameters
def build_model(hp):
    base_model = ResNet50(include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet")
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.6, step=0.1))(x)  # Expanded range
    x = layers.Dense(hp.Int('dense_units', min_value=128, max_value=1024, step=128), activation='relu')(x)  # Expanded range
    x = layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.6, step=0.1))(x)  # Expanded range
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [0.005, 0.003, 0.001, 0.0005])  # Adjusted learning rate values
        ),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']  # Added more metrics
    )

    return model

# Hyperband tuner for optimizing val_auc
tuner = Hyperband(
    build_model,
    objective=Objective("val_auc", direction="max"),
    max_epochs=30,
    factor=4,
    directory='resnet50_tuning_v2',
    project_name='tb_classification_v2'   
)

# Callbacks for tuning
stop_early = EarlyStopping(patience=4,  monitor='val_loss', mode='min', verbose=1)
model_checkpoint_cb = ModelCheckpoint(
    os.path.join(PLOTS_DIR, "best_model_ResNet50.h5"),
    save_best_only=True,
    monitor="val_auc",
    mode="max",
    verbose=1
)

# Search best hyperparameters
tuner.search(train_ds, validation_data=val_ds, epochs=30, callbacks=[stop_early, model_checkpoint_cb])

# Retrieve best model & hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]
# Save best hyperparameters
with open("best_hyperparameters.pkl", "wb") as f:
    pickle.dump(best_hps.values, f)


print("\nBest Hyperparameters:")
for param in ['dropout', 'dropout_2', 'dense_units', 'learning_rate']:
    print(f"{param}: {best_hps.get(param)}")

# Save tuner-best model
best_model.save('best_resnet50_tuned.h5')
print(f"Saved tuned model to 'best_resnet50_tuned.h5'")

# Plot & save functions
def save_plot(fig, name):
    fig.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')



# Visualize hyperparameter tuning results
def visualize_tuning_results(tuner):
    trials = tuner.oracle.get_best_trials(num_trials=80)  # Get top 80 trials

    if not trials:
        print("No trials found in the tuner.")
        return

    # Extract data
    trial_ids = [trial.trial_id for trial in trials]
    val_accuracies = [trial.metrics.get_last_value('val_accuracy') for trial in trials]
    val_losses = [trial.metrics.get_last_value('val_loss') for trial in trials]
    val_aucs = [trial.metrics.get_last_value('val_auc') for trial in trials]
    dropouts = [trial.hyperparameters.get('dropout') for trial in trials]
    dropouts_2 = [trial.hyperparameters.get('dropout_2') for trial in trials]
    dense_units = [trial.hyperparameters.get('dense_units') for trial in trials]
    learning_rates = [trial.hyperparameters.get('learning_rate') for trial in trials]

    # Plot: Validation Accuracy per Trial (Line)
    plt.figure(figsize=(10, 6))
    plt.plot(val_accuracies, marker='o', linestyle='-')
    plt.title('Validation Accuracy per Trial')
    plt.xlabel('Trial Index')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    save_plot(plt, 'val_accuracy_per_trial.png')
    plt.close()

    # Plot: Validation Loss per Trial (Line)
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses, marker='o', linestyle='-', color='red')
    plt.title('Validation Loss per Trial')
    plt.xlabel('Trial Index')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    save_plot(plt, 'val_loss_per_trial.png')
    plt.close()

    # Plot: Validation AUC per Trial (Line)
    plt.figure(figsize=(10, 6))
    plt.plot(val_aucs, marker='o', linestyle='-', color='green')
    plt.title('Validation AUC per Trial')
    plt.xlabel('Trial Index')
    plt.ylabel('Validation AUC')
    plt.grid(True)
    save_plot(plt, 'val_auc_per_trial.png')
    plt.close()

        
    # Scatter Plot: Validation Loss vs Accuracy
    plt.figure(figsize=(8, 6))
    plt.scatter(val_losses, val_accuracies, c='teal', edgecolors='black')
    plt.xlabel('Validation Loss')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Loss vs Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_loss_vs_val_acc.png"))
    plt.close()

    
    # Scatter Plot: Validation Loss vs AUC
    val_aucs = [trial.metrics.get_last_value('val_auc') for trial in trials]
    plt.figure(figsize=(8, 6))
    plt.scatter(val_losses, val_aucs, c='darkorange', edgecolors='black')
    plt.xlabel('Validation Loss')
    plt.ylabel('Validation AUC')
    plt.title('Validation Loss vs AUC')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "val_loss_vs_val_auc.png"))
    plt.close()
    
    # Plot: Hyperparameter Grid Search (Heatmap of Hyperparameters vs AUC)
    hyperparams_df = pd.DataFrame({
        'Dropout': dropouts,
        'Dropout_2': dropouts_2,
        'Dense Units': dense_units,
        'Learning Rate': learning_rates,
        'AUC': val_aucs
    })
    pivot_df = hyperparams_df.pivot_table(
        index='Dense Units', columns='Learning Rate', values='AUC', aggfunc='mean'
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".3f", linewidths=0.5)
    plt.title('Hyperparameter Grid Search (AUC vs Dense Units & Learning Rate)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Dense Units')
    save_plot(plt, 'hyperparameter_grid_search_heatmap.png')
    plt.close()

    # Best Trial Summary
    best_trial = trials[0]  # The best trial is now the first one in the list
    print("\nBest Trial Hyperparameters:")
    for param, value in best_trial.hyperparameters.values.items():
        print(f"{param}: {value}")
    print(f"Best Validation Accuracy: {best_trial.metrics.get_best_value('val_accuracy')}")
    print(f"Best Validation Loss: {best_trial.metrics.get_best_value('val_loss')}")
    print(f"Best Validation AUC: {best_trial.metrics.get_best_value('val_auc')}")

# Visualize tuning results after finding the best model
visualize_tuning_results(tuner)


