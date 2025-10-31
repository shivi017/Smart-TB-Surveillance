import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import time
from tensorflow.keras.metrics import AUC # type: ignore
from matplotlib.cm import get_cmap



IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 18
SEED = 27
FOLDS = 5

MODELS = [
    "EfficientNetB0", "MobileNetV2", "ResNet50",
    "DenseNet121", "InceptionV3", "Xception"
]

CLASS_NAMES = ["Normal", "Tuberculosis"]
SPLITS = ["train", "val", "test"]
BASE_DIR = "TB_Chest_Radiography_Database_split"

all_fpr = {}
all_tpr = {}
all_precision = {}
all_recall = {}



# --- GPU Setup ---
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            

def load_data_as_numpy():
    file_paths = []
    labels = []
    for label, cls in enumerate(CLASS_NAMES):
        for split in SPLITS:
            dir_path = os.path.join(BASE_DIR, split, cls)
            if os.path.exists(dir_path):
                for fname in os.listdir(dir_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_paths.append(os.path.join(dir_path, fname))
                        labels.append(label)
    return np.array(file_paths), np.array(labels)


def get_model_fn(name):
    models = {
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "ResNet50": tf.keras.applications.ResNet50,
        "DenseNet121": tf.keras.applications.DenseNet121,
        "InceptionV3": tf.keras.applications.InceptionV3,
        "Xception": tf.keras.applications.Xception
    }
    preprocess = {
        "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "ResNet50": tf.keras.applications.resnet.preprocess_input,
        "DenseNet121": tf.keras.applications.densenet.preprocess_input,
        "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
        "Xception": tf.keras.applications.xception.preprocess_input
    }
    return models[name], preprocess[name]


def build_model(model_fn):
    base = model_fn(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(base.input, output)


def create_dataset(paths, labels, preprocess_fn, shuffle=False):
    def _load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def count_class_distribution(ds, split_name):
    count_0, count_1 = 0, 0
    for _, labels in ds:
        count_0 += tf.math.count_nonzero(labels == 0).numpy()
        count_1 += tf.math.count_nonzero(labels == 1).numpy()
    print(f"{split_name} split class counts: Normal={count_0}, TB={count_1}")


def plot_class_distribution(y, title, save_path):
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(CLASS_NAMES, counts))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="pastel")
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.pie(list(class_counts.values()), labels=list(class_counts.keys()), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title(title + " (Pie Chart)")
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_pie.png"))
    plt.close()


def train_on_fold(X, y, model_name):
    model_fn, preprocess_fn = get_model_fn(model_name)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    metrics = []
    model_times = []
    all_y_true, all_y_pred = [], []
    model_conf_matrices = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n Fold {fold+1}/{FOLDS} - {model_name}")

        plot_class_distribution(y[train_idx], f"{model_name} Fold {fold+1} - Train Class Distribution", f"{model_name}_fold{fold+1}_train_distribution.png")
        plot_class_distribution(y[test_idx], f"{model_name} Fold {fold+1} - Test Class Distribution", f"{model_name}_fold{fold+1}_test_distribution.png")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        train_ds = create_dataset(X_train, y_train, preprocess_fn, shuffle=True)
        test_ds = create_dataset(X_test, y_test, preprocess_fn, shuffle=False)

        count_class_distribution(train_ds, "Training")
        count_class_distribution(test_ds, "Testing")

        model = build_model(model_fn)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC(name="auc")])

        checkpoint_path = f"model_{model_name}_fold{fold+1}.h5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
        early_stop = EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss', mode='min', verbose=1)

        start_time = time.time()
        history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[early_stop, checkpoint], verbose=1)
        elapsed_time = time.time() - start_time
        model_times.append(elapsed_time)

        y_true, y_probs = [], []
        # Predict in batch
        y_probs = model.predict(test_ds)
        y_preds = (y_probs.flatten() > 0.5).astype(int)

        # Collect true labels
        y_true = np.concatenate([label.numpy() for _, label in test_ds], axis=0)

        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_preds)
        acc = accuracy_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)

        all_fpr.setdefault(model_name, []).append(fpr)
        all_tpr.setdefault(model_name, []).append(tpr)
        all_precision.setdefault(model_name, []).append(precision)
        all_recall.setdefault(model_name, []).append(recall)

        metrics.append({
            "fold": fold+1,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "training_time": elapsed_time
        })

        cm = confusion_matrix(y_true, y_preds)
        model_conf_matrices.append(cm)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name} Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"cm_{model_name}_fold{fold+1}.png")
        plt.close()
        
        
        
        report = classification_report(y_true, y_preds, target_names=CLASS_NAMES)
        with open(f"classification_report_{model_name}_fold{fold+1}.txt", "w") as f:
            f.write(report)



        tf.keras.backend.clear_session()  # Clear TensorFlow session first
        del model, history, test_ds, train_ds, X_train, X_test, y_train, y_test
        gc.collect()  # Trigger garbage collection to free memory
        print(f"Memory cleared for {model_name} after fold {fold+1}...")

    # Compute global statistics across folds
    global_metrics = {
        "accuracy_mean": np.mean([m["accuracy"] for m in metrics]),
        "accuracy_std": np.std([m["accuracy"] for m in metrics]),
        "f1_mean": np.mean([m["f1"] for m in metrics]),
        "f1_std": np.std([m["f1"] for m in metrics]),
        "roc_auc_mean": np.mean([m["roc_auc"] for m in metrics]),
        "roc_auc_std": np.std([m["roc_auc"] for m in metrics]),
        "pr_auc_mean": np.mean([m["pr_auc"] for m in metrics]),
        "pr_auc_std": np.std([m["pr_auc"] for m in metrics]),
        "training_time_mean": np.mean(model_times),
        "training_time_std": np.std(model_times)
    }
    
    return metrics,model_conf_matrices,global_metrics

def plot_avg_confusion_matrices(models_conf_matrices):
    for model_name, cm_list in models_conf_matrices.items():
        avg_cm = np.mean(cm_list, axis=0)
        plt.figure(figsize=(5, 4))
        sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues")
        plt.title(f"Avg Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"avg_cm_{model_name}.png")
        plt.close()




def plot_mean_roc_pr_curves():
    color_map = get_cmap('tab10')  # 10 visually distinct colors
    model_colors = {model: color_map(i) for i, model in enumerate(MODELS)}

    for curve_type, all_curves, xlabel, ylabel, title, filename in [
        ("ROC", (all_fpr, all_tpr), "False Positive Rate", "True Positive Rate", "Mean ROC Curve", "mean_roc_comparison.png"),
        ("PR", (all_recall, all_precision), "Recall", "Precision", "Mean Precision-Recall Curve", "mean_pr_comparison.png")
    ]:
        plt.figure(figsize=(10, 7))

        for idx, model_name in enumerate(MODELS):
            xs = np.linspace(0, 1, 100)
            ys_interp = []
            if model_name in all_curves[0]:
                for fold_x, fold_y in zip(all_curves[0][model_name], all_curves[1][model_name]):
                    interp_y = np.interp(xs, fold_x, fold_y)
                    ys_interp.append(interp_y)

                if ys_interp:
                    mean_y = np.mean(ys_interp, axis=0)
                    std_y = np.std(ys_interp, axis=0)
                    plt.plot(xs, mean_y, label=model_name, color=model_colors[model_name])
                    plt.fill_between(xs, mean_y - std_y, mean_y + std_y, alpha=0.2, color=model_colors[model_name])

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='lower right' if curve_type == "ROC" else 'lower left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_roc_pr_auc_comparison(models_metrics):
    roc_auc_means = []
    pr_auc_means = []
    models = []

    for model_name, metrics in models_metrics.items():
        roc_auc_scores = [m["roc_auc"] for m in metrics]
        pr_auc_scores = [m["pr_auc"] for m in metrics]
        
        roc_auc_means.append(np.mean(roc_auc_scores))
        pr_auc_means.append(np.mean(pr_auc_scores))
        models.append(model_name)
        
    plt.figure(figsize=(10, 6))
    plt.plot(models, roc_auc_means, label="ROC AUC", marker='o')
    plt.plot(models, pr_auc_means, label="PR AUC", marker='o')
    plt.title("ROC AUC and PR AUC Comparison Across Models")
    plt.xlabel("Model")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_pr_auc_comparison.png")
    plt.close()


def plot_global_metrics_bar(global_metrics_dict):
    metrics_to_plot = ["accuracy", "f1", "roc_auc", "pr_auc", "training_time"]
    means = {m: [] for m in metrics_to_plot}
    stds = {m: [] for m in metrics_to_plot}
    model_names = list(global_metrics_dict.keys())

    for m in metrics_to_plot:
        for model in model_names:
            means[m].append(global_metrics_dict[model][f"{m}_mean"])
            stds[m].append(global_metrics_dict[model][f"{m}_std"])

    for m in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, means[m], yerr=stds[m], capsize=5, color="skyblue")
        plt.ylabel(f"{m.replace('_', ' ').title()}")
        plt.title(f"Global {m.replace('_', ' ').title()} Comparison Across Models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"global_{m}_comparison.png")
        plt.close()



def plot_training_time(models_metrics):
    for model_name, metrics in models_metrics.items():
        training_times = [m["training_time"] for m in metrics]

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(metrics) + 1), training_times, label="Training Time (seconds)", marker='o')
        plt.title(f"{model_name} - Training Time per Fold")
        plt.xlabel("Fold")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{model_name}_training_time.png")
        plt.close()
        
        
        
def plot_bar_with_error(models_metrics, metric_key, title, ylabel):
    names = []
    means = []
    stds = []

    for model, metrics in models_metrics.items():
        vals = [m[metric_key] for m in metrics]
        names.append(model)
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5, color='skyblue')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{metric_key}_comparison.png")
    plt.close()


def plot_metrics_boxplots(all_model_metrics):
    # Include train_time
    metric_names = ["accuracy", "f1", "roc_auc", "pr_auc", "training_time"]  # Corrected to use "training_time"

    for metric in metric_names:
        data = []
        labels = []
        for model_name, metrics in all_model_metrics.items():
            values = [m[metric] for m in metrics]  # Make sure to extract training time (elapsed_time) here
            data.append(values)
            labels.append(model_name)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
        plt.ylabel("Training Time (s)" if metric == "training_time" else metric.replace("_", " ").title())
        plt.title(f"{'Training Time' if metric == 'training_time' else metric.replace('_', ' ').title()} Comparison Across Models")
        plt.tight_layout()
        plt.savefig(f"boxplot_{metric}.png")
        plt.close()



def compare_models_summary(
    all_global_metrics, 
    save_csv="model_summary.csv", 
    save_heatmap="model_metrics_heatmap.png",
    save_roc_bar="roc_auc_comparison_bar.png",
    save_roc_pr_bar="roc_vs_pr_auc_barplot.png",
    save_training_time_bar="training_time_barplot.png"
):
    """
    Compare global metrics across models using ROC-AUC (primary) and PR-AUC (secondary).
    Improved version: Fixes heatmap highlight, separates training time, adds PR-ROC delta.
    """
    # Create DataFrame
    df = pd.DataFrame.from_dict(all_global_metrics, orient="index")

    # Add PR-ROC delta column
    df["pr_roc_delta"] = df["pr_auc_mean"] - df["roc_auc_mean"]

    # Sort by roc_auc_mean, then pr_auc_mean
    df_sorted = df.sort_values(by=["roc_auc_mean", "pr_auc_mean"], ascending=False)
    best_model = df_sorted.index[0]

    print(f"Best model based on ROC-AUC + PR-AUC fallback: {best_model}\n")
    print(df_sorted.loc[[best_model]])

    # Save CSV
    df_rounded = df.round(4)
    df_rounded.to_csv(save_csv)

    # -------------------- HEATMAP (Performance Metrics) --------------------
    mean_cols = [col for col in df.columns if "mean" in col and col != "training_time_mean"]
    heatmap_data = df_sorted[mean_cols]

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Score"})

    # Highlight best model row (now correctly aligned with sorted data)
    ax.add_patch(plt.Rectangle((0, 0), len(heatmap_data.columns), 1, color="red", alpha=0.3))

    plt.title("Model Performance Comparison (Mean Metrics)")
    plt.tight_layout()
    plt.savefig(save_heatmap)
    plt.close()

    # -------------------- ROC-AUC BAR PLOT --------------------
    plt.figure(figsize=(8, 5))
    df_sorted["roc_auc_mean"].plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Model Comparison: ROC-AUC")
    plt.ylabel("ROC-AUC Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_roc_bar)
    plt.close()

    # -------------------- ROC-AUC vs PR-AUC BAR PLOT --------------------
    df_sorted[["roc_auc_mean", "pr_auc_mean"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison: ROC-AUC vs PR-AUC")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_roc_pr_bar)
    plt.close()

    # -------------------- TRAINING TIME BAR PLOT --------------------
    plt.figure(figsize=(8, 5))
    df_sorted["training_time_mean"].plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Model Comparison: Training Time (Mean)")
    plt.ylabel("Time (seconds or units)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_training_time_bar)
    plt.close()

    return best_model



def save_metrics_csv(models_metrics):
    rows = []
    for model, metrics in models_metrics.items():
        for m in metrics:
            row = {"model": model, **m}
            rows.append(row)
    pd.DataFrame(rows).to_csv("model_comparison_cv.csv", index=False)


def main():
    
    setup_gpu()  # Setup GPU memory growth
    X, y = load_data_as_numpy()
    plot_class_distribution(y, title="Overall Dataset Class Distribution", save_path="overall_class_distribution.png")

    models_metrics = {}
    models_cms = {}
    all_global_metrics = {}
    
    for model_name in MODELS:
        metrics, cms, global_metrics = train_on_fold(X, y, model_name)
        models_metrics[model_name] = metrics
        models_cms[model_name] = cms
        all_global_metrics[model_name] = global_metrics

    save_metrics_csv(models_metrics)


    # Plot error bar charts for various metrics
    plot_bar_with_error(models_metrics, "accuracy", "Test Accuracy Comparison", "Accuracy")
    plot_bar_with_error(models_metrics, "f1", "F1 Score Comparison", "F1 Score")
    plot_bar_with_error(models_metrics, "roc_auc", "ROC AUC Comparison", "ROC AUC")
    plot_bar_with_error(models_metrics, "pr_auc", "PR AUC Comparison", "PR AUC")
    plot_bar_with_error(models_metrics, "training_time", "Training Time Comparison", "Time (s)")

    # Plot global metrics comparison
    plot_global_metrics_bar(all_global_metrics)
    
    # Plot ROC and PR curves comparison
    plot_roc_pr_auc_comparison(models_metrics)
    
    # Plot training time comparison
    plot_training_time(models_metrics)
    
    # Plot avg confusion matrices for each model
    plot_avg_confusion_matrices(models_cms)
    
    # Plot mean ROC and PR curves
    plot_mean_roc_pr_curves()
    
    # Call the new boxplot plotting function for all metrics
    plot_metrics_boxplots(models_metrics)
    
    # Compare models and get the best model
    best_model = compare_models_summary(all_global_metrics)
    # Output the best model
    print(f"The best model is: {best_model}")

if __name__ == "__main__":
    main()
