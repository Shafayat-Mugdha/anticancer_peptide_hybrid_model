# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers, optimizers, Input, callbacks
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from imblearn.over_sampling import SMOTE

# # -------------------------
# # 1️⃣ Advanced Data Preprocessing
# # -------------------------
# print("🧬 Power Cleaning & Feature Engineering...")
# X_seq = np.load("sequences_padded_filtered.npy") 
# X_pssm = np.load("pssm_features.npy") 
# X_phys = np.load("physicochemical_filtered.npy") 
# y = np.load("labels_filtered.npy")

# # StandardScaler
# scaler = StandardScaler()
# X_phys_scaled = scaler.fit_transform(X_phys)

# # Random Forest ভিত্তিক ফিচার সিলেকশন (Top 250)
# print("🔍 Selecting top 250 features via Random Forest...")
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_phys_scaled, y)
# indices = np.argsort(rf.feature_importances_)[-250:]
# X_phys_selected = X_phys_scaled[:, indices]

# # -------------------------
# # 2️⃣ Custom Attention Layer
# # -------------------------
# class AttentionLayer(layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform")
#         super().build(input_shape)
#     def call(self, x):
#         e = tf.squeeze(tf.tanh(tf.matmul(x, self.W)), axis=-1)
#         a = tf.nn.softmax(e, axis=1)
#         return tf.reduce_sum(x * tf.expand_dims(a, -1), axis=1)

# # -------------------------
# # 3️⃣ The "Winner" Architecture
# # -------------------------
# def build_final_model():
#     # Sequence Branch
#     seq_in = Input(shape=(X_seq.shape[1],))
#     s = layers.Embedding(21, 64)(seq_in)
#     s = layers.Conv1D(64, 3, activation='swish')(s)
#     s = layers.Bidirectional(layers.GRU(64, return_sequences=True))(s)
#     s = AttentionLayer()(s)

#     # PSSM Branch
#     pssm_in = Input(shape=(X_pssm.shape[1], X_pssm.shape[2]))
#     p = layers.Conv1D(128, 3, activation='swish')(pssm_in)
#     p = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(p)
#     p = AttentionLayer()(p)

#     # Phys-Chem Branch
#     phys_in = Input(shape=(250,))
#     ph = layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-3))(phys_in)
#     ph = layers.BatchNormalization()(ph)
#     ph = layers.Dropout(0.4)(ph)

#     # Fusion
#     merged = layers.concatenate([s, p, ph])
#     x = layers.Dense(128, activation='swish')(merged)
#     x = layers.Dropout(0.5)(x)
#     out = layers.Dense(1, activation='sigmoid')(x)

#     model = models.Model(inputs=[seq_in, pssm_in, phys_in], outputs=out)
#     model.compile(optimizer=optimizers.Adam(learning_rate=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # -------------------------
# # 4️⃣ Training Loop with SMOTE & CV
# # -------------------------
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# fold_acc = []

# for f, (t_idx, v_idx) in enumerate(skf.split(X_seq, y)):
#     print(f"\n🚀 Training Fold {f+1} with Augmentation...")
    
#     # SMOTE প্রয়োগ (Training ডাটা বাড়ানোর জন্য)
#     sm = SMOTE(random_state=42)
#     # Flattening for SMOTE
#     X_train_combined = np.hstack([X_seq[t_idx], X_pssm[t_idx].reshape(len(t_idx), -1), X_phys_selected[t_idx]])
#     X_res, y_res = sm.fit_resample(X_train_combined, y[t_idx])
    
#     # Reshape back
#     X_s_res = X_res[:, :X_seq.shape[1]]
#     pssm_end = X_seq.shape[1] + (X_pssm.shape[1]*X_pssm.shape[2])
#     X_p_res = X_res[:, X_seq.shape[1]:pssm_end].reshape(-1, X_pssm.shape[1], X_pssm.shape[2])
#     X_ph_res = X_res[:, pssm_end:]

#     model = build_final_model()
#     stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    
#     model.fit(
#         [X_s_res, X_p_res, X_ph_res], y_res,
#         validation_data=([X_seq[v_idx], X_pssm[v_idx], X_phys_selected[v_idx]], y[v_idx]),
#         epochs=200, batch_size=32, callbacks=[stop], verbose=1
#     )

#     acc = model.evaluate([X_seq[v_idx], X_pssm[v_idx], X_phys_selected[v_idx]], y[v_idx], verbose=0)[1]
#     print(f"✅ Fold {f+1} Results: {acc*100:.2f}%")
#     fold_acc.append(acc)

# print(f"\n🏆 Final Predicted Performance: {np.mean(fold_acc)*100:.2f}%")





### for generating feature count ####
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers, optimizers, Input, callbacks
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # -------------------------
# # 1️⃣ Data Loading & Preprocessing
# # -------------------------
# print("🧬 Loading Data & Preparing Features...")
# try:
#     X_seq = np.load("sequences_padded_filtered.npy") 
#     X_pssm = np.load("pssm_features.npy") 
#     X_phys = np.load("physicochemical_filtered.npy") 
#     y = np.load("labels_filtered.npy")
# except FileNotFoundError:
#     print("❌ Error: .npy files not found! Please ensure data files are in the same folder.")
#     exit()

# # StandardScaler for Phys-Chem features
# scaler = StandardScaler()
# X_phys_scaled = scaler.fit_transform(X_phys)

# # RF-based Feature Selection (Top 250)
# print("🔍 Selecting top 250 features via Random Forest...")
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_phys_scaled, y)
# indices = np.argsort(rf.feature_importances_)[-250:]
# X_phys_selected = X_phys_scaled[:, indices]

# # -------------------------
# # 2️⃣ Custom Attention Layer
# # -------------------------
# @tf.keras.utils.register_keras_serializable()
# class AttentionLayer(layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform")
#         super().build(input_shape)
#     def call(self, x):
#         e = tf.squeeze(tf.tanh(tf.matmul(x, self.W)), axis=-1)
#         a = tf.nn.softmax(e, axis=1)
#         return tf.reduce_sum(x * tf.expand_dims(a, -1), axis=1)

# # -------------------------
# # 3️⃣ Ablation-Friendly Model Builder
# # -------------------------
# def build_model(mode="full"):
#     """
#     Modes: 'seq_only', 'seq_pssm', 'seq_phys', 'full'
#     """
#     inputs = []
#     branches = []

#     # Sequence Branch
#     if "seq" in mode or mode == "full":
#         seq_in = Input(shape=(X_seq.shape[1],), name="seq_in")
#         inputs.append(seq_in)
#         s = layers.Embedding(21, 64)(seq_in)
#         s = layers.Conv1D(64, 3, activation='swish')(s)
#         s = layers.Bidirectional(layers.GRU(64, return_sequences=True))(s)
#         s = AttentionLayer()(s)
#         branches.append(s)

#     # PSSM Branch
#     if "pssm" in mode or mode == "full":
#         pssm_in = Input(shape=(X_pssm.shape[1], X_pssm.shape[2]), name="pssm_in")
#         inputs.append(pssm_in)
#         p = layers.Conv1D(128, 3, activation='swish')(pssm_in)
#         p = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(p)
#         p = AttentionLayer()(p)
#         branches.append(p)

#     # Phys-Chem Branch
#     if "phys" in mode or mode == "full":
#         phys_in = Input(shape=(250,), name="phys_in")
#         inputs.append(phys_in)
#         ph = layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-3))(phys_in)
#         ph = layers.BatchNormalization()(ph)
#         ph = layers.Dropout(0.4)(ph)
#         branches.append(ph)

#     # Fusion Strategy
#     if len(branches) > 1:
#         merged = layers.concatenate(branches)
#     else:
#         merged = branches[0]

#     x = layers.Dense(128, activation='swish')(merged)
#     x = layers.Dropout(0.5)(x)
#     out = layers.Dense(1, activation='sigmoid')(x)

#     model = models.Model(inputs=inputs, outputs=out)
#     model.compile(optimizer=optimizers.Adam(learning_rate=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # -------------------------
# # 4️⃣ Automated Ablation Study Loop
# # -------------------------
# ablation_configs = {
#     "Sequence only": "seq_only",
#     "Sequence + PSSM": "seq_pssm",
#     "Sequence + Physicochemical": "seq_phys",
#     "Full Fusion Model (Proposed)": "full"
# }

# ablation_final_results = {}

# for label, mode in ablation_configs.items():
#     print(f"\n🧪 Starting Experiment: {label}")
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     fold_accuracies = []

#     for f, (t_idx, v_idx) in enumerate(skf.split(X_seq, y)):
#         # SMOTE Logic for Multi-modal alignment
#         sm = SMOTE(random_state=42)
#         X_train_combined = np.hstack([X_seq[t_idx], X_pssm[t_idx].reshape(len(t_idx), -1), X_phys_selected[t_idx]])
#         X_res, y_res = sm.fit_resample(X_train_combined, y[t_idx])

#         # Prepare Resampled Inputs for the Current Mode
#         X_train_final = []
#         X_val_final = []

#         if "seq" in mode or mode == "full":
#             X_train_final.append(X_res[:, :X_seq.shape[1]])
#             X_val_final.append(X_seq[v_idx])
        
#         if "pssm" in mode or mode == "full":
#             p_start = X_seq.shape[1]
#             p_end = p_start + (X_pssm.shape[1] * X_pssm.shape[2])
#             X_train_final.append(X_res[:, p_start:p_end].reshape(-1, X_pssm.shape[1], X_pssm.shape[2]))
#             X_val_final.append(X_pssm[v_idx])

#         if "phys" in mode or mode == "full":
#             p_end = X_seq.shape[1] + (X_pssm.shape[1] * X_pssm.shape[2])
#             X_train_final.append(X_res[:, p_end:])
#             X_val_final.append(X_phys_selected[v_idx])

#         # Build and Train
#         model = build_model(mode=mode)
#         early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
        
#         # Training (reduced epochs for faster ablation, use 100+ for final)
#         model.fit(X_train_final, y_res, validation_data=(X_val_final, y[v_idx]), 
#                   epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

#         acc = model.evaluate(X_val_final, y[v_idx], verbose=0)[1]
#         fold_accuracies.append(acc)
#         print(f"  - Fold {f+1} Accuracy: {acc*100:.2f}%")
    
#     avg_acc = np.mean(fold_accuracies) * 100
#     ablation_final_results[label] = avg_acc
#     print(f"✅ Average for {label}: {avg_acc:.2f}%")

# # -------------------------
# # 5️⃣ Final Table Report
# # -------------------------
# print("\n" + "="*40)
# print("📊 FINAL ABLATION STUDY RESULTS")
# print("="*40)
# df_results = pd.DataFrame(list(ablation_final_results.items()), columns=['Configuration', 'Accuracy (%)'])
# print(df_results.to_string(index=False))
# print("="*40)


### ROC Curve

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, Input, callbacks
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1️⃣ Data Loading & Preprocessing
# -------------------------
print("🧬 Loading Data & Preparing Features...")

X_seq = np.load("sequences_padded_filtered.npy") 
X_pssm = np.load("pssm_features.npy") 
X_phys = np.load("physicochemical_filtered.npy") 
y = np.load("labels_filtered.npy")

# StandardScaler
scaler = StandardScaler()
X_phys_scaled = scaler.fit_transform(X_phys)

# Random Forest feature selection
print("🔍 Selecting top 250 features via Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_phys_scaled, y)
indices = np.argsort(rf.feature_importances_)[-250:]
X_phys_selected = X_phys_scaled[:, indices]

# -------------------------
# 2️⃣ Attention Layer
# -------------------------
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform"
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.squeeze(tf.tanh(tf.matmul(x, self.W)), axis=-1)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * tf.expand_dims(a, -1), axis=1)

# -------------------------
# 3️⃣ Model Builder
# -------------------------
def build_model(mode="full"):
    inputs = []
    branches = []

    # Sequence
    if "seq" in mode or mode == "full":
        seq_in = Input(shape=(X_seq.shape[1],))
        inputs.append(seq_in)
        s = layers.Embedding(21, 64)(seq_in)
        s = layers.Conv1D(64, 3, activation='swish')(s)
        s = layers.Bidirectional(layers.GRU(64, return_sequences=True))(s)
        s = AttentionLayer()(s)
        branches.append(s)

    # PSSM
    if "pssm" in mode or mode == "full":
        pssm_in = Input(shape=(X_pssm.shape[1], X_pssm.shape[2]))
        inputs.append(pssm_in)
        p = layers.Conv1D(128, 3, activation='swish')(pssm_in)
        p = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(p)
        p = AttentionLayer()(p)
        branches.append(p)

    # Physicochemical
    if "phys" in mode or mode == "full":
        phys_in = Input(shape=(250,))
        inputs.append(phys_in)
        ph = layers.Dense(256, activation='swish',
                          kernel_regularizer=regularizers.l2(1e-3))(phys_in)
        ph = layers.BatchNormalization()(ph)
        ph = layers.Dropout(0.4)(ph)
        branches.append(ph)

    merged = layers.concatenate(branches) if len(branches) > 1 else branches[0]

    x = layers.Dense(128, activation='swish')(merged)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------
# 4️⃣ Ablation Study
# -------------------------
ablation_configs = {
    "Sequence only": "seq_only",
    "Sequence + PSSM": "seq_pssm",
    "Sequence + Physicochemical": "seq_phys",
    "Full Fusion Model": "full"
}

results = {}

for label, mode in ablation_configs.items():
    print("\n🧪 Running:", label)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc = []

    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for f, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):

        # -------------------------
        # SMOTE (train only)
        # -------------------------
        sm = SMOTE(random_state=42)

        X_train_combined = np.hstack([
            X_seq[train_idx],
            X_pssm[train_idx].reshape(len(train_idx), -1),
            X_phys_selected[train_idx]
        ])

        X_res, y_res = sm.fit_resample(X_train_combined, y[train_idx])

        X_train_final, X_val_final = [], []

        # Sequence
        if "seq" in mode or mode == "full":
            X_train_final.append(X_res[:, :X_seq.shape[1]])
            X_val_final.append(X_seq[val_idx])

        # PSSM
        if "pssm" in mode or mode == "full":
            s = X_seq.shape[1]
            e = s + (X_pssm.shape[1] * X_pssm.shape[2])

            X_train_final.append(
                X_res[:, s:e].reshape(-1, X_pssm.shape[1], X_pssm.shape[2])
            )
            X_val_final.append(X_pssm[val_idx])

        # Physico
        if "phys" in mode or mode == "full":
            e = X_seq.shape[1] + (X_pssm.shape[1] * X_pssm.shape[2])
            X_train_final.append(X_res[:, e:])
            X_val_final.append(X_phys_selected[val_idx])

        # Model
        model = build_model(mode)

        es = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        )

        model.fit(
            X_train_final, y_res,
            validation_data=(X_val_final, y[val_idx]),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[es]
        )

        # -------------------------
        # Prediction
        # -------------------------
        y_prob = model.predict(X_val_final).ravel()
        y_pred = (y_prob > 0.5).astype(int)

        acc = np.mean(y_pred == y[val_idx])
        fold_acc.append(acc)

        all_y_true.extend(y[val_idx])
        all_y_pred.extend(y_pred)
        all_y_score.extend(y_prob)

        print(f"Fold {f+1} Acc: {acc*100:.2f}%")

    avg_acc = np.mean(fold_acc)

    # -------------------------
    # Metrics
    # -------------------------
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_score = np.array(all_y_score)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    results[label] = avg_acc

    # -------------------------
    # Confusion Matrix Plot
    # -------------------------
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()

    # -------------------------
    # ROC Curve
    # -------------------------
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.title(f"ROC Curve - {label}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

# -------------------------
# 5️⃣ Final Table
# -------------------------
print("\n============================")
print("📊 FINAL RESULTS")
print("============================")

df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

print(df)