# ================================
# ACP Multimodal Model + Diagram
# ================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, Input
from tensorflow.keras.utils import plot_model

# -------------------------
# Load Data (only for shape reference)
# -------------------------
X_seq = np.load("sequences_padded_filtered.npy")
X_pssm = np.load("pssm_features.npy")
X_phys = np.load("physicochemical_filtered.npy")

# -------------------------
# Custom Attention Layer
# -------------------------
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
# Build Full Fusion Model
# -------------------------
def build_model():

    # ===== Sequence Branch =====
    seq_in = Input(shape=(X_seq.shape[1],), name="Sequence_Input")

    s = layers.Embedding(21, 64, name="Seq_Embedding")(seq_in)
    s = layers.Conv1D(64, 3, activation='swish', name="Seq_Conv1D")(s)
    s = layers.Bidirectional(
        layers.GRU(64, return_sequences=True),
        name="Seq_BiGRU"
    )(s)
    s = AttentionLayer(name="Seq_Attention")(s)

    # ===== PSSM Branch =====
    pssm_in = Input(
        shape=(X_pssm.shape[1], X_pssm.shape[2]),
        name="PSSM_Input"
    )

    p = layers.Conv1D(128, 3, activation='swish', name="PSSM_Conv1D")(pssm_in)
    p = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True),
        name="PSSM_BiLSTM"
    )(p)
    p = AttentionLayer(name="PSSM_Attention")(p)

    # ===== Physicochemical Branch =====
    phys_in = Input(shape=(250,), name="Physicochemical_Input")

    ph = layers.Dense(
        256,
        activation='swish',
        kernel_regularizer=regularizers.l2(1e-3),
        name="Phys_Dense"
    )(phys_in)

    ph = layers.BatchNormalization(name="Phys_BN")(ph)
    ph = layers.Dropout(0.4, name="Phys_Dropout")(ph)

    # ===== Feature Fusion =====
    merged = layers.concatenate(
        [s, p, ph],
        name="Feature_Fusion"
    )

    x = layers.Dense(128, activation='swish', name="Fusion_Dense")(merged)
    x = layers.Dropout(0.5, name="Fusion_Dropout")(x)

    out = layers.Dense(1, activation='sigmoid', name="Output")(x)

    model = models.Model(
        inputs=[seq_in, pssm_in, phys_in],
        outputs=out,
        name="ACP_Multimodal_Model"
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# -------------------------
# Build Model
# -------------------------
model = build_model()

# Show model summary
model.summary()

# -------------------------
# Save Architecture Diagram
# -------------------------
plot_model(
    model,
    to_file="ACP_model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=300,
    expand_nested=True
)

print("\n✅ Architecture diagram saved as: ACP_model_architecture.png")