import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── GPU: use all 4 GB ────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3800)]
        )
        print(f"GPU configured: {len(gpus)} GPU(s), memory limit set to 3800MB")
    except Exception as e:
        print("GPU config error:", e)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
NUM_BANDS = 204
NUM_STATS = 5
EMBED_DIM = 128   # larger embedding for richer feature space
N_WAY     = 4
K_SHOT    = 5
Q_QUERY   = 15    # more queries = better gradient signal
EPISODES  = 3000
LR        = 1e-3

# ── Data ─────────────────────────────────────────────────────────────────────
loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_min   = X_train_raw.min()
X_max   = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_test  = (X_test_raw  - X_min) / (X_max - X_min + 1e-8)


def get_episode_tensors(X, y, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    """Return (support_x, support_y, query_x, query_y) as float32/int32 tensors."""
    classes = np.random.choice(np.unique(y), size=n_way, replace=False)
    sx, sy, qx, qy = [], [], [], []
    for lbl, c in enumerate(classes):
        idx = np.where(y == c)[0]
        need = k_shot + q_query
        chosen = np.random.choice(idx, size=need, replace=len(idx) < need)
        sx.append(X[chosen[:k_shot]])
        sy.append(np.full(k_shot, lbl))
        qx.append(X[chosen[k_shot:]])
        qy.append(np.full(q_query, lbl))
    sx = np.concatenate(sx); sy = np.concatenate(sy)
    qx = np.concatenate(qx); qy = np.concatenate(qy)
    p  = np.random.permutation(len(qy))
    return (tf.constant(sx, dtype=tf.float32),
            tf.constant(sy, dtype=tf.int32),
            tf.constant(qx[p], dtype=tf.float32),
            tf.constant(qy[p], dtype=tf.int32))


# ── Prototypical loss (episodic metric learning) ─────────────────────────────
def proto_step(encoder, optimizer, sx, sy, qx, qy, n_way):
    with tf.GradientTape() as tape:
        # Embed support + query
        s_emb = encoder(sx, training=True)     # [N*K, D]
        q_emb = encoder(qx, training=True)     # [N*Q, D]

        # Compute class prototypes
        s_emb_r  = tf.reshape(s_emb, [n_way, K_SHOT, EMBED_DIM])
        protos   = tf.reduce_mean(s_emb_r, axis=1)   # [N, D]

        # Squared Euclidean distance
        diffs    = tf.expand_dims(q_emb, 1) - tf.expand_dims(protos, 0)  # [Q*N, N, D]
        dists    = tf.reduce_sum(diffs ** 2, axis=-1)   # [Q*N, N]

        # Negative distance as logits → cross-entropy
        logits   = -dists
        loss     = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(qy, logits, from_logits=True)
        )
        # L2 regularisation already in the model weights via kernel_regularizer
        reg_loss = tf.add_n(encoder.losses) if encoder.losses else 0.0
        total    = loss + reg_loss

    grads = tape.gradient(total, encoder.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)   # gradient clipping
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc   = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32))
    return loss, acc


def train_proto_model(encoder, name, episodes=EPISODES):
    lr_sched  = tf.keras.optimizers.schedules.CosineDecay(LR, decay_steps=episodes, alpha=1e-5)
    optimizer = tf.keras.optimizers.Adam(lr_sched)
    loss_hist, acc_hist = [], []
    for ep in range(1, episodes + 1):
        sx, sy, qx, qy = get_episode_tensors(X_train, y_train)
        # sy must be sorted for reshape trick - sort support by label
        sort_idx = tf.argsort(sy)
        sx = tf.gather(sx, sort_idx)
        sy = tf.gather(sy, sort_idx)

        loss, acc = proto_step(encoder, optimizer, sx, sy, qx, qy, N_WAY)
        loss_hist.append(float(loss))
        acc_hist.append(float(acc))
        if ep % 300 == 0:
            recent = np.mean(acc_hist[-100:]) * 100
            print(f"  Ep {ep:4d}/{episodes} | Loss: {loss:.4f} | Recent Acc: {recent:.1f}%")
    return loss_hist, acc_hist


# ══════════════════════════════════════════════════════════════════════════════
# Encoder 1: CNN
# ══════════════════════════════════════════════════════════════════════════════
tf.keras.backend.clear_session()

def build_cnn_encoder(l2_reg=1e-4):
    reg    = tf.keras.regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    x = layers.Conv1D(128, 7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs, out, name="CNN_Proto_MAML")

# CNN already trained (99.90%) - weights saved to best_maml_best_cnn_encoder.keras
# cnn_encoder = build_cnn_encoder()
# cnn_encoder(tf.zeros((1, NUM_BANDS, NUM_STATS)))
# print(f"CNN encoder params: {cnn_encoder.count_params():,}")
# print("Training CNN (Prototypical Meta-Learning)...")
# loss_cnn, acc_cnn = train_proto_model(cnn_encoder, "CNN")
# cnn_encoder.save("best_maml_best_cnn_encoder.keras")
# print(f"CNN Final avg acc: {np.mean(acc_cnn[-200:])*100:.2f}%")
loss_cnn = [0.2]; acc_cnn = [0.999]  # placeholder so plotting doesn't crash
print("CNN already done: 99.90% - skipping.")


# ══════════════════════════════════════════════════════════════════════════════
# Encoder 2: CNN + Attention
# ══════════════════════════════════════════════════════════════════════════════
tf.keras.backend.clear_session()

def build_cnn_attention_encoder(l2_reg=1e-4):
    reg    = tf.keras.regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    x = layers.Conv1D(128, 7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.0)(x, x)
    x    = layers.Add()([x, attn])
    x    = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs, out, name="CNN_Attn_Proto_MAML")

cnn_attention_encoder = build_cnn_attention_encoder()
cnn_attention_encoder(tf.zeros((1, NUM_BANDS, NUM_STATS)))
print(f"CNN+Attn encoder params: {cnn_attention_encoder.count_params():,}")
print("Training CNN+Attention (Prototypical Meta-Learning)...")
loss_attention, acc_attention = train_proto_model(cnn_attention_encoder, "CNN+Attn")
cnn_attention_encoder.save("best_maml_best_cnn_attention_encoder.keras")
print(f"CNN+Attn Final avg acc: {np.mean(acc_attention[-200:])*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Mamba Block (shared)
# ══════════════════════════════════════════════════════════════════════════════
class StableMambaBlock(layers.Layer):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.d_state = d_state
        self.d_conv  = d_conv;  self.expand  = expand
        self.d_inner = int(expand * d_model)

    def build(self, input_shape):
        self.in_proj  = layers.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d   = layers.Conv1D(self.d_inner, self.d_conv,
                                      padding='same', groups=self.d_inner, activation='swish')
        self.x_proj   = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj  = layers.Dense(self.d_inner, activation='softplus')
        A_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_val = np.tile(A_val[None, :], (self.d_inner, 1))
        self.A = self.add_weight("A", shape=(self.d_inner, self.d_state),
                                 initializer=tf.keras.initializers.Constant(A_val))
        self.D = self.add_weight("D", shape=(self.d_inner,),
                                 initializer='ones')
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        proj = self.in_proj(x)
        xb, rb = tf.split(proj, 2, axis=-1)
        xb  = self.conv1d(xb)
        sp  = self.x_proj(xb)
        B, C, delta = tf.split(sp, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)
        de    = tf.expand_dims(delta, -1)
        Ae    = tf.expand_dims(tf.expand_dims(self.A, 0), 0)
        A_bar = tf.exp(de * Ae)
        B_bar = de * tf.expand_dims(B, 2)
        u     = B_bar * tf.expand_dims(xb, -1)
        u     = tf.transpose(u, [1, 0, 2, 3])
        A_bar = tf.transpose(A_bar, [1, 0, 2, 3])
        h0    = tf.zeros((tf.shape(x)[0], self.d_inner, self.d_state))
        h_seq = tf.scan(lambda h, e: e[0] * h + e[1], (A_bar, u), initializer=h0)
        h_seq = tf.transpose(h_seq, [1, 0, 2, 3])
        y     = tf.reduce_sum(h_seq * tf.expand_dims(C, 2), axis=-1)
        y     = y + tf.reshape(self.D, (1, 1, self.d_inner)) * xb
        return self.out_proj(y * tf.keras.activations.swish(rb))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "d_state": self.d_state,
                    "d_conv": self.d_conv,   "expand":  self.expand})
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Encoder 3: CNN + Mamba
# ══════════════════════════════════════════════════════════════════════════════
tf.keras.backend.clear_session()

def build_cnn_mamba_encoder(l2_reg=1e-4):
    reg    = tf.keras.regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    x = layers.Conv1D(128, 7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    mb = StableMambaBlock(256, d_state=32, d_conv=4, expand=2)(x)
    x  = layers.Add()([x, mb]); x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs, out, name="CNN_Mamba_Proto_MAML")

cnn_mamba_encoder = build_cnn_mamba_encoder()
cnn_mamba_encoder(tf.zeros((1, NUM_BANDS, NUM_STATS)))
print(f"CNN+Mamba encoder params: {cnn_mamba_encoder.count_params():,}")
print("Training CNN+Mamba (Prototypical Meta-Learning)...")
loss_mamba, acc_mamba = train_proto_model(cnn_mamba_encoder, "CNN+Mamba")
cnn_mamba_encoder.save("best_maml_best_cnn_mamba_encoder.keras")
print(f"CNN+Mamba Final avg acc: {np.mean(acc_mamba[-200:])*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Encoder 4: CNN + Attention + Mamba
# ══════════════════════════════════════════════════════════════════════════════
tf.keras.backend.clear_session()

def build_cnn_attention_mamba_encoder(l2_reg=1e-4):
    reg    = tf.keras.regularizers.l2(l2_reg)
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    x = layers.Conv1D(128, 7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.0)(x, x)
    x    = layers.Add()([x, attn]); x = layers.LayerNormalization()(x)
    mb   = StableMambaBlock(256, d_state=32, d_conv=4, expand=2)(x)
    x    = layers.Add()([x, mb]);  x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs, out, name="CNN_AttnMamba_Proto_MAML")

cnn_attention_mamba_encoder = build_cnn_attention_mamba_encoder()
cnn_attention_mamba_encoder(tf.zeros((1, NUM_BANDS, NUM_STATS)))
print(f"CNN+Attn+Mamba encoder params: {cnn_attention_mamba_encoder.count_params():,}")
print("Training CNN+Attn+Mamba (Prototypical Meta-Learning)...")
loss_attn_mamba, acc_attn_mamba = train_proto_model(cnn_attention_mamba_encoder, "CNN+Attn+Mamba")
cnn_attention_mamba_encoder.save("best_maml_best_cnn_attention_mamba_encoder.keras")
print(f"CNN+Attn+Mamba Final avg acc: {np.mean(acc_attn_mamba[-200:])*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Holdout evaluation (5-shot, 100 trials)
# ══════════════════════════════════════════════════════════════════════════════
eval_encoders = {
    'CNN':            (build_cnn_encoder,               "best_maml_best_cnn_encoder.keras"),
    'CNN+Attention':  (build_cnn_attention_encoder,     "best_maml_best_cnn_attention_encoder.keras"),
    'CNN+Mamba':      (build_cnn_mamba_encoder,         "best_maml_best_cnn_mamba_encoder.keras"),
    'CNN+Attn+Mamba': (build_cnn_attention_mamba_encoder,"best_maml_best_cnn_attention_mamba_encoder.keras"),
}
eval_results = {}

for name, (builder, wpath) in eval_encoders.items():
    tf.keras.backend.clear_session()
    enc = builder()
    enc(tf.zeros((1, NUM_BANDS, NUM_STATS)))
    try:
        enc.load_weights(wpath)
    except Exception:
        print(f"  (weights not found for {name}, using random init)")
    print(f"\nEvaluating {name} (5-shot, 100 trials)...")
    trial_accs = []
    for _ in range(100):
        sx, sy, qx, qy = get_episode_tensors(X_test, y_test, k_shot=5, q_query=15)
        sort_idx = tf.argsort(sy)
        sx = tf.gather(sx, sort_idx); sy = tf.gather(sy, sort_idx)
        s_emb = enc(sx, training=False)
        q_emb = enc(qx, training=False)
        protos = tf.reduce_mean(tf.reshape(s_emb, [N_WAY, 5, EMBED_DIM]), axis=1)
        dists  = tf.reduce_sum((tf.expand_dims(q_emb, 1) - tf.expand_dims(protos, 0))**2, axis=-1)
        preds  = tf.argmax(-dists, axis=-1, output_type=tf.int32)
        acc    = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
        trial_accs.append(acc * 100.0)
    eval_results[name] = trial_accs
    print(f"  Mean: {np.mean(trial_accs):.2f}%  Max: {np.max(trial_accs):.2f}%  Min: {np.min(trial_accs):.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
histories = {
    'CNN':            (loss_cnn,        acc_cnn),
    'CNN+Attention':  (loss_attention,  acc_attention),
    'CNN+Mamba':      (loss_mamba,      acc_mamba),
    'CNN+Attn+Mamba': (loss_attn_mamba, acc_attn_mamba),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, (lh, ah) in histories.items():
    ls = np.convolve(lh, np.ones(20)/20, mode='valid')
    ac = np.convolve(ah, np.ones(20)/20, mode='valid')
    axes[0].plot(ls, label=name)
    axes[1].plot(ac * 100, label=name)
axes[0].set(title='Loss vs Episodes (smoothed)', xlabel='Episode', ylabel='Loss')
axes[0].legend(); axes[0].grid(True)
axes[1].set(title='Accuracy vs Episodes (smoothed)', xlabel='Episode', ylabel='Accuracy (%)')
axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig('fsl_maml_best_plot.png', dpi=150)
print("Saved fsl_maml_best_plot.png")

if eval_results:
    plt.figure(figsize=(10, 6))
    plt.boxplot([eval_results[n] for n in eval_results], tick_labels=list(eval_results.keys()))
    plt.title('5-Shot Holdout Accuracy Distribution (100 Trials)')
    plt.ylabel('Accuracy (%)'); plt.grid(True)
    plt.savefig('fsl_maml_best_eval_plot.png', dpi=150)
    print("Saved fsl_maml_best_eval_plot.png")
