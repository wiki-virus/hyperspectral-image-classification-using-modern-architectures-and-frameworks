import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Allow TF to use full GPU VRAM (up to 3800MB of 4GB, leaving headroom for driver)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=3800)]
        )
        print(f"GPU configured: {len(gpus)} GPU(s), memory limit set to 3800MB")
    except Exception as e:
        print("GPU config error:", e)

NUM_BANDS = 204
NUM_STATS = 5
EMBED_DIM = 64   # Increased from 32 for richer embeddings
N_WAY = 4
K_SHOT = 5
Q_QUERY = 10    # More query samples per episode for better gradient signal

loaded_data = np.load("hyperleaf_processed.npz")
X_data = loaded_data['X']
y_data = loaded_data['y']

X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_min = X_train_raw.min()
X_max = X_train_raw.max()
X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_test  = (X_test_raw  - X_min) / (X_max - X_min + 1e-8)

def get_few_shot_episode(X, y, n_way=4, k_shot=5, q_query=10):
    unique_classes = np.unique(y)
    # Randomly select n_way classes each episode for diversity
    selected_classes = np.random.choice(unique_classes, size=n_way, replace=False)
    support_x, support_y, query_x, query_y = [], [], [], []
    for label_idx, c in enumerate(selected_classes):
        class_indices = np.where(y == c)[0]
        needed = k_shot + q_query
        replace = len(class_indices) < needed
        sampled_indices = np.random.choice(class_indices, needed, replace=replace)
        support_x.append(X[sampled_indices[:k_shot]])
        support_y.append(np.full((k_shot,), label_idx))
        query_x.append(X[sampled_indices[k_shot:]])
        query_y.append(np.full((q_query,), label_idx))
    support_x = np.concatenate(support_x, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    query_x   = np.concatenate(query_x, axis=0)
    query_y   = np.concatenate(query_y, axis=0)
    shuffle_idx = np.random.permutation(len(query_y))
    query_x = query_x[shuffle_idx]
    query_y = query_y[shuffle_idx]
    return (tf.convert_to_tensor(support_x, dtype=tf.float32),
            tf.convert_to_tensor(support_y, dtype=tf.int32),
            tf.convert_to_tensor(query_x, dtype=tf.float32),
            tf.convert_to_tensor(query_y, dtype=tf.int32))

INNER_STEPS = 1      # Inner loop gradient steps
INNER_LR    = 0.01   # Inner loop learning rate alpha

class MAMLClassifier(tf.keras.Model):
    def __init__(self, encoder, n_way=4):
        super().__init__()
        self.encoder = encoder
        self.head    = layers.Dense(n_way, activation=None,
                                    kernel_initializer='glorot_uniform')

    def call(self, x, training=True):
        return self.head(self.encoder(x, training=training))

def fomaml_train_step(model, optimizer, sx, sy, qx, qy,
                       inner_steps=INNER_STEPS, alpha=INNER_LR):
    # Save original weights
    original = [tf.identity(v) for v in model.trainable_variables]

    # Inner loop: multiple gradient steps on support set
    for _ in range(inner_steps):
        with tf.GradientTape() as inner_tape:
            preds_s = model(sx, training=True)
            loss_s  = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(sy, preds_s, from_logits=True)
            )
        grads_s = inner_tape.gradient(loss_s, model.trainable_variables)
        for var, g in zip(model.trainable_variables, grads_s):
            if g is not None:
                var.assign(var - alpha * g)

    # Outer loop: compute query loss on adapted weights
    with tf.GradientTape() as outer_tape:
        preds_q = model(qx, training=True)
        loss_q  = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(qy, preds_q, from_logits=True)
        )
    grads_q = outer_tape.gradient(loss_q, model.trainable_variables)

    # Restore original weights before applying meta-update
    for var, orig in zip(model.trainable_variables, original):
        var.assign(orig)

    pairs = [(g, v) for g, v in zip(grads_q, model.trainable_variables) if g is not None]
    optimizer.apply_gradients(pairs)

    acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(preds_q, axis=-1, output_type=tf.int32), qy), tf.float32
    ))
    return loss_q, acc

def train_maml_model(maml_net, episodes=3000, lr=1e-3):
    # Slightly higher outer LR for MAML
    lr_sched  = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=episodes)
    optimizer = tf.keras.optimizers.Adam(lr_sched)
    loss_hist, acc_hist = [], []
    for ep in range(1, episodes + 1):
        sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, N_WAY, K_SHOT, Q_QUERY)
        loss, acc = fomaml_train_step(optimizer=optimizer,
                                       model=maml_net,
                                       sx=sx, sy=sy, qx=qx, qy=qy)
        loss_hist.append(float(loss))
        acc_hist.append(float(acc))
        if ep % 300 == 0:
            print(f"  Ep {ep:4d}/{episodes} | Loss: {loss:.4f} | Recent Acc: {np.mean(acc_hist[-100:])*100:.1f}%")
    return loss_hist, acc_hist


tf.keras.backend.clear_session()

def build_cnn_encoder(l2_reg=1e-4):
    # No Dropout - removed to improve accuracy
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(128, kernel_size=7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM, activation=None)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs=inputs, outputs=out, name="CNN_Encoder_MAML_OPT")

cnn_encoder = build_cnn_encoder()
maml_cnn    = MAMLClassifier(cnn_encoder, n_way=N_WAY)
# Build model by running a dummy forward pass
_ = maml_cnn(tf.zeros((1, NUM_BANDS, NUM_STATS)), training=False)
print(f"CNN MAML total params: {maml_cnn.count_params():,}")
print("Training CNN MAML (optimized)...")
loss_cnn, acc_cnn = train_maml_model(maml_cnn, episodes=3000)
cnn_encoder.save("best_opt_maml_cnn_encoder.keras")
maml_cnn.save_weights("best_opt_maml_cnn_weights.weights.h5")
print(f"CNN Final avg acc: {np.mean(acc_cnn[-200:])*100:.2f}%")


tf.keras.backend.clear_session()

def build_cnn_attention_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(128, kernel_size=7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.0)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM, activation=None)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs=inputs, outputs=out, name="CNN_Attn_Encoder_MAML_OPT")

cnn_attention_encoder = build_cnn_attention_encoder()
maml_attention        = MAMLClassifier(cnn_attention_encoder, n_way=N_WAY)
_ = maml_attention(tf.zeros((1, NUM_BANDS, NUM_STATS)), training=False)
print(f"CNN+Attn MAML total params: {maml_attention.count_params():,}")
print("Training CNN+Attention MAML (optimized)...")
loss_attention, acc_attention = train_maml_model(maml_attention, episodes=3000)
cnn_attention_encoder.save("best_opt_maml_cnn_attention_encoder.keras")
maml_attention.save_weights("best_opt_maml_cnn_attention_weights.weights.h5")
print(f"CNN+Attn Final avg acc: {np.mean(acc_attention[-200:])*100:.2f}%")


class StableMambaBlock(layers.Layer):
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = int(expand * d_model)

    def build(self, input_shape):
        self.in_proj  = layers.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d   = layers.Conv1D(
            filters=self.d_inner, kernel_size=self.d_conv,
            padding='same', groups=self.d_inner, activation='swish'
        )
        self.x_proj   = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj  = layers.Dense(self.d_inner, activation='softplus')
        A_init_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_init_val = np.tile(A_init_val[np.newaxis, :], (self.d_inner, 1))
        self.A = self.add_weight(
            name="A", shape=(self.d_inner, self.d_state),
            initializer=tf.keras.initializers.Constant(A_init_val), trainable=True
        )
        self.D = self.add_weight(
            name="D", shape=(self.d_inner,),
            initializer=tf.keras.initializers.Ones(), trainable=True
        )
        self.out_proj = layers.Dense(self.d_model, use_bias=False)
        super().build(input_shape)

    def call(self, x):
        projected = self.in_proj(x)
        x_branch, res_branch = tf.split(projected, num_or_size_splits=2, axis=-1)
        x_branch = self.conv1d(x_branch)
        ssm_params = self.x_proj(x_branch)
        B, C, delta = tf.split(ssm_params, [self.d_state, self.d_state, self.d_inner], axis=-1)
        delta = self.dt_proj(delta)
        delta_expanded = tf.expand_dims(delta, axis=-1)
        A_expanded = tf.expand_dims(tf.expand_dims(self.A, 0), 0)
        A_bar = tf.exp(delta_expanded * A_expanded)
        B_bar = delta_expanded * tf.expand_dims(B, 2)
        inputs_ssm = B_bar * tf.expand_dims(x_branch, -1)
        inputs_ssm = tf.transpose(inputs_ssm, [1, 0, 2, 3])
        A_bar      = tf.transpose(A_bar,      [1, 0, 2, 3])
        h_0 = tf.zeros((tf.shape(x)[0], self.d_inner, self.d_state), dtype=tf.float32)
        def scan_fn(h, elem):
            a, s = elem
            return a * h + s
        h_seq = tf.scan(scan_fn, (A_bar, inputs_ssm), initializer=h_0)
        h_seq = tf.transpose(h_seq, [1, 0, 2, 3])
        y = tf.reduce_sum(h_seq * tf.expand_dims(C, 2), axis=-1)
        y = y + tf.reshape(self.D, (1, 1, self.d_inner)) * x_branch
        return self.out_proj(y * tf.keras.activations.swish(res_branch))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "d_state": self.d_state,
                    "d_conv": self.d_conv, "expand": self.expand})
        return cfg

tf.keras.backend.clear_session()

def build_cnn_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(128, kernel_size=7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    mamba = StableMambaBlock(d_model=256, d_state=32, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba])
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM, activation=None)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs=inputs, outputs=out, name="CNN_Mamba_Encoder_MAML_OPT")

cnn_mamba_encoder = build_cnn_mamba_encoder()
maml_mamba        = MAMLClassifier(cnn_mamba_encoder, n_way=N_WAY)
_ = maml_mamba(tf.zeros((1, NUM_BANDS, NUM_STATS)), training=False)
print(f"CNN+Mamba MAML total params: {maml_mamba.count_params():,}")
print("Training CNN+Mamba MAML (optimized)...")
loss_mamba, acc_mamba = train_maml_model(maml_mamba, episodes=3000)
cnn_mamba_encoder.save("best_opt_maml_cnn_mamba_encoder.keras")
maml_mamba.save_weights("best_opt_maml_cnn_mamba_weights.weights.h5")
print(f"CNN+Mamba Final avg acc: {np.mean(acc_mamba[-200:])*100:.2f}%")


tf.keras.backend.clear_session()

def build_cnn_attention_mamba_encoder(l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(128, kernel_size=7, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.0)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    mamba = StableMambaBlock(d_model=256, d_state=32, d_conv=4, expand=2)(x)
    x = layers.Add()([x, mamba])
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw = layers.Dense(EMBED_DIM, activation=None)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw)
    return models.Model(inputs=inputs, outputs=out, name="CNN_AttnMamba_Encoder_MAML_OPT")

cnn_attention_mamba_encoder = build_cnn_attention_mamba_encoder()
maml_attn_mamba             = MAMLClassifier(cnn_attention_mamba_encoder, n_way=N_WAY)
_ = maml_attn_mamba(tf.zeros((1, NUM_BANDS, NUM_STATS)), training=False)
print(f"CNN+Attn+Mamba MAML total params: {maml_attn_mamba.count_params():,}")
print("Training CNN+Attn+Mamba MAML (optimized)...")
loss_attn_mamba, acc_attn_mamba = train_maml_model(maml_attn_mamba, episodes=3000)
cnn_attention_mamba_encoder.save("best_opt_maml_cnn_attention_mamba_encoder.keras")
maml_attn_mamba.save_weights("best_opt_maml_cnn_attention_mamba_weights.weights.h5")
print(f"CNN+Attn+Mamba Final avg acc: {np.mean(acc_attn_mamba[-200:])*100:.2f}%")


# --- Testing Cell: 5-Shot Holdout Evaluation with Test-Time Adaptation ---
maml_models = {
    'CNN':            lambda: MAMLClassifier(build_cnn_encoder(), n_way=N_WAY),
    'CNN+Attention':  lambda: MAMLClassifier(build_cnn_attention_encoder(), n_way=N_WAY),
    'CNN+Mamba':      lambda: MAMLClassifier(build_cnn_mamba_encoder(), n_way=N_WAY),
    'CNN+Attn+Mamba': lambda: MAMLClassifier(build_cnn_attention_mamba_encoder(), n_way=N_WAY),
}
eval_results = {}

for name, builder in maml_models.items():
    tf.keras.backend.clear_session()
    net = builder()
    _ = net(tf.zeros((1, NUM_BANDS, NUM_STATS)), training=False) # dummy pass
    print(f"\nEvaluating {name} MAML (test-time adaptation, {INNER_STEPS} steps)...")
    key = name.lower().replace('+','_').replace(' ','_')
    try:
        net.load_weights(f"best_opt_maml_{key}_weights.weights.h5")
    except Exception:
        print(f"  (using current weights for {name})")

    trial_accs = []
    for i in range(100):
        sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, N_WAY, K_SHOT, 15)

        # Save weights before adaptation
        saved = [tf.identity(v) for v in net.trainable_variables]

        # Test-time inner adaptation
        for _ in range(INNER_STEPS):
            with tf.GradientTape() as tape:
                ps = net(sx, training=False)
                ls = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(sy, ps, from_logits=True)
                )
            gs = tape.gradient(ls, net.trainable_variables)
            for var, g in zip(net.trainable_variables, gs):
                if g is not None:
                    var.assign(var - INNER_LR * g)

        logits = net(qx, training=False)
        preds  = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc    = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
        trial_accs.append(acc * 100.0)

        # Restore pre-adaptation weights
        for var, orig in zip(net.trainable_variables, saved):
            var.assign(orig)

    eval_results[name] = trial_accs
    print(f"  Mean: {np.mean(trial_accs):.2f}%  Max: {np.max(trial_accs):.2f}%  Min: {np.min(trial_accs):.2f}%")


# --- Plotting Cell ---
histories = {
    'CNN':               (loss_cnn,       acc_cnn)       if 'loss_cnn'       in locals() else None,
    'CNN+Attention':     (loss_attention,  acc_attention)  if 'loss_attention'  in locals() else None,
    'CNN+Mamba':         (loss_mamba,      acc_mamba)      if 'loss_mamba'      in locals() else None,
    'CNN+Attn+Mamba':    (loss_attn_mamba, acc_attn_mamba) if 'loss_attn_mamba' in locals() else None,
}

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
for name, h in histories.items():
    if h:
        # Smooth with rolling mean for readability
        loss_smooth = np.convolve(h[0], np.ones(20)/20, mode='valid')
        plt.plot(loss_smooth, label=name)
plt.title('Loss vs Episodes (smoothed)')
plt.xlabel('Episode'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
for name, h in histories.items():
    if h:
        acc_smooth = np.convolve(h[1], np.ones(20)/20, mode='valid')
        plt.plot(acc_smooth * 100, label=name)
plt.title('Accuracy vs Episodes (smoothed)')
plt.xlabel('Episode'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig('fsl_maml_opt_plot.png')

if 'eval_results' in locals() and eval_results:
    plt.figure(figsize=(10, 6))
    data_to_plot = [eval_results[n] for n in eval_results]
    plt.boxplot(data_to_plot, tick_labels=list(eval_results.keys()))
    plt.title('5-Shot Holdout Accuracy Distribution (100 Trials)')
    plt.ylabel('Accuracy (%)'); plt.grid(True); plt.savefig('fsl_maml_opt_plot.png')


