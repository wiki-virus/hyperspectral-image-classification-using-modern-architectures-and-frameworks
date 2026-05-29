import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

NUM_BANDS = 204
NUM_STATS = 5

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
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

def get_few_shot_episode(X, y, n_way=4, k_shot=5, q_query=5):
    unique_classes = np.unique(y)
    selected_classes = unique_classes[:n_way]
    support_x, support_y, query_x, query_y = [], [], [], []
    for label_idx, c in enumerate(selected_classes):
        class_indices = np.where(y == c)[0]
        sampled_indices = np.random.choice(class_indices, k_shot + q_query, replace=False)
        support_x.append(X[sampled_indices[:k_shot]])
        support_y.append(np.full((k_shot,), label_idx))
        query_x.append(X[sampled_indices[k_shot:]])
        query_y.append(np.full((q_query,), label_idx))
    support_x = np.concatenate(support_x, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    query_x = np.concatenate(query_x, axis=0)
    query_y = np.concatenate(query_y, axis=0)
    shuffle_idx = np.random.permutation(len(query_y))
    query_x = query_x[shuffle_idx]
    query_y = query_y[shuffle_idx]
    return (tf.convert_to_tensor(support_x, dtype=tf.float32),
            tf.convert_to_tensor(support_y, dtype=tf.int32),
            tf.convert_to_tensor(query_x, dtype=tf.float32),
            tf.convert_to_tensor(query_y, dtype=tf.int32))

class StableMambaBlock(layers.Layer):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

    def build(self, input_shape):
        self.in_proj = layers.Dense(self.d_inner * 2, use_bias=False)
        self.conv1d = layers.Conv1D(
            filters=self.d_inner, kernel_size=self.d_conv, padding='same', groups=self.d_inner, activation='swish'
        )
        self.x_proj = layers.Dense(self.d_state * 2 + self.d_inner, use_bias=False)
        self.dt_proj = layers.Dense(self.d_inner, activation='softplus')
        
        A_init_val = -np.log(np.arange(1, self.d_state + 1, dtype=np.float32))
        A_init_val = np.tile(A_init_val[np.newaxis, :], (self.d_inner, 1))
        A_init = tf.keras.initializers.Constant(A_init_val)
        self.A = self.add_weight(
            name="A",
            shape=(self.d_inner, self.d_state),
            initializer=A_init,
            trainable=True
        )
        
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
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
        A_expanded = tf.expand_dims(self.A, axis=0)
        A_expanded = tf.expand_dims(A_expanded, axis=0)
        
        A_bar = tf.exp(delta_expanded * A_expanded)
        
        B_expanded = tf.expand_dims(B, axis=2)
        B_bar = delta_expanded * B_expanded
        
        x_branch_expanded = tf.expand_dims(x_branch, axis=-1)
        inputs_ssm = B_bar * x_branch_expanded
        
        inputs_ssm = tf.transpose(inputs_ssm, perm=[1, 0, 2, 3])
        A_bar = tf.transpose(A_bar, perm=[1, 0, 2, 3])
        
        elems = (A_bar, inputs_ssm)
        batch_size = tf.shape(x)[0]
        h_0 = tf.zeros((batch_size, self.d_inner, self.d_state), dtype=tf.float32)
        
        def scan_fn(h_prev, elem):
            a_bar_t, ssm_in_t = elem
            return a_bar_t * h_prev + ssm_in_t
        
        h_seq = tf.scan(scan_fn, elems, initializer=h_0)
        h_seq = tf.transpose(h_seq, perm=[1, 0, 2, 3])
        
        C_expanded = tf.expand_dims(C, axis=2)
        y = tf.reduce_sum(h_seq * C_expanded, axis=-1)
        
        D_expanded = tf.reshape(self.D, (1, 1, self.d_inner))
        y = y + D_expanded * x_branch
        
        gated_output = y * tf.keras.activations.swish(res_branch)
        return self.out_proj(gated_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand
        })
        return config

def build_encoder(arch_type, embedding_dim=32, l2_reg=1e-4):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    reg = tf.keras.regularizers.l2(l2_reg)
    x = layers.Conv1D(64, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LayerNormalization()(x)
    
    if arch_type == 'cnn':
        x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LayerNormalization()(x)
    elif arch_type == 'cnn_attention':
        attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(128, kernel_size=5, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LayerNormalization()(x)
    elif arch_type == 'cnn_mamba':
        mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
        x = layers.Add()([x, mamba_out])
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
    elif arch_type == 'cnn_attention_mamba':
        attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.05)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)
        mamba_out = StableMambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)(x)
        x = layers.Add()([x, mam_out := mamba_out])
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, activation='leaky_relu', padding='same', kernel_regularizer=reg)(x)
        x = layers.MaxPooling1D(2)(x)
        
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=reg)(x)
    raw_embeddings = layers.Dense(embedding_dim, activation=None)(x)
    embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=embeddings, name=f"Encoder_{arch_type}")

class MAMLClassifier(tf.keras.Model):
    def __init__(self, encoder, n_way=4):
        super(MAMLClassifier, self).__init__()
        self.encoder = encoder
        self.classifier_head = layers.Dense(n_way, activation=None)
    def call(self, x, training=True):
        embeddings = self.encoder(x, training=training)
        logits = self.classifier_head(embeddings)
        return logits

@tf.function
def train_step_fomaml(model, optimizer, support_x, support_y, query_x, query_y, alpha=0.01):
    original_weights = [tf.identity(v) for v in model.trainable_variables]
    with tf.GradientTape() as inner_tape:
        support_preds = model(support_x, training=True)
        loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            support_y, support_preds, from_logits=True
        ))
    grads_support = inner_tape.gradient(loss_support, model.trainable_variables)
    for var, grad in zip(model.trainable_variables, grads_support):
        if grad is not None:
            var.assign(var - alpha * grad)
            
    with tf.GradientTape() as outer_tape:
        query_preds = model(query_x, training=True)
        loss_query = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, query_preds, from_logits=True
        ))
    grads_outer = outer_tape.gradient(loss_query, model.trainable_variables)
    
    for var, orig_val in zip(model.trainable_variables, original_weights):
        var.assign(orig_val)
        
    trainable_vars, applied_grads = [], []
    for var, grad in zip(model.trainable_variables, grads_outer):
        if grad is not None:
            trainable_vars.append(var)
            applied_grads.append(grad)
    optimizer.apply_gradients(zip(applied_grads, trainable_vars))
    preds = tf.argmax(query_preds, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss_query, accuracy

architectures = ['cnn', 'cnn_attention', 'cnn_mamba', 'cnn_attention_mamba']
results = {}

for arch in architectures:
    print(f"\n==================== TRAINING {arch.upper()} FOMAML ====================")
    encoder = build_encoder(arch)
    maml_net = MAMLClassifier(encoder, n_way=4)
    optimizer = tf.keras.optimizers.Adam(2e-4) # Tuning optimizer to 2e-4
    
    loss_history = []
    acc_history = []
    
    # Train for 2000 episodes
    for ep in range(1, 2001):
        sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
        loss, acc = train_step_fomaml(maml_net, optimizer, sx, sy, qx, qy, alpha=0.01)
        loss_history.append(loss.numpy())
        acc_history.append(acc.numpy())
        if ep % 500 == 0:
            print(f"  Episode {ep:4d} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
            
    # Save weights
    encoder.save(f"best_maml_{arch}_encoder.keras")
    maml_net.save_weights(f"best_maml_{arch}_weights.h5")
    
    # Evaluate over 100 trials with test-time adaptation
    trial_accs = []
    for i in range(100):
        sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
        
        # Save weights before adaptation
        original_weights = [tf.identity(v) for v in maml_net.trainable_variables]
        
        # Adaptation steps (fine-tuning on support set)
        for _ in range(5):
            with tf.GradientTape() as tape:
                preds_support = maml_net(sx, training=False)
                loss_support = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    sy, preds_support, from_logits=True
                ))
            grads = tape.gradient(loss_support, maml_net.trainable_variables)
            for var, grad in zip(maml_net.trainable_variables, grads):
                if grad is not None:
                    var.assign(var - 0.01 * grad)
                    
        logits = maml_net(qx, training=False)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
        trial_accs.append(acc * 100.0)
        
        # Restore weights to pre-adapt state
        for var, orig in zip(maml_net.trainable_variables, original_weights):
            var.assign(orig)
            
    mean_acc = np.mean(trial_accs)
    max_acc = np.max(trial_accs)
    min_acc = np.min(trial_accs)
    
    print(f"==> {arch.upper()} FOMAML 5-Shot Holdout Test Accuracy over 100 trials:")
    print(f"  Mean Accuracy: {mean_acc:.2f}%")
    print(f"  Max Accuracy:  {max_acc:.2f}%")
    print(f"  Min Accuracy:  {min_acc:.2f}%")
    
    results[arch] = {
        'loss': loss_history,
        'accuracy': acc_history,
        'trial_accs': trial_accs,
        'mean_acc': mean_acc,
        'max_acc': max_acc,
        'min_acc': min_acc
    }

# Save results dictionary as npz file
np.savez("maml_results.npz", **results)
print("\nAll FOMAML architectures trained and saved!")
