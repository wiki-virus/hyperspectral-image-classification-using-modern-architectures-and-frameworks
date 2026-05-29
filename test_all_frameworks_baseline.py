import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

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

def build_backbone_encoder(activation='relu'):
    inputs = layers.Input(shape=(NUM_BANDS, NUM_STATS))
    x = layers.Conv1D(64, kernel_size=5, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=5, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=activation)(x)
    raw_embeddings = layers.Dense(32, activation=None)(x)
    embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(raw_embeddings)
    return models.Model(inputs=inputs, outputs=embeddings)

# ProtoNet
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    def call(self, support_x, support_y, query_x, n_way, k_shot):
        support_embeddings = self.encoder(support_x, training=True)
        query_embeddings = self.encoder(query_x, training=True)
        d_dim = tf.shape(support_embeddings)[-1]
        support_reshaped = tf.reshape(support_embeddings, (n_way, k_shot, d_dim))
        prototypes = tf.reduce_mean(support_reshaped, axis=1)
        query_expanded = tf.expand_dims(query_embeddings, axis=1)
        proto_expanded = tf.expand_dims(prototypes, axis=0)
        distances = tf.reduce_sum(tf.square(query_expanded - proto_expanded), axis=-1)
        return -distances

@tf.function(jit_compile=True)
def train_step_protonet(model, optimizer, support_x, support_y, query_x, query_y, n_way, k_shot):
    with tf.GradientTape() as tape:
        logits = model(support_x, support_y, query_x, n_way, k_shot)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, logits, from_logits=True
        ))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss, accuracy

# RelationNet
def build_relation_module(embedding_dim=32):
    inputs = layers.Input(shape=(4 * embedding_dim,))
    x = layers.LayerNormalization()(inputs)
    x = layers.Dense(64, activation=None)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(32, activation=None)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    outputs = layers.Dense(1, activation=None)(x)
    return models.Model(inputs=inputs, outputs=outputs, name="Relation_Module")

class RelationNetwork(tf.keras.Model):
    def __init__(self, encoder, relation_module):
        super(RelationNetwork, self).__init__()
        self.encoder = encoder
        self.relation_module = relation_module
    def call(self, support_x, support_y, query_x, n_way, k_shot):
        support_embeddings = self.encoder(support_x, training=True)
        query_embeddings = self.encoder(query_x, training=True)
        d_dim = tf.shape(support_embeddings)[-1]
        support_reshaped = tf.reshape(support_embeddings, (n_way, k_shot, d_dim))
        prototypes = tf.reduce_mean(support_reshaped, axis=1)
        num_queries = tf.shape(query_embeddings)[0]
        query_expanded = tf.expand_dims(query_embeddings, axis=1)
        query_tiled = tf.tile(query_expanded, [1, n_way, 1])
        proto_expanded = tf.expand_dims(prototypes, axis=0)
        proto_tiled = tf.tile(proto_expanded, [num_queries, 1, 1])
        diff = tf.abs(query_tiled - proto_tiled)
        prod = query_tiled * proto_tiled
        relation_input = tf.concat([query_tiled, proto_tiled, diff, prod], axis=-1)
        flat_shape = tf.shape(relation_input)
        flat_input = tf.reshape(relation_input, (flat_shape[0] * flat_shape[1], flat_shape[2]))
        relation_logits = self.relation_module(flat_input, training=True)
        relation_logits = tf.reshape(relation_logits, (flat_shape[0], flat_shape[1]))
        return relation_logits

@tf.function(jit_compile=True)
def train_step_relationnet(model, optimizer, support_x, support_y, query_x, query_y, n_way, k_shot):
    with tf.GradientTape() as tape:
        relation_logits = model(support_x, support_y, query_x, n_way, k_shot)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            query_y, relation_logits, from_logits=True
        ))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    preds = tf.argmax(relation_logits, axis=-1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
    return loss, accuracy

# FOMAML
class MAMLClassifier(tf.keras.Model):
    def __init__(self, encoder, n_way):
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

# Let's run a quick training of ProtoNet
print("--- Training ProtoNet ---")
encoder_proto = build_backbone_encoder(activation='relu')
proto_net = PrototypicalNetwork(encoder_proto)
optimizer_proto = tf.keras.optimizers.Adam(1e-3)
for ep in range(1, 1501):
    sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
    loss, acc = train_step_protonet(proto_net, optimizer_proto, sx, sy, qx, qy, 4, 5)
    if ep % 500 == 0:
        print(f"ProtoNet Episode {ep} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# Let's evaluate ProtoNet
accs = []
for i in range(100):
    sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
    logits = proto_net(sx, sy, qx, 4, 5)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
    accs.append(acc)
print(f"ProtoNet test accuracy: {np.mean(accs)*100:.2f}%")

# Let's run a quick training of RelationNet
print("--- Training RelationNet ---")
encoder_rel = build_backbone_encoder(activation='leaky_relu')
rel_module = build_relation_module()
relation_net = RelationNetwork(encoder_rel, rel_module)
optimizer_rel = tf.keras.optimizers.Adam(1e-3)
for ep in range(1, 1501):
    sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
    loss, acc = train_step_relationnet(relation_net, optimizer_rel, sx, sy, qx, qy, 4, 5)
    if ep % 500 == 0:
        print(f"RelationNet Episode {ep} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# Let's evaluate RelationNet
accs = []
for i in range(100):
    sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
    logits = relation_net(sx, sy, qx, 4, 5)
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, qy), tf.float32)).numpy()
    accs.append(acc)
print(f"RelationNet test accuracy: {np.mean(accs)*100:.2f}%")

# Let's run a quick training of FOMAML
print("--- Training FOMAML ---")
encoder_maml = build_backbone_encoder(activation='leaky_relu')
maml_net = MAMLClassifier(encoder_maml, 4)
optimizer_maml = tf.keras.optimizers.Adam(1e-3)
for ep in range(1, 1501):
    sx, sy, qx, qy = get_few_shot_episode(X_train, y_train, 4, 5, 5)
    loss, acc = train_step_fomaml(maml_net, optimizer_maml, sx, sy, qx, qy, alpha=0.01)
    if ep % 500 == 0:
        print(f"FOMAML Episode {ep} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# Let's evaluate FOMAML
accs = []
for i in range(100):
    sx, sy, qx, qy = get_few_shot_episode(X_test, y_test, 4, 5, 5)
    original_weights = [tf.identity(v) for v in maml_net.trainable_variables]
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
    accs.append(acc)
    for var, orig in zip(maml_net.trainable_variables, original_weights):
        var.assign(orig)
print(f"FOMAML test accuracy: {np.mean(accs)*100:.2f}%")
