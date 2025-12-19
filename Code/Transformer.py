import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_data(filename='hexameter_labeled.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    valid_data = [item for item in data if 'pattern_first_4' in item and len(item['pattern_first_4']) == 4]
    print(f"Loaded {len(valid_data)} valid labeled lines")
    return valid_data


def prepare_data(data, max_length=80):
    all_text = ''.join([item['line'] for item in data])
    chars = sorted(set(all_text.lower()))
    char_to_id = {char: idx + 1 for idx, char in enumerate(chars)}
    char_to_id['<PAD>'] = 0

    X = []
    for item in data:
        line = item['line'].lower()[:max_length]
        encoded = [char_to_id.get(c, 0) for c in line]
        encoded += [0] * (max_length - len(encoded))
        X.append(encoded)

    X = np.array(X)

    y = []
    for item in data:
        pattern = item['pattern_first_4']
        encoded_pattern = [1 if c == 'D' else 0 for c in pattern]
        y.append(encoded_pattern)

    y = np.array(y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Vocabulary size: {len(char_to_id)}")

    return X, y, char_to_id


# Fixed custom layers with get_config()
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=rate
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dropout(rate),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


def build_enhanced_transformer(vocab_size, max_length=80):
    """
    Enhanced Transformer for 17K+ lines
    """
    inputs = keras.layers.Input(shape=(max_length,))

    # Embedding
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim=64)
    x = embedding_layer(inputs)

    # 4 Transformer blocks
    x = TransformerBlock(embed_dim=64, num_heads=8, ff_dim=256, rate=0.15)(x)
    x = TransformerBlock(embed_dim=64, num_heads=8, ff_dim=256, rate=0.15)(x)
    x = TransformerBlock(embed_dim=64, num_heads=8, ff_dim=256, rate=0.15)(x)
    x = TransformerBlock(embed_dim=64, num_heads=8, ff_dim=256, rate=0.15)(x)

    # Pooling
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    # Output
    outputs = keras.layers.Dense(4, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_enhanced_transformer(filename='hexameter_labeled.json'):
    print("=" * 70)
    print("ENHANCED TRANSFORMER - Built for Hard Lines")
    print("=" * 70)

    data = load_data(filename)
    X, y, char_to_id = prepare_data(data)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")

    vocab_size = X.max() + 1
    model = build_enhanced_transformer(vocab_size, max_length=X.shape[1])

    print("\nModel Architecture:")
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=35,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=0.000001,
        verbose=1
    )

    # Save best weights (not full model to avoid serialization issues)
    checkpoint = keras.callbacks.ModelCheckpoint(
        '../Misc/best_transformer_weights.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,  # Only save weights
        mode='max',
        verbose=1
    )

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # Load best weights
    print("\nLoading best weights...")
    model.load_weights('best_transformer_weights.weights.h5')

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    y_pred = model.predict(X_val, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Per-foot accuracy
    print("\nPer-foot accuracy:")
    for i in range(4):
        acc = accuracy_score(y_val[:, i], y_pred_binary[:, i])
        print(f"  Foot {i + 1}: {acc * 100:.2f}%")

    # Exact match
    exact_matches = np.all(y_val == y_pred_binary, axis=1).sum()
    exact_accuracy = exact_matches / len(y_val)

    print(f"\n{'=' * 70}")
    print(f"EXACT MATCH ACCURACY: {exact_accuracy * 100:.2f}%")
    print(f"{'=' * 70}")

    # Show examples
    print("\nExample predictions (first 30):")
    correct_count = 0
    for i in range(min(30, len(y_val))):
        true_pattern = ''.join(['D' if x == 1 else 'S' for x in y_val[i]])
        pred_pattern = ''.join(['D' if x == 1 else 'S' for x in y_pred_binary[i]])
        match = "✓" if true_pattern == pred_pattern else "✗"
        if match == "✓":
            correct_count += 1

        # Show confidence for errors
        if match == "✗":
            confidences = [f"{p:.2f}" for p in y_pred[i]]
            print(f"  {match} True: {true_pattern} | Pred: {pred_pattern} | Conf: {confidences}")
        else:
            print(f"  {match} True: {true_pattern} | Pred: {pred_pattern}")

    print(f"\nIn sample: {correct_count}/30 = {correct_count / 30 * 100:.0f}%")

    # Compare
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"Original LSTM (3,360 lines):  95.0%")
    print(f"Transformer (17,381 lines):   {exact_accuracy * 100:.1f}%")
    print(f"{'=' * 70}")

    if exact_accuracy > 0.975:
        print("\n🎉 INCREDIBLE! 97.5%+ - Near perfect!")
    elif exact_accuracy > 0.96:
        print("\n🚀 EXCELLENT! 96%+ accuracy!")
    elif exact_accuracy > 0.95:
        print("\n✅ Competitive with LSTM!")
    else:
        print("\n📊 Good but LSTM might be better")

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Transformer Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Transformer Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('transformer_training.png', dpi=150)
    print("\nTraining curves saved!")

    # Save model and vocab
    with open('../Misc/char_to_id_transformer.json', 'w') as f:
        json.dump(char_to_id, f)

    # Save model architecture separately
    model_json = model.to_json()
    with open('../Misc/transformer_architecture.json', 'w') as f:
        f.write(model_json)

    print("\nModel weights and architecture saved!")
    print("To load: create model, then model.load_weights('best_transformer_weights.weights.h5')")

    return model, exact_accuracy, history


if __name__ == "__main__":
    model, accuracy, history = train_enhanced_transformer('../Misc/hexameter_lines.json')