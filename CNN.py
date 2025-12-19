import json
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

"""
CNN model for predicting Hexameter lines
Claude.ai assisted
"""
def load_data(filename='hexameter_lines.json'):
    """

    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_data = [item for item in data if 'pattern_first_4' in item and len(item['pattern_first_4']) == 4]
    print(f"Loaded {len(valid_data)} valid labeled lines")
    return valid_data


def analyze_data_distribution(data):
    """
    Check for class imbalance in FIRST 4 POSITIONS ONLY
    """
    print("\n" + "=" * 50)
    print("DATA DISTRIBUTION ANALYSIS (First 4 feet only)")
    print("=" * 50)

    for pos in range(4):
        d_count = sum(1 for item in data if item['pattern_first_4'][pos] == 'D')
        s_count = sum(1 for item in data if item['pattern_first_4'][pos] == 'S')
        total = len(data)
        print(
            f"Position {pos + 1}: D={d_count} ({d_count / total * 100:.1f}%), S={s_count} ({s_count / total * 100:.1f}%)")

    pattern_counts = Counter([item['pattern_first_4'] for item in data])
    print(f"\nTop 10 most common patterns:")
    for pattern, count in pattern_counts.most_common(10):
        print(f"  {pattern}: {count} ({count / len(data) * 100:.1f}%)")

    print(f"\nTotal unique patterns: {len(pattern_counts)}")


def prepare_dataset(data, max_length=80):
    """
    Encodes the first 4 positions
    """

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

    # ONLY first 4 positions!
    y = []
    for item in data:
        pattern = item['pattern_first_4']
        encoded_pattern = [1 if c == 'D' else 0 for c in pattern]
        y.append(encoded_pattern)

    y = np.array(y)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Vocabulary size: {len(char_to_id)}")

    return X, y, char_to_id


def build_model_with_weighted_loss(vocab_size, max_length=80):
    """
    Model with weighted outputs for each position
    """
    input_layer = keras.layers.Input(shape=(max_length,))

    # Embedding
    x = keras.layers.Embedding(input_dim=vocab_size, output_dim=48)(input_layer)

    # Convolutions
    x = keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalMaxPooling1D()(x)

    # Dense layers
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)

    # Separate output for each position (important for handling imbalance)
    foot1 = keras.layers.Dense(1, activation='sigmoid', name='foot1')(x)
    foot2 = keras.layers.Dense(1, activation='sigmoid', name='foot2')(x)
    foot3 = keras.layers.Dense(1, activation='sigmoid', name='foot3')(x)
    foot4 = keras.layers.Dense(1, activation='sigmoid', name='foot4')(x)

    # Concatenate
    outputs = keras.layers.Concatenate()([foot1, foot2, foot3, foot4])

    model = keras.Model(inputs=input_layer, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def calculate_sample_weights(y_train):
    """
    Calculate sample weights to handle class imbalance
    Give more weight to underrepresented classes
    """
    weights = np.ones(len(y_train))

    for pos in range(4):
        pos_values = y_train[:, pos]
        n_samples = len(pos_values)
        n_d = pos_values.sum()
        n_s = n_samples - n_d

        # Weight samples where this position is D more heavily if D is rare
        weight_d = n_samples / (2 * n_d) if n_d > 0 else 1.0
        weight_s = n_samples / (2 * n_s) if n_s > 0 else 1.0

        # Apply weights for this position
        for i in range(len(y_train)):
            if y_train[i, pos] == 1:  # Is D
                weights[i] *= weight_d
            else:  # Is S
                weights[i] *= weight_s

    # Normalize weights
    weights = weights / weights.mean()

    print(f"\nSample weights range: {weights.min():.2f} to {weights.max():.2f}")
    print(f"Mean weight: {weights.mean():.2f}")

    return weights


def train_model(X, y, epochs=100, batch_size=32):
    """
    Train with sample weighting
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Calculate sample weights
    sample_weights = calculate_sample_weights(y_train)

    vocab_size = X.max() + 1
    model = build_model_with_weighted_loss(vocab_size, max_length=X.shape[1])

    print("\nModel architecture:")
    model.summary()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        verbose=1,
        min_lr=0.00001
    )

    # Train with sample weights
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=sample_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history, X_val, y_val


def evaluate_model(model, X_val, y_val):
    """
    Detailed evaluation
    """
    y_pred = model.predict(X_val, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\nPer-foot accuracy:")
    for i in range(4):
        accuracy = accuracy_score(y_val[:, i], y_pred_binary[:, i])
        cm = confusion_matrix(y_val[:, i], y_pred_binary[:, i])

        print(f"\n  Foot {i + 1}: {accuracy * 100:.2f}%")
        print(f"    Confusion matrix:")
        print(f"    {cm}")

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            s_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
            d_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"    S precision: {s_precision * 100:.1f}%, D precision: {d_precision * 100:.1f}%")

    # Exact match
    exact_matches = np.all(y_val == y_pred_binary, axis=1).sum()
    exact_accuracy = exact_matches / len(y_val)
    print(f"\n{'=' * 50}")
    print(f"EXACT PATTERN MATCH: {exact_accuracy * 100:.2f}%")
    print(f"{'=' * 50}")

    # Show examples
    print("\nExample predictions:")
    correct_count = 0
    for i in range(min(20, len(y_val))):
        true_pattern = ''.join(['D' if x == 1 else 'S' for x in y_val[i]])
        pred_pattern = ''.join(['D' if x == 1 else 'S' for x in y_pred_binary[i]])
        match = "✓" if true_pattern == pred_pattern else "✗"
        if match == "✓":
            correct_count += 1
        print(f"  {match} True: {true_pattern} | Pred: {pred_pattern}")

    print(f"\nIn sample of 20: {correct_count} correct ({correct_count / 20 * 100:.0f}%)")

    return exact_accuracy

def save_model(model, char_to_id):
    """
    Save CNN model and character mapping
    """
    model.save('hexameter_cnn_model.keras')
    with open('char_to_id.json', 'w') as f:
        json.dump(char_to_id, f)
    print("\nModel saved!")


def load_cnn_model():
    """
    Load CNN model and character mapping
    """
    model = keras.models.load_model('hexameter_cnn_model.keras')
    with open('char_to_id.json', 'r') as f:
        char_to_id = json.load(f)
    return model, char_to_id


def predict_cnn(model, char_to_id, latin_line, max_length=80):
    """
    Predict scansion using CNN model
    """
    # Encode the line
    line = latin_line.lower()[:max_length]
    encoded = [char_to_id.get(c, 0) for c in line]
    encoded += [0] * (max_length - len(encoded))

    # Predict
    X = np.array([encoded])
    y_pred = model.predict(X, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=-1)[0]

    # Convert to pattern string
    class_to_char = {0: 'D', 1: 'S', 2: 'T'}
    pattern = ''.join([class_to_char[x] for x in y_pred_classes])

    return pattern


def get_scansion_from_cnn(latin_line):
    """
    Main function to call from other files
    """
    model, char_to_id = load_cnn_model()
    return predict_cnn(model, char_to_id, latin_line)

if __name__ == "__main__":
    print("=" * 50)
    print("HEXAMETER CNN TRAINING (FIRST 4 FEET ONLY)")
    print("=" * 50)

    data = load_data('hexameter_lines.json')
    analyze_data_distribution(data)

    X, y, char_to_id = prepare_dataset(data, max_length=80)

    print("\nStarting training...")
    model, history, X_val, y_val = train_model(X, y, epochs=100, batch_size=32)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    accuracy = evaluate_model(model, X_val, y_val)

    save_model(model, char_to_id)

    print("\n" + "=" * 50)
    print(f"TRAINING COMPLETE - {accuracy * 100:.1f}% exact match")
    print("=" * 50)

    if accuracy > 0.40:
        print("\n🎉 Good enough to test against the baseline!")
    elif accuracy > 0.25:
        print("\n📈 Getting better - consider collecting more diverse data")
    else:
        print("\n💡 More work needed - the problem might be harder than expected")