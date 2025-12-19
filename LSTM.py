
import json
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os

"""
LSTM model for predicting Hexameter lines
Claude translated a Wolfram alpha code into python
"""

# Step 1: Prepare data in the character-level format
def prepare_character_level_data(filename='hexameter_lines.json'):
    """
    Convert our data to character-level labels
    """
    # Try to find the file in parent directory if not in current directory
    if not os.path.exists(filename):
        parent_path = os.path.join('..', filename)
        if os.path.exists(parent_path):
            filename = parent_path
        else:
            raise FileNotFoundError(f"Could not find {filename} in current or parent directory")

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # We need to convert pattern to character labels
    # This is tricky because we don't have the actual syllable boundaries
    # So we'll use a simplified approach: just predict the pattern positions

    valid_data = [item for item in data if 'pattern_first_4' in item]

    print(f"Loaded {len(valid_data)} lines")

    # For now, let's use a hybrid approach:
    # Input: character sequence
    # Output: 4 labels (simpler than character-level)

    return valid_data


# Step 2: Build Bidirectional LSTM (like they did)
def build_lstm_model(vocab_size, max_length=80):
    """
    Sequence-to-sequence with Bidirectional LSTM
    (This is what the Wolfram paper used)
    """
    model = keras.Sequential([
        # Embedding layer (like EmbeddingLayer[12])
        keras.layers.Embedding(input_dim=vocab_size, output_dim=16,
                               input_length=max_length, mask_zero=True),

        # Bidirectional LSTM (like their NetBidirectionalOperator[LongShortTermMemoryLayer[32]])
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Dropout(0.3),

        keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False)),
        keras.layers.Dropout(0.3),

        # Dense layers
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.4),

        # Output: 4 positions
        keras.layers.Dense(4, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# Step 3: Prepare data
def prepare_lstm_data(data, max_length=80):
    """Prepare data for LSTM"""

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

    # Labels - ensure all patterns are exactly 4 characters
    y = []
    valid_indices = []
    for idx, item in enumerate(data):
        pattern = item['pattern_first_4']
        # Skip if pattern is not exactly 4 characters
        if len(pattern) != 4:
            continue
        encoded_pattern = [1 if c == 'D' else 0 for c in pattern]
        y.append(encoded_pattern)
        valid_indices.append(idx)

    # Filter X to match valid y entries
    X = [X[i] for i in valid_indices]
    X = np.array(X)
    y = np.array(y)

    print(f"Filtered to {len(X)} lines with valid 4-character patterns")

    return X, y, char_to_id


# Step 4: Train
def train_lstm_model(filename='hexameter_lines.json'):
    """Train the LSTM model"""

    print("=" * 50)
    print("LSTM MODEL (Wolfram-style)")
    print("=" * 50)

    data = prepare_character_level_data(filename)
    X, y, char_to_id = prepare_lstm_data(data)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Vocab size: {len(char_to_id)}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Build model
    vocab_size = X.max() + 1
    model = build_lstm_model(vocab_size, max_length=X.shape[1])

    print("\nModel:")
    model.summary()

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate exact match accuracy
    exact_matches = np.all(y_val == y_pred_binary, axis=1).sum()
    exact_accuracy = exact_matches / len(y_val) * 100

    print(f"\n{'=' * 50}")
    print(f"EXACT MATCH ACCURACY: {exact_accuracy:.2f}%")
    print(f"{'=' * 50}")

    # Per-foot accuracy
    print("\nPer-foot accuracy:")
    for i in range(4):
        acc = (y_val[:, i] == y_pred_binary[:, i]).sum() / len(y_val) * 100
        print(f"  Foot {i + 1}: {acc:.2f}%")

    # Save
    model.save('hexameter_lstm_model.keras')
    with open('char_to_id_lstm.json', 'w') as f:
        json.dump(char_to_id, f)

    print("\nModel saved!")

    return model, exact_accuracy


def load_lstm_model():
    """Load LSTM model and character mapping"""
    model = keras.models.load_model('hexameter_lstm_model.keras')
    with open('char_to_id_lstm.json', 'r') as f:
        char_to_id = json.load(f)
    return model, char_to_id


def predict_lstm(model, char_to_id, latin_line, max_length=80):
    """
    Predict the scansion pattern using LSTM
    Binary classification: D vs S (no T)
    """
    # Encode the line
    line = latin_line.lower()[:max_length]
    encoded = [char_to_id.get(c, 0) for c in line]
    encoded += [0] * (max_length - len(encoded))

    # Predict
    X = np.array([encoded])
    prediction = model.predict(X, verbose=0)[0]

    # Convert to pattern (binary: D if > 0.5, else S)
    pattern = ''.join(['D' if p > 0.5 else 'S' for p in prediction])

    return pattern


def get_scansion_from_lstm(latin_line):
    """Main function to call from other files"""
    model, char_to_id = load_lstm_model()
    return predict_lstm(model, char_to_id, latin_line)

if __name__ == "__main__":
    model, accuracy = train_lstm_model('hexameter_lines.json')

    print(f"\n{'=' * 50}")
    if accuracy > 40:
        print("🎉 LSTM beats CNN! Bidirectional networks work better for sequential patterns.")
    elif accuracy > 25:
        print("📈 Better than CNN, but still room for improvement")
        print("   Consider: More data (they used 9,700 lines)")
    else:
        print("🤔 Similar to CNN - the problem might be data quality")