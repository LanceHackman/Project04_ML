import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(filename='hexameter_lines.json'):
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

# Custom Attention Layer for LSTM
class AttentionLayer(keras.layers.Layer):
    """
    Attention mechanism to focus on important parts of the sequence
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)
    
    def call(self, features):
        # features shape: (batch_size, time_steps, hidden_size)
        
        # Compute attention scores
        score = tf.nn.tanh(self.W(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # Apply attention weights
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

def build_lstm(vocab_size, max_length=80):
    """
    LSTM with:
    - Deeper layers (3 Bi-LSTM layers)
    - Larger hidden dimensions (128 → 96 → 64)
    - Attention mechanism
    - Bigger embeddings (64-dim)
    """
    # Input
    inputs = keras.layers.Input(shape=(max_length,))
    
    # Larger embedding layer
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,  # Increased from 32
        mask_zero=True
    )(inputs)
    
    # First Bidirectional LSTM - largest
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Second Bidirectional LSTM - medium
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Third Bidirectional LSTM - smaller
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    
    # Attention mechanism - focuses on important parts
    attention_layer = AttentionLayer(128)
    context_vector, attention_weights = attention_layer(x)
    
    # Dense layers with more capacity
    x = keras.layers.Dense(256, activation='relu')(context_vector)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = keras.layers.Dense(4, activation='sigmoid')(x)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Use a learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.96
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lstm(filename='hexameter_lines.json'):
    print("="*70)
    print("LSTM - BEAST MODE (17,000+ lines)")
    print("="*70)
    
    data = load_data(filename)
    X, y, char_to_id = prepare_data(data)
    
    # With 17K lines, we can use a larger validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42  # 10% = ~1700 validation samples
    )
    
    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")
    
    vocab_size = X.max() + 1
    model = build_lstm(vocab_size, max_length=X.shape[1])
    
    print("\nModel Architecture:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Advanced callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Less patience since we have more data
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.000001,
        verbose=1
    )
    
    # Save best model
    checkpoint = keras.callbacks.ModelCheckpoint(
        '../Models/hexameter_lstm_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # TensorBoard logging (optional but useful)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
    
    # Train with larger batch size (more stable with more data)
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,  # Larger batch size with more data
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Load best model
    print("\nLoading best model...")
    model = keras.models.load_model('../Models/hexameter_lstm_model.keras', custom_objects={'AttentionLayer': AttentionLayer})
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    y_pred = model.predict(X_val, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Per-foot accuracy
    print("\nPer-foot accuracy:")
    for i in range(4):
        acc = accuracy_score(y_val[:, i], y_pred_binary[:, i])
        print(f"  Foot {i+1}: {acc*100:.2f}%")
    
    # Exact match
    exact_matches = np.all(y_val == y_pred_binary, axis=1).sum()
    exact_accuracy = exact_matches / len(y_val)
    
    print(f"\n{'='*70}")
    print(f"EXACT MATCH ACCURACY: {exact_accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    # Show confidence on predictions
    print("\nExample predictions with confidence:")
    for i in range(min(20, len(y_val))):
        true_pattern = ''.join(['D' if x == 1 else 'S' for x in y_val[i]])
        pred_pattern = ''.join(['D' if x == 1 else 'S' for x in y_pred_binary[i]])
        
        # Calculate average confidence (distance from 0.5)
        confidences = np.abs(y_pred[i] - 0.5) * 2  # Scale to 0-1
        avg_confidence = np.mean(confidences)
        
        match = "✓" if true_pattern == pred_pattern else "✗"
        print(f"  {match} True: {true_pattern} | Pred: {pred_pattern} | Conf: {avg_confidence:.2f}")
    
    # Compare to previous
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"Original LSTM (3,360 lines):  95.0%")
    print(f" LSTM (17,000 lines): {exact_accuracy*100:.1f}%")
    print(f"Improvement: +{(exact_accuracy - 0.95)*100:.1f}%")
    print(f"{'='*70}")
    
    if exact_accuracy > 0.975:
        print("\n🎉 INCREDIBLE! 97.5%+ - Near perfect!")
        print("   This should handle rating 80+ lines!")
    elif exact_accuracy > 0.96:
        print("\n🚀 EXCELLENT! 96%+ accuracy!")
        print("   Should easily handle rating 70+ lines")
    elif exact_accuracy > 0.95:
        print("\n✅ Good improvement over baseline!")
    else:
        print("\n🤔 Similar to baseline - may need different approach")
    
    # Plot training curves
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Plot per-foot accuracy
    foot_accs = [accuracy_score(y_val[:, i], y_pred_binary[:, i]) for i in range(4)]
    plt.bar(['Foot 1', 'Foot 2', 'Foot 3', 'Foot 4'], foot_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Foot Accuracy', fontsize=14, fontweight='bold')
    plt.ylim([0.9, 1.0])
    for i, v in enumerate(foot_accs):
        plt.text(i, v + 0.005, f'{v*100:.1f}%', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('lstm_training.png', dpi=150, bbox_inches='tight')
    print("\nTraining curves saved to lstm_training.png")
    
    # Save final model
    model.save('hexameter_lstm_final.keras')
    with open('../Misc/char_to_id_lstm.json', 'w') as f:
        json.dump(char_to_id, f)
    
    print("\nFinal model saved!")
    
    return model, exact_accuracy, history


if __name__ == "__main__":
    model, accuracy, history = train_lstm('../Misc/hexameter_lines.json')
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Test on rating 70+ lines in live game")
    print("2. If accuracy < 97%, try the Transformer")
    print("3. If still struggling, try ensemble (LSTM + Transformer)")
    print("="*70)