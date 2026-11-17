import tensorflow as tf
import keras
from keras._tf_keras.keras import initializers
from keras._tf_keras.keras.layers import Layer, Dense, LSTM, Bidirectional, Dropout, LayerNormalization
import random
import numpy as np

# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

@tf.keras.utils.register_keras_serializable()
class EnhancedFCM(Layer):
    def __init__(self, input_dim):
        super(EnhancedFCM, self).__init__()
        self.W_fcm = self.add_weight(
            shape=(input_dim, input_dim),
            initializer=initializers.GlorotNormal(seed=SEED),
            regularizer='l2',
            trainable=True
        )
        self.gate = Dense(input_dim, activation='sigmoid',
                          kernel_initializer=initializers.GlorotNormal(seed=SEED))
    
    def call(self, x):
        transformed = tf.tanh(tf.matmul(x, self.W_fcm))
        gated = self.gate(x) * transformed  
        return x + gated

        
@tf.keras.utils.register_keras_serializable()
class MHSA(Layer):
    def __init__(self, hidden_dim, num_heads):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.depth = hidden_dim // num_heads
        
        initializer = initializers.GlorotNormal(seed=SEED)
        self.W_q = Dense(hidden_dim, kernel_initializer=initializer)
        self.W_k = Dense(hidden_dim, kernel_initializer=initializer)
        self.W_v = Dense(hidden_dim, kernel_initializer=initializer)
        self.W_o = Dense(hidden_dim, kernel_initializer=initializer)
        self.attn_dropout = Dropout(0.1)
        self.ln = LayerNormalization()

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, h):
        batch_size = tf.shape(h)[0]
        Q = self.split_heads(self.W_q(h), batch_size)
        K = self.split_heads(self.W_k(h), batch_size)
        V = self.split_heads(self.W_v(h), batch_size)
        
        attn_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = tf.matmul(attn_weights, V)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat_attn = tf.reshape(attn_output, (batch_size, -1, self.num_heads * self.depth))
        return self.ln(self.W_o(concat_attn) + h)  # Residual connection

        
@tf.keras.utils.register_keras_serializable()
class CausalAttention(Layer):
    def __init__(self, hidden_dim):
        super(CausalAttention, self).__init__()
        self.W_q = Dense(hidden_dim)
        self.W_k = Dense(hidden_dim)
        self.W_v = Dense(hidden_dim)
        self.ln = LayerNormalization()
        self.dropout = Dropout(0.1)
    
    def call(self, h):
        seq_len = tf.shape(h)[1]
        Q = self.W_q(h)
        K = self.W_k(h)
        V = self.W_v(h)
        
        attn_scores = tf.matmul(Q, K, transpose_b=True)
        
        # Improved causal mask
        mask = 1 - tf.linalg.band_part(tf.ones_like(attn_scores), -1, 0)
        attn_scores = attn_scores - 1e9 * mask
        
        attn_weights = tf.nn.softmax(attn_scores / tf.sqrt(tf.cast(tf.shape(K)[-1], tf.float32)))
        attn_weights = self.dropout(attn_weights)
        
        return self.ln(tf.matmul(attn_weights, V) + h)  # Residual connection

        
@tf.keras.utils.register_keras_serializable()
class SoftAttention(Layer):
    """Multi-Head Soft Attention."""
    def __init__(self, hidden_dim, num_heads):
        super(SoftAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = hidden_dim // num_heads
        self.W_q = Dense(hidden_dim)
        self.W_k = Dense(hidden_dim)
        self.W_v = Dense(hidden_dim)
        self.W_o = Dense(hidden_dim)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, h):
        batch_size = tf.shape(h)[0]
        Q = self.split_heads(self.W_q(h), batch_size)
        K = self.split_heads(self.W_k(h), batch_size)
        V = self.split_heads(self.W_v(h), batch_size)
        attn_scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_output = tf.matmul(attn_weights, V)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        return self.W_o(tf.reshape(attn_output, (batch_size, -1, self.num_heads * self.depth)))

        
@tf.keras.utils.register_keras_serializable()
class AdaptiveAttentionFusion(Layer):
    def __init__(self, hidden_dim):
        super(AdaptiveAttentionFusion, self).__init__()
        self.attention = Dense(3, activation='softmax')
        self.ln = LayerNormalization()
    
    def call(self, mhsa, causal, soft, fcm_guide):
        # Learn dynamic weights using FCM guidance
        weights = self.attention(fcm_guide)  # Shape: [batch, seq, 3]
        fused = weights[:, :, 0:1] * mhsa + weights[:, :, 1:2] * causal + weights[:, :, 2:3] * soft
        return self.ln(fused + mhsa + causal + soft)  # Enhanced residual


@tf.keras.utils.register_keras_serializable()
class LSTMForecastingModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, pred_len, output_dim,**kwargs):
        super(LSTMForecastingModel, self).__init__(**kwargs)
        
        self.fcm = EnhancedFCM(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # Improved Encoder with dropout
        self.lstm_encoder = tf.keras.Sequential([
            tf.keras.Sequential([
                Bidirectional(LSTM(hidden_dim//2, return_sequences=True,
                                 kernel_initializer=initializers.GlorotNormal(seed=SEED))),
                LayerNormalization(),
                Dropout(0.2)
            ]) for _ in range(num_layers)
        ])
        
        # Attention components with residual connections
        self.attn_mhsa = MHSA(hidden_dim, num_heads)
        self.attn_causal = CausalAttention(hidden_dim)
        self.attn_soft = SoftAttention(hidden_dim,num_heads)  # Reuse causal with full visibility
        
        # Adaptive fusion
        self.fusion = AdaptiveAttentionFusion(hidden_dim)
        
        # Causal Decoder for temporal coherence
        self.lstm_decoder = tf.keras.Sequential([
            tf.keras.Sequential([
                LSTM(hidden_dim, return_sequences=True,
                    kernel_initializer=initializers.GlorotNormal(seed=SEED)),
                LayerNormalization(),
                Dropout(0.2)
            ]) for _ in range(num_layers)
        ])
        
        # Improved output projection
        self.time_distribute = tf.keras.Sequential([
            Dense(hidden_dim, activation='tanh'),
            Dropout(0.2),
            Dense(pred_len * output_dim,
                kernel_initializer=initializers.GlorotNormal(seed=SEED))
        ])
        
        self.pred_len = pred_len
        self.output_dim = output_dim
    
    def call(self, x):
        # Feature enhancement
        x_fcm = self.fcm(x)
        
        # Encoder processing
        enc_out = self.lstm_encoder(x_fcm)
        
        # Attention mechanisms
        mhsa_out = self.attn_mhsa(enc_out)
        causal_out = self.attn_causal(enc_out)
        soft_out = self.attn_soft(enc_out)
        
        # Adaptive fusion
        fused_attn = self.fusion(mhsa_out, causal_out, soft_out, x_fcm)
        
        # Causal decoding
        dec_out = self.lstm_decoder(fused_attn)
        
        # Final prediction
        dense_out = self.time_distribute(dec_out[:, -1, :])
        return tf.reshape(dense_out, [-1, self.pred_len, self.output_dim])

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'pred_len': self.pred_len,
            'output_dim': self.output_dim
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)