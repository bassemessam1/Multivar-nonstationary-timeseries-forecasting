import tensorflow as tf
import keras
from keras import initializers
from keras._tf_keras.keras.layers import Layer, Dense, LSTM, Bidirectional, Dropout, LayerNormalization
import random
import numpy as np

# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

@tf.keras.utils.register_keras_serializable()
class EnhancedFCM_mhsa(Layer):
    def __init__(self, input_dim):
        super(EnhancedFCM_mhsa, self).__init__()
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
class MHSA_mhsa(Layer):
    def __init__(self, hidden_dim, num_heads):
        super(MHSA_mhsa, self).__init__()
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
class LSTMForecastingModel_mhsa(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, pred_len, output_dim,**kwargs):
        super(LSTMForecastingModel_mhsa, self).__init__(**kwargs)
        
        self.fcm = EnhancedFCM_mhsa(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        # Improved Encoder with dropout
        self.lstm_encoder = tf.keras.Sequential([
            Bidirectional(LSTM(hidden_dim // 2,
                               return_sequences=True,
                               kernel_initializer=initializers.GlorotNormal(seed=SEED)))
            for _ in range(num_layers)
        ])
        
        # Attention components
        self.attn_mhsa = MHSA_mhsa(hidden_dim, num_heads)
        
        # Bidirectional LSTM Decoder
        self.lstm_decoder = tf.keras.Sequential([
            tf.keras.Sequential([
                LSTM(hidden_dim, return_sequences=True,
                    kernel_initializer=initializers.GlorotNormal(seed=SEED)),
                LayerNormalization(),
                Dropout(0.2)
            ]) for _ in range(num_layers)
        ])

        
        self.time_distribute = Dense(pred_len * output_dim,
                                    kernel_initializer=initializers.GlorotNormal(seed=SEED))
        self.pred_len = pred_len
        self.output_dim = output_dim
    
    def call(self, x):
        # Enhanced FCM processing
        x_fcm = self.fcm(x)
        
        # Bidirectional LSTM encoding
        lstm_out = self.lstm_encoder(x_fcm)
        
        # Attention mechanism
        mhsa_out = self.attn_mhsa(lstm_out)
        
        # Bidirectional LSTM decoding
        decoded_out = self.lstm_decoder(mhsa_out)
        
        # Final prediction
        dense_out = self.time_distribute(decoded_out[:, -1, :])
        y_pred = tf.reshape(dense_out, [-1, self.pred_len, self.output_dim])
        return y_pred


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