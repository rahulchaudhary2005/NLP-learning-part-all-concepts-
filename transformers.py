"""
COMPLETE TRANSFORMER PIPELINE IN TENSORFLOW
Text → Tokenization → Embedding → Positional Encoding
→ MultiHead Attention → Feed Forward → Transformer Block → Output
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# ------------------------------------------------
# 1 TOKENIZATION
# ------------------------------------------------

vocab = {"I":0, "love":1, "AI":2, "<PAD>":3}
vocab_size = len(vocab)

def tokenize(sentence):
    return [vocab[word] for word in sentence.split()]

sentence = "I love AI"
tokens = tokenize(sentence)
tokens = tf.constant([tokens])


# ------------------------------------------------
# 2 EMBEDDING LAYER
# ------------------------------------------------

embedding_dim = 64
embedding = layers.Embedding(vocab_size, embedding_dim)

x = embedding(tokens)


# ------------------------------------------------
# 3 POSITIONAL ENCODING
# ------------------------------------------------

def positional_encoding(length, depth):

    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis,:]/depth

    angle_rates = 1/(10000**depths)
    angle_rads = positions*angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

pos = positional_encoding(50, embedding_dim)
x = x + pos[:x.shape[1]]


# ------------------------------------------------
# 4 MULTI HEAD SELF ATTENTION
# ------------------------------------------------

class MultiHeadSelfAttention(layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

    def call(self, inputs):
        return self.att(inputs, inputs)


# ------------------------------------------------
# 5 FEED FORWARD NETWORK
# ------------------------------------------------

def feed_forward(embed_dim):

    return tf.keras.Sequential([
        layers.Dense(2048, activation="relu"),
        layers.Dense(embed_dim)
    ])


# ------------------------------------------------
# 6 TRANSFORMER BLOCK
# ------------------------------------------------

class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = feed_forward(embed_dim)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs):

        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)

        out1 = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)

        return self.norm2(out1 + ffn_output)


# ------------------------------------------------
# 7 TRANSFORMER MODEL
# ------------------------------------------------

class TransformerModel(tf.keras.Model):

    def __init__(self, vocab_size, embed_dim=64, num_heads=8, layers_count=4):
        super().__init__()

        self.embedding = layers.Embedding(vocab_size, embed_dim)

        self.blocks = [
            TransformerBlock(embed_dim, num_heads)
            for _ in range(layers_count)
        ]

        self.final = layers.Dense(vocab_size)

    def call(self, inputs):

        x = self.embedding(inputs)

        for block in self.blocks:
            x = block(x)

        return self.final(x)


# ------------------------------------------------
# 8 RUN MODEL
# ------------------------------------------------

model = TransformerModel(vocab_size)

output = model(tokens)

prediction = tf.argmax(output, axis=-1)

print("Prediction:", prediction)