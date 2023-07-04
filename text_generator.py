import os
import time

import numpy as np
import tensorflow as tf

path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt', 
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

example_texts = ["abcdefg", "xyz"]
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)
ids = ids_from_chars(chars)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)
chars = chars_from_ids(ids)

text_from_chars = tf.strings.reduce_join(chars, axis=-1).numpy()

