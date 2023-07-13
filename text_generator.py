import os
import time

import numpy as np
import tensorflow as tf

# Download the file from the URL if it's not already present in the cache directory.
# The function returns the path to the downloaded file.
path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt', 
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
# Open the file in binary mode, read the contents, and then decode it into utf-8 format.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# Create a vocabulary from the text.
# Here, vocab is a sorted list of all unique characters present in the text.
vocab = sorted(set(text))

# Create a StringLookup layer which will turn characters into integer IDs
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)
# Create another StringLookup layer which will do the reverse:
# turn integer IDs back into characters
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)
# Define a function that can take a list of IDs and turn them back into text
def text_from_ids(ids):
    '''
    Convert the IDs to characters and then join them into a string
    '''
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Convert the entire text into integer IDs
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
# Create a TensorFlow Dataset object from the array of character IDs
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
# Define the length of each sequence
# Calculate the number of examples in each epoch
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)
# Batch the dataset into sequences of the specified length
sequences = ids_dataset.batch(seq_length + 1, drop_remainder = True)
# Define a function to split each sequence into input and target text
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
# Use the function to split each sequence in the dataset into input and target text
dataset = sequences.map(split_input_target)
# Define the batch size and buffer size for shuffling the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 10000
# Shuffle and batch the dataset, and prefetch batches for efficiency
dataset = (
    dataset.shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# Define the size of the vocabulary
# Define the number of dimensions for the embedding layer
# Define the number of units in the GRU layer
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
# Define a custom model class that inherits from tf.keras.Model
class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        # Call the constructor of the parent class
        super().__init__(self)
        # Define an Embedding layer that turns character IDs into dense vectors
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Define a GRU layer with rnn_units number of units
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        # Define a Dense layer that outputs a probability distribution over the vocabulary
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        # Pass the inputs through the Embedding layer
        x = self.embedding(inputs, training=training)
        # If no initial state is provided, get the initial state of the GRU layer
        if states is None:
            states = self.gru.get_initial_state(x)
        # Pass the output of the Embedding layer through the GRU layer
        x, states = self.gru(x, initial_state=states, training=training)
        # Pass the output of the GRU layer through the Dense layer
        x = self.dense(x, training=training)

        # Return the output and the final state of the GRU layer if return_state is True
        if return_state:
            return x, states
        # Otherwise, only return the output
        else:
            return x
# Create an instance of the custom model class
model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()), 
    embedding_dim=embedding_dim, 
    rnn_units=rnn_units
)

# Take the first batch of the dataset
for input_example_batch, target_example_batch in dataset.take(1):
    # Make predictions for the batch using the model
    example_batch_predictions = model(input_example_batch)

model.summary()

# Sample character indices from the categorical distribution defined by the predictions
sampled_indices = tf.random.categorical(
    example_batch_predictions[0], num_samples=1
)
# Remove dimensions of size 1 from the shape of the tensor
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
# Define the loss function
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# Calculate the mean loss for the example batch
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
# Compile the model with the Adam optimizer and the previously defined loss function
model.compile(optimizer='adam', 
              loss=loss)

# Define the directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Define the prefix of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
# Define a ModelCheckpoint callback to save the model's weights after every epoch
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True
)
# Define the number of epochs
EPOCHS = 30
# Train the model for a given number of epochs
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Define a custom model class for generating text one step at a time
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        # Initialize the class with a model, two StringLookup layers, and a temperature
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        # Create a mask to prevent "[UNK]" from being generated
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values = [-float('inf')] * len(skip_ids), 
            indices = skip_ids, 
            dense_shape = [len(ids_from_chars.get_vocabulary())]
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    # Decorate the function with tf.function for performance
    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        # Run the model to get predicted logits and states
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True
        )
        # Only use the last prediction
        predicted_logits = predicted_logits[:, -1, :]
        # Control the randomness of the output by dividing by the temperature
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask to prevent "[UNK]" from being generated
        predicted_logits = predicted_logits + self.prediction_mask
        # Sample from the output logits to generate token IDs
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        # Convert from token IDs to characters
        predicted_chars = self.chars_from_ids(predicted_ids)
        # Return the generated characters and the model state
        return predicted_chars, states    
# Create an instance of the OneStep model
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# Get the current time
start = time.time()
# Initialize the state of the OneStep model to None
states = None
# Define the initial string to start generating text from
next_char = tf.constant(['ROMEO:'])
# Initialize a list to hold the generated text
result = [next_char]

# Generate 1000 characters of text
for n in range(1000):
    # Use the OneStep model to generate the next character and the model state
    next_char, states = one_step_model.generate_one_step(
        next_char, states=states
    )
    # Append the generated character to the result list
    result.append(next_char)

# Join the characters in the result list into a string
result = tf.strings.join(result)
# Get the current time
end = time.time()
# Print the generated text
print(result[0].numpy().decode("utf-8"), "\n\n" + "_" * 80)
# Print the time taken to generate the text
print("\nRun time:", end - start)

tf.saved_model.save(one_step_model, "one_step")
# one_step_reloaded = tf.saved_model.load("one_step")