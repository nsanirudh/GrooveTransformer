# pip install magenta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import magenta
import tensorflow as tf
#print(tf.__version__)
#print(magenta.__version__)

# Magenta Imports
import magenta
import magenta.music as mm
from magenta.models.music_vae.data import GrooveConverter
from magenta.models.music_vae import configs
from magenta.models.music_vae import data
from magenta.models.music_vae.data import *

#Tensorflow Imports

import tensorflow as tf
#!pip install tensorflow_datasets
import tensorflow_datasets as tfds

# Enable Eager Execution
# tf.enable_eager_execution()

# tf.compat_v1.disable_eager_execution()

# DATA PROCESSING FUNCTIONS

# Load available configs for GROOVAE Dataset
config_2_bar = configs.CONFIG_MAP['groovae_2bar_humanize']
config_tap_fixed_velocity = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
config_tap_fixed_velocity_dropout = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity_note_dropout']
configs_add_closed_hh = configs.CONFIG_MAP['groovae_2bar_add_closed_hh']
configs_hit_control = configs.CONFIG_MAP['groovae_2bar_hits_control_tfds']

# Get dataset from TFDS and store it in a tf.Data Object

def initialize_dataset_as_iterator(config, batch_size, is_training=False, cache_dataset=True):
    data_converter = config.data_converter
    data_converter.set_mode('train' if is_training else 'eval')

        # tf.compat.v1.logging('Reading examples from TFDS: %s',config.tfds_name)
    dataset = tfds.load(
        config.tfds_name,
        split=tfds.Split.TRAIN if is_training else tfds.Split.VALIDATION,
        shuffle_files=is_training,
        try_gcs=False
    )
    
    print("The number of elements is this dataset is: {}".format(tf.compat.v2.data.experimental.cardinality(dataset)))

    # print(dataset)

    midi_to_note = lambda x:[mm.midi_to_note_sequence(x.numpy()).SerializeToString()]

#### FUNCTION 1
    def _tf_midi_to_notesequence(ex):
        return tf.py_function(
            midi_to_note,
            inp = [ex['midi']],
            Tout= tf.string,
            name='midi_to_note_sequence')


####FUNCTION 2
    def _binary_to_decimal_2(inputs,outputs,_,__):
    
      sequence_length = 32
      drum_features = 9
      prev = 0

      inputs = tf.squeeze(inputs)
      outputs = tf.squeeze(outputs)
      
      def binary_logic(x):
        
        result = tf.reduce_sum(
        tf.cast( x, dtype=tf.int64)
        * 2 ** tf.range(tf.cast(tf.size(x), dtype=tf.int64)))
        return result
    
      input_digit = [binary_logic(inputs[i][:9]) for i in range(sequence_length)]
      output_digit = [binary_logic(outputs[i][:9]) for i in range(sequence_length)]

      #If required to return a tensor
      input_tensor_digit = tf.convert_to_tensor(input_digit, dtype=tf.int64)
      output_tensor_digit = tf.convert_to_tensor(output_digit, dtype=tf.int64)

      return input_tensor_digit, output_tensor_digit

#### FUNCTION 3
    def _remove_pad_fn(padded_seq_1, padded_seq_2, padded_seq_3, length):
        if length.shape.ndims == 0:
            return (padded_seq_1[0:length], padded_seq_2[0:length],
                    padded_seq_3[0:length], length)
        else:
            # Don't remove padding for hierarchical examples.
            return padded_seq_1, padded_seq_2, padded_seq_3, length     

#### MAP FUNCTION 1
    dataset = (dataset.map(_tf_midi_to_notesequence,num_parallel_calls=tf.data.experimental.AUTOTUNE))

    # print(dataset)

#### MAP TO TENSORS
    dataset = dataset.map(
      tf.autograph.experimental.do_not_convert(
          functools.partial(convert_to_tensors_op, converter=data_converter)),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

####
    dataset = dataset.unbatch()

#### MAP FUNCTION 2
    dataset = dataset.map(_binary_to_decimal_2,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

#### MAP FUNCTION3
    # dataset = dataset.map(
    #   _remove_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # print(dataset)

#### SHUFFLE IF IS_TRAINING
    if is_training:
        dataset = dataset.shuffle(buffer_size=10 * batch_size)


#### MAKE THE DATASET INTO A BATCH
    # dataset = dataset.padded_batch(
    #     1,
    #     dataset.output_shapes,
    #     drop_remainder=True
    # ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.padded_batch(
      batch_size,
      tf.data.get_output_shapes(dataset),
      drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)



#### MAKE DATASET AN ITERATOR
    # dataset = iter(dataset)
    
    # print(dataset)

    return dataset

# val_dataset = Initialize_dataset_as_iterator(configs_add_closed_hh,64)

# train_dataset = Initialize_dataset_as_iterator(configs_add_closed_hh,64,is_training=True)

# TRANSFORMER CODE

# Posiontal Encoding

def get_angles(pos, i, d_model): 

    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

#Masking

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


#Attention

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


# Point wise feed forward layer

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)])  # (batch_size, seq_len, d_model)

# Encoder

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)

# Decoder

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


# Assemble the transformer


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights

    num_layers = 4
    d_model = 512
    dff = 512
    num_heads = 8

    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2
    dropout_rate = 0.1

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)    


d_model = 512
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)

#plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
#plt.ylabel("Learning Rate")
#plt.xlabel("Train Step")

# LOSS

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')

val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

# Training and Experiments 

# tensors = Initialize_dataset_as_iterator(configs_add_closed_hh,64,is_training = True)

# Hyperparameters

num_layers = 2
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = 512 + 2
target_vocab_size = 512 + 2
dropout_rate = 0.25

EPOCHS = 1

# Create the Transformer

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


# Training 

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')



# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)

transformer.call  


# Validation step

val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=val_step_signature)
def val_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  

  predictions, _ = transformer(inp, tar_inp, 
                                False, 
                                enc_padding_mask, 
                                combined_mask, 
                                dec_padding_mask)
  loss = loss_function(tar_real, predictions)
  
  val_loss(loss)
  val_accuracy(tar_real, predictions)


# Train Loop

loss = []
val_Loss = []
for epoch in range(EPOCHS):

  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  start = time.time()

  loss.append(train_loss.result().numpy())
  val_Loss.append(val_loss.result().numpy())
  
  train_loss.reset_states()
  train_accuracy.reset_states()
  val_loss.reset_states()
  val_accuracy.reset_states()

  train_dataset = initialize_dataset_as_iterator(configs_add_closed_hh,64,is_training = True)
  val_dataset = initialize_dataset_as_iterator(configs_add_closed_hh,64)

  for (batch,(inp,tar)) in enumerate(train_dataset):
    train_step(inp, tar)
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))

  for (batch,(inp,tar)) in enumerate(val_dataset):
    val_step(inp, tar)
    if batch % 50 == 0:
      print ('Validation: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, val_loss.result(), val_accuracy.result()))
  

  if (epoch + 1) % 2 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, loss, label="train_loss")
plt.plot(N, val_Loss, label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("losses.png")

print(loss)

transformwe.summary()

print(dir(transformer.layers[2]))