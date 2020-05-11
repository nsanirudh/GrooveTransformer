import tensorflow as tf
import tensorflow_datasets as tfds
import numpy
import magenta.music as mm
from magenta.models.music_vae.data import GrooveConverter
from magenta.models.music_vae import configs
from magenta.models.music_vae import data
tf.enable_eager_execution()

#Load and process data

#Load available configurations for available tasks
config_2_bar = configs.CONFIG_MAP['groovae_2bar_humanize']
config_tap_fixed_velocity = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
config_tap_fixed_velocity_dropout = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity_note_dropout']
configs_add_closed_hh = configs.CONFIG_MAP['groovae_2bar_add_closed_hh']
configs_hit_control = configs.CONFIG_MAP['groovae_2bar_hits_control_tfds']

def get_input_tensors(dataset,config):

    batch_size = config.hparams.batch_size
    # batch_size = 32
    iterator = dataset.make_one_shot_iterator()
    # dataset = dataset.unbatch()
    # dataset = dataset.batch(batch_size)

    input_sequence,output_sequence,control_sequence,sequence_length = iterator.get_next()
    #input shape
    input_sequence.set_shape([batch_size,None,config.data_converter.input_depth])
    #output shape
    output_sequence.set_shape([batch_size,None,config.data_converter.output_depth])
    #control shape
    if not config.data_converter.control_depth:
        control_sequence=None
    else:
        control_sequence.set_shape([batch_size,None,config.data_converter.control_depth])
    #sequence length
    sequence_length.set_shape([batch_size]+sequence_length.shape[1:].as_list())

    return {
        'input_sequence': input_sequence,
        'output_sequence': output_sequence,
        'control_sequence':control_sequence,
        'sequence_length':sequence_length
    }

def get_dataset(config,is_training=False,cache_dataset=True):

    batch_size = config.hparams.batch_size

    data_converter = config.data_converter
    data_converter.set_mode('train' if is_training else 'eval')

    tf.logging.info('Reading examples from TFDS: %s',config.tfds_name)
    dataset = tfds.load(
        config.tfds_name,
        split=tfds.Split.TRAIN if is_training else tfds.Split.VALIDATION,
        shuffle_files=is_training,
        try_gcs=False
    )

    def _tf_midi_to_notesequence(ex):
        return tf.py_function(
            lambda x:[mm.midi_to_note_sequence(x.numpy()).SerializeToString()],
            inp = [ex['midi']],
            Tout= tf.string,
            name='midi_to_note_sequence')

    def _remove_pad_fn(padded_seq_1, padded_seq_2, padded_seq_3, length):
        if length.shape.ndims == 0:
            return (padded_seq_1[0:length], padded_seq_2[0:length],
                    padded_seq_3[0:length], length)
        else:
            # Don't remove padding for hierarchical examples.
            return padded_seq_1, padded_seq_2, padded_seq_3, length

    dataset = (dataset.map(
        _tf_midi_to_notesequence,
        num_parallel_calls=tf.data.experimental.AUTOTUNE))

    dataset = (dataset
               .map(data_converter.tf_to_tensors,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .flat_map(lambda *t: tf.data.Dataset.from_tensor_slices(t))
               .map(_remove_pad_fn,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE))

    if is_training:
        dataset =dataset.shuffle(buffer_size=10 * batch_size).repeat()

    dataset = dataset.padded_batch(
        batch_size,
        dataset.output_shapes,
        drop_remainder=True
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def get_input_output_tensors(dataset):
    batch_size = 32

    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size)
    dataset = iter(dataset)
    inputs,outputs, ctrl, seqlen = next(dataset)


    return inputs, outputs