import os
import tensorflow as tf
from six.moves import urllib
from typing import List

from magenta.models.music_vae import TrainedModel, configs
import midi_utils as mu
from magenta.protobuf.music_pb2 import NoteSequence


#######################################################################################
#######################################################################################
##################### UTILS FROM MUSIC GENERATION BY MAGENTA ##########################
############################# BY Alexandre DuBreuil ###################################
#######################################################################################
# Codes in this section come from chapter 4 (page 113-)

def download_checkpoint(model_name: str, checkpoint_name: str, target_dir: str):
    '''
    Code Written by Alexandre DuBreuil
    Comments by BH
    
    This function downloads a pretrained MUSICVAE checkpoint
    Available checkpoints:
    https://github.com/magenta/magenta-js/blob/master/music/checkpoints/README.md
    
    Analogus to download_bundle in magenta.music.notebook_utils

    
    Inputs:
        (1) model_name: name of model (refer to the above GitHub link) 
                examples: "MusicVAE", "MusicRNN", "OnsetsAndFrames"
                
        (2) checkpoint_name: name of checkpoint (refer to the above GitHub link)
                examples: "tap2drum_2bar", "groovae_4bar", "groovae_2bar_humanize"
            
        (3) target_dir: directory to save model
    '''
    tf.gfile.MakeDirs(target_dir)
    checkpoint_target = os.path.join(target_dir, checkpoint_name)
    if not os.path.exists(checkpoint_target):
        response = urllib.request.urlopen(f"https://storage.googleapis.com/magentadata/models/" f"{model_name}/checkpoints/{checkpoint_name}")
        data = response.read()
        local_file = open(checkpoint_target, 'wb')
        local_file.write(data)
        local_file.close()
        


def get_model(name: str):
    '''
    Code Written by Alexandre DuBreuil
    Comments by BH
    
    From the book:
    
    instantiates the MusicVAE model using the checkpoint
    
    In this method, we first download the checkpoint for the given model name with our
    download_checkpoint method. Then, we instantiate the TrainedModel class from
    magenta.models.music_vae with the checkpoint, batch_size=8. This value defines
    how many sequences the model will process at the same time.
        
    Having a batch size that's too big will result in wasted overhead; a batch size too small will
    result in multiple passes, probably making the whole code run slower. Unlike during
    training, the batch size doesn't need to be big
    
    CONFIG_MAPs can be found here
    https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/configs.py
    
    Inputs:
        (1) name: model name
        
    '''                             
    checkpoint = name + ".tar"
    download_checkpoint("music_vae", checkpoint, "bundles")
    return TrainedModel(
        # Removes the .lohl in some training checkpoints
        # which shares the same config
        configs.CONFIG_MAP[name.split(".")[0] if "." in name else name],
        # The batch size changes the number of sequences
        # to be run together
        batch_size=8,
        checkpoint_dir_or_path=os.path.join("bundles", checkpoint))
                                          

def sample(model_name: str,num_steps_per_sample: int) -> List[NoteSequence]:
    '''
    Code Written by Alexandre DuBreuil
    Comments by BH
    
    From the book:
    
    In this method, we first instantiate the model using our previous get_model
    method. We then call the sample method, asking for n=2 sequences that the
    method will return. We are keeping the default temperature (which is 1.0, for all
    models), but we can change it using the temperature parameter. Finally, we
    save the MIDI files and the plot files using the save_midi and
    save_plot methods respectively, from the previous chapter, present in the
    utils.py file.
    
    CONFIG_MAPs can be found here
    https://github.com/tensorflow/magenta/blob/master/magenta/models/music_vae/configs.py
    
    Inputs:
        (1) model_name: model name
        (2) num_steps_per_sample: length of the sample in number of grid lines     
        
    Output:
        (1) sample_sequences: List of generated (sampled) note sequences
        
    '''                                           
    model = get_model(model_name)
    # Uses the model to sample 2 sequences
    sample_sequences = model.sample(n=2, length=num_steps_per_sample)
    # Saves the midi and the plot in the sample folder
    mu.save_midi(sample_sequences, "sample", model_name)
    mu.save_plot(sample_sequences, "sample", model_name)
    return sample_sequences