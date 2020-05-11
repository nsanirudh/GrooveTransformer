from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
from typing import Union, List, Optional

import tensorflow as tf
# tfds works in both Eager and Graph modes
tf.enable_eager_execution() #not needed in TF V2, as it is already the default
import tensorflow_datasets as tfds
import magenta.music as mm

from magenta.protobuf.music_pb2 import NoteSequence

from visual_midi import Plotter
from pretty_midi import PrettyMIDI

from IPython.core.display import display, HTML
from IPython.display import IFrame


#######################################################################################
#######################################################################################
##################### UTILS FROM MUSIC GENERATION BY MAGENTA ##########################
############################# BY ALEXANDRE DuBreuil ###################################
#######################################################################################
# Codes in this section come from chapter 4: note_sequence_utils.py
# PASTED HERE BY BH

def save_midi(sequences: Union[NoteSequence, List[NoteSequence]], output_dir: Optional[str] = None, prefix: str = "sequence"):
    '''
    Writes the sequences as MIDI files to the "output" directory, with the
  filename pattern "<prefix>_<index>_<date_time>" and "mid" as extension.
  
  :param sequences: a NoteSequence or list of NoteSequence to be saved
  :param output_dir: an optional subdirectory in the output directory
  :param prefix: an optional prefix for each file
    '''
    output_dir = os.path.join("output", output_dir) if output_dir else "output"
    os.makedirs(output_dir, exist_ok=True)
    if not isinstance(sequences, list):
        sequences = [sequences]
    for (index, sequence) in enumerate(sequences):
        date_and_time = time.strftime("%Y-%m-%d_%H%M%S")
        filename = f"{prefix}_{index:02}_{date_and_time}.mid"
        path = os.path.join(output_dir, filename)
        mm.midi_io.note_sequence_to_midi_file(sequence, path)
        print(f"Generated midi file: {os.path.abspath(path)}")

def save_plot(sequences: Union[NoteSequence, List[NoteSequence]],
              output_dir: Optional[str] = None,
              prefix: str = "sequence",
              **kwargs):
    '''
  Writes the sequences as HTML plot files to the "output" directory, with the
  filename pattern "<prefix>_<index>_<date_time>" and "html" as extension.

      :param sequences: a NoteSequence or list of NoteSequence to be saved
      :param output_dir: an optional subdirectory in the output directory
      :param prefix: an optional prefix for each file
      :param kwargs: the keyword arguments to pass to the Plotter instance
    '''
    output_dir = os.path.join("output", output_dir) if output_dir else "output"
    os.makedirs(output_dir, exist_ok=True)
    if not isinstance(sequences, list):
        sequences = [sequences]
    for (index, sequence) in enumerate(sequences):
        date_and_time = time.strftime("%Y-%m-%d_%H%M%S")
        filename = f"{prefix}_{index:02}_{date_and_time}.html"
        path = os.path.join(output_dir, filename)
        midi = mm.midi_io.note_sequence_to_pretty_midi(sequence)
        plotter = Plotter(**kwargs)
        plotter.save(midi, path)
        print(f"Generated plot file: {os.path.abspath(path)}")

#######################################################################################
#######################################################################################
##################### PLOTTING UTILS ##################################################
#######################################################################################
#######################################################################################

def plot_midi_file(midi_filepath, width=1000, height=300, 
                   temp_folder = "misc/pretty midi images/", 
                   temp_filename = "temp"):
    '''
    Written by BH
    
    This Function Plots a midi file in a Jupyter Notebook. 
    To do so, using visual_midi and pretty_midi libraries and interactive 
    html plot (html_filename) of the midi file is generated in html_folder 
    and an IFrame object is returned
    
    Inputs:
        (1) midi_filepath: path for the midi file
                Note: midi_filepath should end in ".mid"
        (2) width, height (opt): width and height of IFrame
        (3) temp_folder (opt): folder to store the interactive html and (possibly) the midi
        (4) temp_filename (opt): filename of the interactive html and (possibly) the midi
    
    Output:
        (1) IFrame object containing the interactive midi plot
    '''
     
    # Create path if doesnt exist
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    # load the midi file into a PrettyMIDI object (& remove midi if necessary)
    pm = PrettyMIDI(midi_filepath)
        
    # Create an interactive html midi Plotter 
    plotter = Plotter()
    html_path = temp_folder+temp_filename+".html"
    plotter.show(pm, html_path)
    
    # Return IFrame of the interactive plot
    ifm = IFrame(src=html_path, width=width, height=height)    
    
    return ifm


def plot_midi_tf_tensor(midi_tf_tensor, width=1000, height=300, 
                  temp_folder="misc/pretty midi images/", 
                  temp_filename="temp", 
                  keep_midi=True):
    '''
    Written by BH
    
    This Function Plots a midi tensor file in a Jupyter Notebook. 
    midi tensor file is a midi file stored in a tf.Tensor (directly accessible 
    through tf.tensorflow_datasets)
    To do so, using visual_midi and pretty_midi libraries and interactive 
    html plot (html_filename) of the midi file is generated in html_folder 
    and an IFrame object is returned
    
    Inputs:
        (1) midi_filepath: path for the midi file
                Note: midi_filepath should end in ".mid"
        (2) width, height (opt): width and height of IFrame
        (3) temp_folder (opt): folder to store the interactive html and (possibly) the midi
        (4) temp_filename (opt): filename of the interactive html and (possibly) the midi
    
    Output:
        (1) IFrame object containing the interactive midi plot
    '''
    
    # Convert midifile tf.tensor to note_sequence and plot using plot_note_seq()
    note_seq = mm.midi_to_note_sequence(tfds.as_numpy(midi_tf_tensor))

    ifm = plot_note_seq(note_seq, width=width, height=height, 
                        temp_folder=temp_folder, temp_filename=temp_filename, 
                        keep_midi=keep_midi)
    
    return ifm
    
    
def plot_note_seq(note_seq, width=1000, height=300, 
                  temp_folder="misc/pretty midi images/", 
                  temp_filename="temp", 
                  keep_midi=True):
    '''
    Written by BH
    
    This Function Plots a note sequence in a Jupyter Notebook. 
    To do so, a temporary midi file of the the note_seq is generated. 
    Then using visual_midi and pretty_midi libraries and interactive 
    html plot (html_filename) of the midi file is generated in html_folder 
    and an IFrame object is returned
    
    Inputs:
        (1) note_seq: input note sequence
                Type: 'music_pb2.NoteSequence' (Magenta's NoteSequence Object)
        (2) width, height (opt): width and height of IFrame
        (3) temp_folder (opt): folder to store the interactive html and (possibly) the midi
        (4) temp_filename (opt): filename of the interactive html and (possibly) the midi
        (5) keep_midi (opt): Deletes temp midi if set to False
    
    Output:
        (1) IFrame object containing the interactive midi plot
    '''
    # Create path if doesnt exist
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Convert note sequence to midi file
    midi_path = temp_folder+temp_filename+".mid"
    mm.sequence_proto_to_midi_file(note_seq, midi_path)  #export as a temporary midi file
    
    # load the midi file into a PrettyMIDI object 
    pm = PrettyMIDI(midi_path)

    
    ifm = plot_midi_file(midi_path, width=width, height=height, 
                         temp_folder = temp_folder, 
                         temp_filename = temp_filename)
    
    if not keep_midi:
        os.remove(midi_path)
        
    return ifm
    
    