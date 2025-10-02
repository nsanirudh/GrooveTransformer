# GrooveTransformer

This is a research project to work with the expanded groove MIDI dataset and train a 
transformer to create MIDI files with certain input visualizable parameters.


Set up via git:

```commandline
git clone https://github.com/nsanirudh/GrooveTransformer.git
git remote add origin https://github.com/nsanirudh/GrooveTransformer.git
```


Should be ready to go.

First Steps:
```commandline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_updated.txt
```

```commandline
INPUT_DIRECTORY=e-gmd-v1.0.0

# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive
```

```commandline
fix numpy error in by changing np.bool to np.bool_ inside tensorflow-probability
```

