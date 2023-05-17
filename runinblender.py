import json
import os
import re
import subprocess
import sys
import time

import bpy
import nltk
import numpy as np
import openai
import requests
import speech_recognition as sr
import wave
from gtts import gTTS
from mutagen.mp3 import MP3
from nltk.corpus import cmudict
from pydub import AudioSegment
from pydub.utils import mediainfo
from vosk import Model, KaldiRecognizer, SetLogLevel

# Set up OpenAI credentials
openai.api_key = "APIKEYHERE"

# Set up Elvenlabs credentials
api_key = "APIKEYHERE"

# extract_time_value function contains FPS to set(Default is 24)

# Check if a sequence editor exists, create one if not
if bpy.context.scene.sequence_editor is None:
    bpy.context.scene.sequence_editor_create()

# Clear the current sequence editor by removing all sound strips
sequence_editor = bpy.context.scene.sequence_editor
if sequence_editor:
    for strip in sequence_editor.sequences_all:
        sequence_editor.sequences.remove(strip)

# Clear the current timeline markers
bpy.context.scene.timeline_markers.clear()

# Iterate over all objects in the scene
for obj in bpy.context.scene.objects:
    # Check if the object is a mesh
    if obj.type == 'MESH':
        mesh = obj.data

        # Check if the mesh has shape keys
        if mesh.shape_keys is not None:
            # Remove all shape key animation data
            if mesh.animation_data is not None:
                mesh.animation_data_clear()

# Define the function to get audio and send to OpenAI
def get_audio():
    
    # Check if a sequence editor exists, create one if not
    if bpy.context.scene.sequence_editor is None:
        bpy.context.scene.sequence_editor_create()
    
    # Clear the current sequence editor by removing all sound strips
    sequence_editor = bpy.context.scene.sequence_editor
    if sequence_editor:
        for strip in sequence_editor.sequences_all:
            sequence_editor.sequences.remove(strip)
    
    # Clear the current timeline markers
    bpy.context.scene.timeline_markers.clear()
    
    # Iterate over all objects in the scene
    for obj in bpy.context.scene.objects:
        # Check if the object is a mesh
        if obj.type == 'MESH':
            mesh = obj.data
    
            # Check if the mesh has shape keys
            if mesh.shape_keys is not None:
                # Remove all shape key animation data
                if mesh.animation_data is not None:
                    mesh.animation_data_clear()
                    
    # Record audio from microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    # Convert audio to text using speech recognition library
    try:
        text = r.recognize_google(audio)
        print("You said: " + text)

        # Check if the user said "ask AI" and send the rest of the text to OpenAI
        if "ask AI" in text:
            said = text.split("ask AI", 1)[1].strip()
            tosend = "respond in a short precise manner" + said
            print("Sending to OpenAI: " + said)
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": tosend}])
                ChatGPTresponse = completion.choices[0].message.content
                print("OpenAI response: " + ChatGPTresponse)
                
                # Call the elvenlabs_audio function to create an audio file
                mp3_file = elvenlabs_audio(ChatGPTresponse)
                if mp3_file is not None:
                    return mp3_file  # Return the filename
            except openai.OpenAIError as e:
                print("OpenAI API error:", e)
        else:
            print("Please say 'ask AI' followed by your question or statement.")
    except sr.UnknownValueError:
        print("Could not understand audio")
        return
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return


def elvenlabs_audio(ChatGPTresponse):
    url = 'https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL/stream?optimize_streaming_latency=0'

    headers = {
        'accept': '*/*',
        'xi-api-key': api_key,
        'Content-Type': 'application/json'
    }

    data = {
        "text": ChatGPTresponse,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        # Get the current blend file directory
        blend_file_directory = bpy.path.abspath("//")
        # Create a unique filename using a timestamp
        timestamp = str(int(time.time()))
        mp3_file = os.path.join(blend_file_directory, 'output' + timestamp + '.mp3')
        with open(mp3_file, 'wb') as f:
            f.write(response.content)
        print("MP3 file saved to:", mp3_file)
        audio_filename = mp3_file  # Assign mp3_file to audio_filename
        print(f"{mp3_file}") 
        print(f"{audio_filename}") 
        return mp3_file  # Return the filename

    else:
        print("Failed to download MP3 file. Status code:", response.status_code)
        return None

audio_filename = get_audio()

nltk.download('cmudict')

pronunciation_dictionary = cmudict.dict()

class Word:
    def __init__(self, data):
        self.confidence = data['conf']
        self.start_time = data['start']
        self.end_time = data['end']
        self.word = data['word']

    def to_string(self):
        return f"{self.word:20s} from {self.start_time:.2f} sec to {self.end_time:.2f} sec, confidence is {self.confidence:.2%}"

SetLogLevel(0)

model_path = "C:/vosk-model/models/vosk-model-en-us-0.22"

if not os.path.exists(model_path):
    print(f"Please download the model from https://alphacephei.com/vosk/models and unpack it as {model_path}")

print(f"Reading your vosk model '{model_path}'...")
model = Model(model_path)
print(f"'{model_path}' model was successfully read")

# Extract base name and extension
base_name, _ = os.path.splitext(audio_filename)

# Create names for the converted and text files
converted_filename = f"{base_name}.wav"
text_filename = f"{base_name}_speech_recognition_systems_vosk_with_timestamps.txt"

# Check if the file exists
if not os.path.exists(audio_filename):
    print(f"File '{audio_filename}' doesn't exist")
    
# Convert MP3 to WAV
command = f"ffmpeg -i {audio_filename} {converted_filename}"
subprocess.run(command, shell=True, check=True)

print(f"File '{audio_filename}' was successfully converted to WAV format")

print(f"Reading your file '{converted_filename}'...")
wf = wave.open(converted_filename, "rb")
print(f"'{converted_filename}' file was successfully read")

if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)

results = []

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        part_result = json.loads(rec.Result())
        results.append(part_result)

part_result = json.loads(rec.FinalResult())
results.append(part_result)

list_of_words = []
for sentence in results:
    if len(sentence) == 1:
        continue
    for obj in sentence['result']:
        w = Word(obj)
        list_of_words.append(w)

output_filename = "outputtimes.txt"

with open(output_filename, "w") as output_file:
    for word in list_of_words:
        output_file.write(word.to_string() + "\n")

text = ''
for r in results:
    text += r['text'] + ' '

print("\tVosk thinks you said:\n")
print(text)

print(f"Saving text to '{text_filename}'...")
os.makedirs(os.path.dirname(text_filename), exist_ok=True)  # Create directory if it doesn't exist
with open(text_filename, "w") as text_file:
    text_file.write(text)
print(f"Text successfully saved")

def convert_to_phonemes(word):
    phonemes = pronunciation_dictionary.get(word.lower())
    return phonemes if phonemes else []

def extract_time_value(part):
    match = re.search(r'\d+\.\d+', part)
    if match:
        time_value = float(match.group()) * 24
        return round(time_value)
    else:
        return 0

# Initialize your list
phoneme_sequence = []

with open('outputtimes.txt', 'r') as file:
    for line in file:
        split_parts = line.split()
        word = split_parts[0]

        start_index = next((i for i, part in enumerate(split_parts) if part == 'from'), None)
        end_index = next((i for i, part in enumerate(split_parts) if part == 'to'), None)

        if start_index is not None and end_index is not None:
            start_time_str = split_parts[start_index + 1]
            end_time_str = split_parts[end_index + 1]

            start_time = extract_time_value(start_time_str)
            end_time = extract_time_value(end_time_str)
        else:
            start_time = 0
            end_time = 0

        other_part = f'from {start_time} keyframe to {end_time} keyframe'
        phonemes = convert_to_phonemes(word)
        phonemes_str = str(phonemes[0]) if phonemes else ''
        print(f'    (\'{word}\', ({phonemes_str}, {start_time}, {end_time})),')

        # Add these values to the list as a tuple
        phoneme_sequence.append((word, (phonemes_str, start_time, end_time)))

print(f"{phoneme_sequence}")

# Define the mapping
shape_key_mapping = {
    'AO': 'AOOW',  
    'OW': 'AOOW',
    'AW': 'AWAA',
    'AA': 'AWAA',
    'B': 'BMP',
    'M': 'BMP',
    'P': 'BMP',
    'CH': 'CHJH',
    'JH': 'CHJH',
    'F': 'FV',
    'V': 'FV',
    'K': 'KGNG',
    'G': 'KGNG',
    'NG': 'KGNG',
    'T': 'TDN',
    'D': 'TDN',
    'N': 'TDN',
    'S': 'SZ',
    'Z': 'SZ',
    'W': 'WUW',
    'UW': 'WUW',
    'AY': 'AY',
    'AE': 'AE',
    'AH': 'AH',
    'EH': 'EH',
    'ER': 'ER',
    'EY': 'EY',
    'DH': 'DH',
    'HH': 'HH',
    'IY': 'IY',
    'IH': 'IH',
    'L': 'L',
    'OY': 'OY',
    'R': 'R',
    'SH': 'SH',
    'TH': 'TH',
    'UH': 'UH',
    'Y': 'Y',
    'ZH': 'ZH'
}

# Assume the object with the shape keys is the active object
obj = bpy.context.object

# Set all shape keys to 0 at keyframe 0
for key in obj.data.shape_keys.key_blocks:
    key.value = 0
    key.keyframe_insert(data_path="value", frame=0)

def lerp(a, b, t):
    """Linear interpolation between a and b"""
    return a + (b - a) * t * 0.75

# Iterate through each word in the input
for word, data in phoneme_sequence:
    phonemes, start_frame, end_frame = data
    frames_per_phoneme = np.linspace(start_frame, end_frame, len(phonemes) * 2)

    for i, phoneme in enumerate(phonemes):
        # Remove any numbers from the phoneme
        phoneme = ''.join([i for i in phoneme if not i.isdigit()])

        # Find the matching shape key
        shape_key_name = shape_key_mapping.get(phoneme)

        if shape_key_name:
            # Get the shape key
            shape_key = obj.data.shape_keys.key_blocks[shape_key_name]
            
            # Set all shape keys to 0 at the start frame
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0
                key.keyframe_insert(data_path="value", frame=start_frame)

            # Insert the keyframes
            for j in range(2):
                # Define the value
                # Use the lerp function to smoothly transition between 0 and 1, and 1 and 0
                value = lerp(0, 0.75, j / (2 - 1))

                # Set the value of the shape key
                shape_key.value = value

                # Insert a keyframe
                frame_position = frames_per_phoneme[i * 2 + j]
                shape_key.keyframe_insert(data_path="value", frame=frame_position)

# Get the active object
obj = bpy.context.active_object

# Get the shape key animation data
shape_key_anim_data = obj.data.shape_keys.animation_data

# Check if shape key animation data exists
if shape_key_anim_data:
    # Get the shape key fcurves
    shape_key_fcurves = shape_key_anim_data.action.fcurves
    
    # Find the last inserted shape key fcurve
    last_shape_key_fcurve = None
    last_keyframe_number = -1

    for fcurve in shape_key_fcurves:
        if fcurve.data_path.startswith('key_blocks["') and fcurve.data_path.endswith('"].value'):
            keyframe_points = fcurve.keyframe_points

            if keyframe_points:
                last_keyframe = keyframe_points[-1]
                keyframe_number = last_keyframe.co.x

                if keyframe_number > last_keyframe_number:
                    last_shape_key_fcurve = fcurve
                    last_keyframe_number = keyframe_number
    
    # Check if a shape key fcurve was found
    if last_shape_key_fcurve:
        shape_key_name = last_shape_key_fcurve.data_path.split('"')[1]
        print("Last inserted shape key:", shape_key_name)
        print("Keyframe number:", last_keyframe_number)
    
    else:
        print("No shape key fcurves found in the dope sheet.")

else:
    print("No shape key animation data found in the object.")
    

# Convert the keyframe number to an integer
last_keyframe_number = int(last_keyframe_number)

# Calculate the end frame by adding 5 keyframes to the last keyframe number
end_frame = last_keyframe_number + 5

# Set the start and end frames
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = end_frame

# Set the current frame to the start frame
bpy.context.scene.frame_set(0)

# Get the sequence editor
sequence_editor = bpy.context.scene.sequence_editor

# Ensure that the sequence editor is enabled
if sequence_editor is None:
    sequence_editor = bpy.context.scene.sequence_editor_create()

# Remove existing audio strips
audio_strips = [strip for strip in sequence_editor.sequences if strip.type == 'SOUND']
for strip in audio_strips:
    sequence_editor.sequences.remove(strip)

# Add the audio strip
sequence_editor.sequences.new_sound(name="Audio", filepath=audio_filename, channel=1, frame_start=1)

# Play the animation
bpy.ops.screen.animation_play()
