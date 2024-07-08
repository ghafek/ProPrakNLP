# Populism in Podcasts
## Podcast as a Medium: The Big Picture
Podcasts are on the rise, with approximately half a billion listeners by 2023 and generating a revenue of $3.46 billion in the same year. The growth of podcasts as a medium is undeniable, bringing both advantages and disadvantages. Podcasts have become a great medium, especially in science, due to their ease of production and consumption. Their digital nature has removed barriers of place and time in human communication, and has also broken down stereotypes about who can talk about science.

However, the disadvantages are also significant. The fight against disinformation and misinformation is more serious in this medium than in others. Click-bait driven production, the lack of fact-checking, and the absence of feedback options from listeners are major challenges. The relationship between podcasters and listeners often makes it easier to spread false information, as listeners may accept misinformation more readily from their podcasters.

In health and lifestyle podcasts, misinformation is particularly prevalent. Podcasters and guests may twist information and oversimplify research results. This is not just a problem of podcasts, but a broader issue within the science communication ecosystem.

## The Goal 
The goal of this small NLP project is to make a small contribution towards setting the record straight. With the help of NLP, AI models will be trained and fine-tuned to highlight the most important information in podcasts. The project aims to compare the information presented in podcasts with research papers and validate it, providing a correctness score.

Other important facts mentioned in the podcast will be highlighted and categorized as either (No Resources: NR) or (MPPO: Might be Personal Opinion).


## Technologies Used
- **[Whisper by OpenAI]([LINK](https://github.com/openai/whisper))**: Used for transcribing audio to text.
- **[WhisperX]([LINK](https://github.com/m-bain/whisperX))**: Utilized for transcription and timestamping with speaker diarization.
- **[Llama2]([LINK](https://github.com/meta-llama/llama))**: Employed for training and fine-tuning AI models to highlight and validate information.
- **[Google Colab]([LINK](https://colab.research.google.com))**: Leveraged for cloud computing to efficiently run the code.

## Methodology
**Transcription**: Use Whisper to transcribe podcast audio into text.
**Speaker Identification**: Apply WhisperX to identify and assign speakers in the transcript.
**Information Highlighting**: Fine-tune Llama2 to highlight the most important information.
**Validation**: Cross-check highlighted information with external resources to validate or flag it as unverifiable.

## Installation

Make sure to have the necessary packages (e.g., pip) and required models (Whisper, WhisperX) installed.

We will first demonstrate a small piece of the transcription step using an audio file of short length.

### Google Colab
For better performance, run the notebook with a GPU instance. Upload a short audio file with more than one speaker and then run the following code:

```python
# Import the necessary modules
import torch
import whisperx 
# Check if CUDA is available, otherwise use CPU
device = "cuda"

# Parameters
batch_size = 4
compute_type = "float16"
audio_file = "YOUR_AUDIO_FILE.wav"

# Load the Whisper model using the available device
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# Load and process the audio file
audio = whisperx.load_audio(audio_file)

# Transcribe the audio
result= model.transcribe(audio, batch_size=batch_size)
# Initialize the diarization pipeline
diarize_model = whisperx.DiarizationPipeline(use_auth_token="HUGGIN_FACE_TOKEN")

# Perform diarization
diarize_segments = diarize_model(audio_file, min_speakers=2, max_speakers=3)

# Print the results
print(diarize_segments)
print(result["segments"])
# showing the speakers with IDs
diarize_segments.speaker.unique()
```

### expected Output
```text
segment                           label    speaker       start          end
0   [00:00:00.008 --> 00:00:23.641]    A    SPEAKER_02    0.008489     23.641766  
1   [00:00:30.772 --> 00:00:32.809]    B    SPEAKER_01   30.772496     32.809847
2   [00:00:33.471 --> 00:00:40.212]    C    SPEAKER_01   33.471986     40.212224
3   [00:00:40.466 --> 00:00:47.342]    D    SPEAKER_01   40.466893     47.342954
4   [00:00:48.378 --> 00:01:11.230]    E    SPEAKER_01   48.378608     71.230900
[
    {'text': 'Sample text A', 'start': 0.008489, 'end': 23.641766},
    {'text': 'Sample text B', 'start': 30.772496, 'end': 32.809847},
    {'text': 'Sample text C', 'start': 33.471986, 'end': 40.212224},
    {'text': 'Sample text D', 'start': 40.466893, 'end': 47.342954},
    {'text': 'Sample text E', 'start': 48.378608, 'end': 71.230900},
]
```

### Local on CPU
To run the code locally on a CPU, use the following commands after ensuring the necessary packages and models are installed:

```python
import ssl
import whisperx
import urllib.request
import json

# Bypass SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Rest of your code
device = "cpu"
batch_size = 4
compute_type = "float32"
audio_file = "YOUR_AUDIO_FILE.wav"


model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
diarize_model = whisperx.DiarizationPipeline(use_auth_token="HUGGIN_FACE_TOKEN")
diarize_segments_df = diarize_model(audio_file, min_speakers=2, max_speakers=3)

# Convert diarize_segments to a list of dictionaries
diarize_segments = diarize_segments_df.to_dict('records')

# Print diarize_segments to understand its structure
print("Diarization segments structure:")
print(diarize_segments)

# Manually assign speakers to the transcription segments
segments_with_speakers = []
for segment in result["segments"]:
    start_time = segment["start"]
    end_time = segment["end"]
    text = segment["text"]
    speaker = None

    # Iterate over diarize_segments to find the correct speaker
    for diarize_segment in diarize_segments:
        diarize_start = diarize_segment["start"]
        diarize_end = diarize_segment["end"]
        diarize_speaker = diarize_segment["speaker"]

        # Check if the transcription segment overlaps with the diarization segment
        if diarize_start <= start_time < diarize_end or diarize_start < end_time <= diarize_end:
            speaker = diarize_speaker
            break

    duration = end_time - start_time
    segments_with_speakers.append({
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "speaker": speaker,
        "text": text
    })
```
### expected Output 
``` text
segment                           label    speaker       start          end
0   [00:00:00.008 --> 00:00:23.641]    A    SPEAKER_02    0.008488964    23.641765704  
1   [00:00:30.772 --> 00:00:32.809]    B    SPEAKER_01   30.772495755    32.809847198
2   [00:00:33.471 --> 00:00:40.212]    C    SPEAKER_01   33.471986417    40.212224108
3   [00:00:40.466 --> 00:00:47.342]    D    SPEAKER_01   40.466893039    47.342954159
```
``` JSON
[
  {
    "start_time": 0.009,
    "end_time": 23.643,
    "duration": 23.634,
    "speaker": "SPEAKER_02",
    "text": "Sample text A"
  },
  {
    "start_time": 30.776,
    "end_time": 47.312,
    "duration": 16.535999999999998,
    "speaker": "SPEAKER_01",
    "text": "Sample text B"
  },
  {
    "start_time": 48.37,
    "end_time": 71.254,
    "duration": 22.884000000000007,
    "speaker": null,
    "text": "Sample text C"
  },
  {
    "start_time": 71.459,
    "end_time": 84.377,
    "duration": 12.917999999999992,
    "speaker": "SPEAKER_00",
    "text": "Sample text D"
  },
  {
    "start_time": 84.514,
    "end_time": 103.336,
    "duration": 18.822000000000003,
    "speaker": null,
    "text": "Sample text E"
  }
]
```
## Contributing


## Acknowledgements
