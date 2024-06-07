import whisper
import os

def transcribe_audio(name, audio_path=None, use_existing=True):
    # Load the Whisper model
    model = whisper.load_model("large")
    shortened_name = name.replace(" ", "").lower()
    if audio_path is None:
        audio_path = f'{shortened_name}_transcript.wav'


    # Check if the transcript already exists
    if os.path.isfile(f'{shortened_name}_transcript.txt') and use_existing:
        concatenated_name = name.replace(" ", "").lower()
        with open(f'{shortened_name}_styleguide.json', 'r') as file:
            transcribed_text = file.read()
    # Else, create
    else:
        # Load and transcribe the audio file
        result = model.transcribe(audio_path)

        # Output the transcription
        print(result["text"])
        transcribed_text = result["text"]

        if name is None:
            # Save the transcription to a text file
            with open('transcript.txt', 'w') as file:
                file.write(result["text"])
        else:
            with open(f'{shortened_name}_transcript.txt', 'w') as file:
                file.write(result["text"])

    return transcribed_text



if __name__ == "__main__":
    # TESTING
    # Specify the path to your audio file
    path = "billnye_transcript.wav"  # default naming of audio file
    name =  "Bill Nye"
    transcribe_audio(name)


