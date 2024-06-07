import whisper

def transcribe_audio(audio_path, name=None):
    # Load the Whisper model
    model = whisper.load_model("large")

    # Load and transcribe the audio file
    result = model.transcribe(audio_path)

    # Output the transcription
    print(result["text"])

    if name is None:
        # Save the transcription to a text file
        with open('transcript.txt', 'w') as file:
            file.write(result["text"])
    else:
        concatenated_name = name.replace(" ", "").lower()
        with open(f'{concatenated_name}_transcript.txt', 'w') as file:
            file.write(result["text"])


if __name__ == "__main__":
    # TESTING
    # Specify the path to your audio file
    path = "lecture.wav"  # default naming of audio file
    transcribe_audio(path)
