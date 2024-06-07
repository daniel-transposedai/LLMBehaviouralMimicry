import argparse
from audioTranscription import transcribe_audio

from textSplitHDPTopic import transcript_to_upsert


'''


'''

if __name__ == "__main__":
    # TESTING
    # Specify the path to your audio file

    parser = argparse.ArgumentParser(description='Transcribe audio file.')
    parser.add_argument('--audio_file', type=str, help='The path to the audio transcription.')
    parser.add_argument('--speaker_name', type=str, help='The name of the speaker.')

    args = parser.parse_args()

    print(f"File: {args.file}")
    print(f"Text: {args.text}")


    # Transcribe audio
    transcription_path = args.file  # default naming of audio file
    speaker_name = args.text
    text = transcribe_audio(transcription_path, speaker_name)

    # HDP, Topic Model, Split, Encode, and Upsert into Pinecone Vector DB
    transcript_to_upsert(speaker_name)

    # now you are ready to inference!
    # TODO: create an automated update of the display in the ui to reflect additions
    print("Done!")

