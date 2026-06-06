from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import (
    download_youtube_audio,
    process_input
)

from core.transcriber import transcribe_all

source = input("Enter YouTube URL or File Path: ").strip().strip('"').strip("'")
language = "english"
if source.startswith(("http://", "https://")):
    transcript = download_youtube_audio(source)
else:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=language)


if source.startswith("http"):

    transcript = download_youtube_audio(source)

else:

    chunks = process_input(source)

    transcript = transcribe_all(
        chunks,
        language=language
    )

print("\n" + "=" * 60)
print("📝 TRANSCRIPT")
print("=" * 60)

print(
    transcript[:500] + "..."
    if len(transcript) > 500
    else transcript
)