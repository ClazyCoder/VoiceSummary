import whisperx
from whisperx.diarize import DiarizationPipeline
import gc
import torch
import logging
import os


def format_transcript(segments):
    """
    Formats WhisperX segments by:
    1. Merging consecutive utterances from the same speaker
    2. Converting to a clean transcript format (String)
    """
    if not segments or not all('text' in seg for seg in segments):
        return ""

    formatted_lines = []

    current_speaker = segments[0].get('speaker', 'UNKNOWN')
    current_text = [segments[0]['text'].strip()]

    for seg in segments[1:]:
        speaker = seg.get('speaker', 'UNKNOWN')
        text = seg['text'].strip()

        if speaker == current_speaker:
            current_text.append(text)
        else:
            formatted_lines.append(
                f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [text]

    formatted_lines.append(f"{current_speaker}: {' '.join(current_text)}")

    return "\n\n".join(formatted_lines)


def parse_speakers_and_transcript(audio_path: str, language: str, min_speakers: int, max_speakers: int, hf_token: str) -> str:
    """
    Parses the speakers and transcript from the given audio file using WhisperX and diarization.

    Args:
        audio_path (str): Path to the audio file to be processed.
        language (str): Language code for transcription (e.g., 'en', 'fr').
        min_speakers (int): Minimum number of speakers to expect in the audio.
        max_speakers (int): Maximum number of speakers to expect in the audio.
        hf_token (str): Hugging Face authentication token for diarization model access.

    Returns:
        str: A formatted string containing the transcript with speaker labels, where each line is in the form "{speaker}: {text}". Multiple segments from the same speaker are merged, and segments are separated by double newlines.

    Raises:
        FileNotFoundError: If the audio file at `audio_path` does not exist or cannot be loaded.
        ValueError: If the specified `language` is not supported or invalid parameters are provided.
        RuntimeError: If authentication with Hugging Face using `hf_token` fails.
        Exception: For other errors raised by WhisperX or diarization pipeline.
    """
    logger = logging.getLogger(__name__)

    # Validate input parameters
    if not audio_path:
        raise ValueError("audio_path is not provided.")
    if not language:
        raise ValueError("language is not provided.")
    if min_speakers < 1:
        raise ValueError(
            f"min_speakers must be at least 1. Current value: {min_speakers}")
    if max_speakers < min_speakers:
        raise ValueError(
            f"max_speakers must be at least min_speakers. min: {min_speakers}, max: {max_speakers}")
    if not hf_token:
        raise ValueError("hf_token is not provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # reduce if low on GPU mem
        batch_size = int(os.getenv("BATCH_SIZE", 16))
    except ValueError:
        raise ValueError(
            f"BATCH_SIZE must be a number. Current value: {os.getenv('BATCH_SIZE')}")
    if batch_size < 1:
        raise ValueError(
            f"BATCH_SIZE must be a positive integer. Current value: {batch_size}")
    # change to "int8" if low on GPU mem (may reduce accuracy)
    compute_type = os.getenv("COMPUTE_TYPE", "float16")
    valid_compute_types = {"float16", "float32", "int8"}
    if compute_type not in valid_compute_types:
        raise ValueError(
            f"COMPUTE_TYPE must be one of {valid_compute_types}. Current value: {compute_type}")
    try:
        # 1. Transcribe with original whisper (batched)
        logger.info(f"Loading WhisperX model for language: {language}")
        model = whisperx.load_model(
            os.getenv("WHISPERX_MODEL", "large-v2"), device, compute_type=compute_type, language=language)

        logger.info(f"Loading audio file: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        if audio is None or len(audio) == 0:
            raise ValueError(
                f"Failed to load audio file or file is empty: {audio_path}")

        logger.info("Starting transcription...")
        result = model.transcribe(audio, batch_size=batch_size)
        logger.info("Transcription completed!")
        logger.debug(result["segments"])  # before alignment

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 2. Align whisper output
        logger.info("Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a,
                                metadata, audio, device, return_char_alignments=False)
        logger.info("Alignment completed!")
        logger.debug(result["segments"])  # after alignment

        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 3. Assign speaker labels
        logger.info("Initializing diarization pipeline...")
        diarize_model = DiarizationPipeline(
            use_auth_token=hf_token, device=device)

        logger.info(
            f"Starting diarization (min_speakers: {min_speakers}, max_speakers: {max_speakers})...")
        # add min/max number of speakers if known
        diarize_segments = diarize_model(
            audio, min_speakers=min_speakers, max_speakers=max_speakers)

        logger.info("Diarization completed!")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.debug(diarize_segments)
        # segments are now assigned speaker IDs
        logger.debug(result["segments"])
        full_diarization = format_transcript(result["segments"])
        return full_diarization

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input value: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error occurred: {e}")
        if "authentication" in str(e).lower() or "token" in str(e).lower():
            raise RuntimeError(
                f"Hugging Face authentication failed: {e}") from e
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error occurred during audio processing: {e}", exc_info=True)
        raise RuntimeError(
            f"An error occurred while processing the audio file: {e}") from e
