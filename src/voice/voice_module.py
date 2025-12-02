import whisperx
from whisperx.diarize import DiarizationPipeline
import gc
import torch
import logging


def parse_speakers_and_transcript(audio_path: str, language: str, hf_token: str) -> list[dict]:
    """
    Parse the speakers and transcript from the audio file.
    """
    logger = logging.getLogger(__name__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16  # reduce if low on GPU mem
    # change to "int8" if low on GPU mem (may reduce accuracy)
    compute_type = "float16"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(
        "large-v2", device, compute_type=compute_type, language=language)

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    logger.info("Transcription completed!")
    logger.info(result["segments"])  # before alignment

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a,
                            metadata, audio, device, return_char_alignments=False)
    logger.info("Alignment completed!")
    logger.info(result["segments"])  # after alignment

    del model_a
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Assign speaker labels
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio, min_speakers=4, max_speakers=4)

    logger.info("Diarization completed!")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    logger.info(diarize_segments)
    logger.info(result["segments"])  # segments are now assigned speaker IDs
    return result["segments"]
