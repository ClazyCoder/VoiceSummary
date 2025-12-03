import whisperx
from whisperx.diarize import DiarizationPipeline
import gc
import torch
import logging


def format_transcript(segments):
    """
    WhisperX 결과(segments)를 받아서
    1. 같은 화자의 연속된 발언은 합치고
    2. 깔끔한 대본 형식(String)으로 변환
    """
    if not segments:
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
        list[str]: A list of formatted strings, each in the form "{speaker}: {text}" representing the transcript and speaker label for each segment.

    Raises:
        FileNotFoundError: If the audio file at `audio_path` does not exist or cannot be loaded.
        ValueError: If the specified `language` is not supported or invalid parameters are provided.
        RuntimeError: If authentication with Hugging Face using `hf_token` fails.
        Exception: For other errors raised by WhisperX or diarization pipeline.
    """
    logger = logging.getLogger(__name__)

    # 입력 파라미터 검증
    if not audio_path:
        raise ValueError("audio_path가 제공되지 않았습니다.")
    if not language:
        raise ValueError("language가 제공되지 않았습니다.")
    if min_speakers < 1:
        raise ValueError(f"min_speakers는 1 이상이어야 합니다. 현재 값: {min_speakers}")
    if max_speakers < min_speakers:
        raise ValueError(
            f"max_speakers는 min_speakers 이상이어야 합니다. min: {min_speakers}, max: {max_speakers}")
    if not hf_token:
        raise ValueError("hf_token이 제공되지 않았습니다.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16  # reduce if low on GPU mem
    # change to "int8" if low on GPU mem (may reduce accuracy)
    compute_type = "float16"

    try:
        # 1. Transcribe with original whisper (batched)
        logger.info(f"Loading WhisperX model for language: {language}")
        model = whisperx.load_model(
            "large-v2", device, compute_type=compute_type, language=language)

        logger.info(f"Loading audio file: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        if audio is None or len(audio) == 0:
            raise ValueError(f"오디오 파일을 로드할 수 없거나 파일이 비어있습니다: {audio_path}")

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
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        raise
    except ValueError as e:
        logger.error(f"잘못된 입력 값: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"런타임 오류 발생: {e}")
        if "authentication" in str(e).lower() or "token" in str(e).lower():
            raise RuntimeError(f"Hugging Face 인증 실패: {e}") from e
        raise
    except Exception as e:
        logger.error(f"오디오 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
        raise RuntimeError(f"오디오 파일 처리 중 오류가 발생했습니다: {e}") from e
