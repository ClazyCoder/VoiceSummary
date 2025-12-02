import dotenv
import logging
from voice import parse_speakers_and_transcript
import os
import argparse
from pathlib import Path


def validate_audio_path(audio_path: str) -> None:
    """
    오디오 파일 경로의 유효성을 검사합니다.

    Args:
        audio_path (str): 검사할 오디오 파일 경로

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: 파일 확장자가 지원되지 않거나 파일이 아닌 경우
    """
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    if not path.is_file():
        raise ValueError(f"지정된 경로가 파일이 아닙니다: {audio_path}")

    # 지원되는 오디오 파일 확장자
    supported_extensions = {'.mp3', '.wav', '.m4a',
                            '.flac', '.ogg', '.opus', '.aac', '.wma'}
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"지원되지 않는 오디오 파일 형식입니다: {path.suffix}. "
            f"지원되는 형식: {', '.join(supported_extensions)}"
        )


def main():
    parser = argparse.ArgumentParser(description='VoiceSummary')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    parser.add_argument('language', type=str,
                        help='Language of the audio file')
    parser.add_argument('min_speakers', type=int,
                        help='Minimum number of speakers to expect in the audio')
    parser.add_argument('max_speakers', type=int,
                        help='Maximum number of speakers to expect in the audio')
    args = parser.parse_args()
    # Ensure the 'logs' directory exists before setting up logging
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler('logs/voicesummary.log')])
    logger.info("VoiceSummary started!")

    # 오디오 파일 경로 유효성 검사
    try:
        logger.info(f"Validating audio file path: {args.audio_path}")
        validate_audio_path(args.audio_path)
        logger.info("Audio file path is valid!")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"오디오 파일 경로 검증 실패: {e}")
        raise

    logger.info("Loading environment variables...")
    dotenv.load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN 환경 변수가 설정되지 않았습니다.")
        raise ValueError("HF_TOKEN 환경 변수가 필요합니다.")
    logger.info("Environment variables loaded!")
    logger.info("Parsing speakers and transcript...")

    try:
        result = parse_speakers_and_transcript(
            args.audio_path, args.language, args.min_speakers, args.max_speakers, hf_token)
        logger.info("Parsing completed!")
        return result
    except Exception as e:
        logger.error(f"오디오 파일 처리 중 오류가 발생했습니다: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    result = main()
    print(result)
