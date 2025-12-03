import dotenv
import logging
from voice import parse_speakers_and_transcript
import os
import argparse
from pathlib import Path
from llm import LLMModule
from datetime import datetime


def validate_audio_path(audio_path: str) -> None:
    """
    Validates the audio file path.

    Args:
        audio_path (str): Audio file path to validate

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file extension is not supported or the path is not a file
    """
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not path.is_file():
        raise ValueError(f"The specified path is not a file: {audio_path}")

    # Supported audio file extensions
    supported_extensions = {'.mp3', '.wav', '.m4a',
                            '.flac', '.ogg', '.opus', '.aac', '.wma'}
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported audio file format: {path.suffix}. "
            f"Supported formats: {', '.join(supported_extensions)}"
        )


def main():
    parser = argparse.ArgumentParser(description='VoiceSummary')
    parser.add_argument('--audio_path', type=str,
                        help='Path to the audio file')
    parser.add_argument('--language', type=str,
                        help='Language of the audio file', default='en')
    parser.add_argument('--min_speakers', type=int,
                        help='Minimum number of speakers to expect in the audio', default=1)
    parser.add_argument('--max_speakers', type=int,
                        help='Maximum number of speakers to expect in the audio', default=4)
    args = parser.parse_args()
    dotenv.load_dotenv()
    # Ensure the 'logs' directory exists before setting up logging
    os.makedirs('logs', exist_ok=True)
    # Ensure the RESULTS_DIR directory exists before saving the results
    os.makedirs(os.getenv("RESULTS_DIR", "results"), exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler('logs/voicesummary.log')])
    logger.info("VoiceSummary started!")

    # Validate audio file path
    try:
        logger.info(f"Validating audio file path: {args.audio_path}")
        validate_audio_path(args.audio_path)
        logger.info("Audio file path is valid!")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Audio file path validation failed: {e}")
        raise

    logger.info("Loading environment variables...")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable is not set.")
        raise ValueError("HF_TOKEN environment variable is required.")
    logger.info("Environment variables loaded!")
    logger.info("Parsing speakers and transcript...")

    try:
        transcripts = parse_speakers_and_transcript(
            args.audio_path, args.language, args.min_speakers, args.max_speakers, hf_token)
        logger.info("Parsing completed!")
        logger.info("Saving transcript to results directory...")
        transcript_file_name = f"transcript_{args.language}_{os.path.basename(args.audio_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(os.path.join(os.getenv("RESULTS_DIR", "results"), transcript_file_name), "w", encoding="utf-8") as f:
            f.write(transcripts)
        logger.info("Transcript saved to results directory!")

        model_name = os.getenv("LLM_MODEL", "qwen3:8b")
        llm_module = LLMModule(model_name)
        summary = llm_module.summarize_transcript(transcripts, args.language)
        logger.info("Summary completed!")

        logger.info("Saving summary to results directory...")
        summary_file_name = f"summary_{args.language}_{os.path.basename(args.audio_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(os.path.join(os.getenv("RESULTS_DIR", "results"), summary_file_name), "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info("Summary saved to results directory!")
        logger.debug(summary)

        return summary
    except Exception as e:
        logger.error(
            f"An error occurred while summarizing the audio file: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    summary = main()
    print(summary)
