import dotenv
import logging
from voice import parse_speakers_and_transcript
import os
import argparse


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
    logger.info("Loading environment variables...")
    dotenv.load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    logger.info("Environment variables loaded!")
    logger.info("Parsing speakers and transcript...")
    result = parse_speakers_and_transcript(
        args.audio_path, args.language, args.min_speakers, args.max_speakers, hf_token)
    logger.info("Parsing completed!")
    return result


if __name__ == "__main__":
    result = main()
    print(result)
