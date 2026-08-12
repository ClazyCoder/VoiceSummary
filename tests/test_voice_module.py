import os
import unittest
from unittest.mock import MagicMock, patch

from src.voice import voice_module


class ParseSpeakersAndTranscriptTests(unittest.TestCase):
    @patch.object(voice_module.torch.cuda, "is_available", return_value=False)
    @patch.object(voice_module.whisperx, "assign_word_speakers")
    @patch.object(voice_module.whisperx, "align")
    @patch.object(voice_module.whisperx, "load_align_model")
    @patch.object(voice_module.whisperx, "load_audio")
    @patch.object(voice_module.whisperx, "load_model")
    @patch.object(voice_module, "DiarizationPipeline")
    def test_passes_hugging_face_token_to_current_diarization_api(
        self,
        diarization_pipeline,
        load_model,
        load_audio,
        load_align_model,
        align,
        assign_word_speakers,
        _cuda_available,
    ):
        transcription_model = MagicMock()
        transcription_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": "Hello"}],
        }
        load_model.return_value = transcription_model
        load_audio.return_value = [0.0]
        load_align_model.return_value = (MagicMock(), {})
        align.return_value = {"segments": [{"text": "Hello"}]}

        diarization_model = diarization_pipeline.return_value
        diarization_segments = MagicMock()
        diarization_model.return_value = diarization_segments
        assign_word_speakers.return_value = {
            "segments": [{"speaker": "SPEAKER_00", "text": "Hello"}]
        }

        with patch.dict(
            os.environ,
            {"BATCH_SIZE": "2", "COMPUTE_TYPE": "float32"},
        ):
            transcript = voice_module.parse_speakers_and_transcript(
                "audio.wav", "en", 1, 2, "hf-token"
            )

        diarization_pipeline.assert_called_once_with(token="hf-token", device="cpu")
        diarization_model.assert_called_once_with(
            [0.0], min_speakers=1, max_speakers=2
        )
        self.assertEqual(transcript, "SPEAKER_00: Hello")


if __name__ == "__main__":
    unittest.main()
