import os
import unittest
from unittest.mock import MagicMock, patch

from src.voice import voice_module
from src.voice.voice_module import format_timestamp, format_transcript


class FormatTimestampTests(unittest.TestCase):
    def test_zero_seconds(self):
        self.assertEqual(format_timestamp(0), "00:00:00")

    def test_seconds_only(self):
        self.assertEqual(format_timestamp(45.7), "00:00:45")

    def test_minutes_and_seconds(self):
        self.assertEqual(format_timestamp(125.3), "00:02:05")

    def test_hours_minutes_seconds(self):
        self.assertEqual(format_timestamp(3661.9), "01:01:01")

    def test_none_returns_zero(self):
        self.assertEqual(format_timestamp(None), "00:00:00")


class FormatTranscriptTests(unittest.TestCase):
    def test_empty_segments(self):
        self.assertEqual(format_transcript([]), "")

    def test_single_segment_with_timestamps(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.5}]
        result = format_transcript(segments)
        self.assertEqual(result, "[00:00:00 -> 00:00:01] SPEAKER_00: Hello")

    def test_multiple_segments_same_speaker_merged(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.5},
            {"speaker": "SPEAKER_00", "text": "world", "start": 1.5, "end": 3.0},
        ]
        result = format_transcript(segments)
        self.assertEqual(result, "[00:00:00 -> 00:00:03] SPEAKER_00: Hello world")

    def test_different_speakers_separated(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.5},
            {"speaker": "SPEAKER_01", "text": "Hi there", "start": 2.0, "end": 4.0},
        ]
        result = format_transcript(segments)
        expected = "[00:00:00 -> 00:00:01] SPEAKER_00: Hello\n\n[00:00:02 -> 00:00:04] SPEAKER_01: Hi there"
        self.assertEqual(result, expected)

    def test_missing_timestamps_defaults_to_zero(self):
        segments = [{"speaker": "SPEAKER_00", "text": "Hello"}]
        result = format_transcript(segments)
        self.assertEqual(result, "[00:00:00 -> 00:00:00] SPEAKER_00: Hello")


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
            "segments": [{"speaker": "SPEAKER_00", "text": "Hello", "start": 0.0, "end": 1.5}]
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
        self.assertEqual(transcript, "[00:00:00 -> 00:00:01] SPEAKER_00: Hello")


if __name__ == "__main__":
    unittest.main()
