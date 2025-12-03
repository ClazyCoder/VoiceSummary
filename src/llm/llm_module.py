from huggingface_hub import snapshot_download
import os
import logging
from langchain_community.chat_models import ChatLlamaCpp


logger = logging.getLogger(__name__)


prompt_template = """
You are a helpful assistant that summarizes transcripts of meetings.
You will be given a list of transcripts and you will need to summarize them.
The participants of the meeting is appeared in the transcript as "SPEAKER_00", "SPEAKER_01", ... and if the speaker is unknown, "UNKNOWN".
You will need to use the following user's transcript:
{transcript}

Return the summary in the following template:
{summary_template}
/no_think
"""

summary_template = f"""
# Head (Main Content of the transcript)
...
# Summary (Summary of the transcript)
...
# Participants (Participants of the transcript if their names are appeared in the transcript)
...
# Conclusion (Conclusion of the transcript)
...
# Notes (Notes of the transcript)
...
# Important Points (Important texts of the transcript)
...
"""


class LLMModule:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.model_path = os.join(
            os.getenv("MODEL_DIR", "./models"), self.model_name)
        if not os.path.exists(self.model_path):
            self.download_llm_model()
        self.model = ChatLlamaCpp(model_path=self.model_name, think=False, temperature=0.7, top_p=0.8,
                                  top_k=20, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                                  repetition_penalty=1.5, )

    def download_llm_model(self) -> None:
        logger.info(f"Downloading LLM model: {self.model_name}")
        MODEL_DIR = os.getenv("MODEL_DIR", "./models")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        snapshot_download(self.model_name, local_dir=MODEL_DIR)
        logger.info(f"LLM model downloaded successfully: {self.model_name}")

    def summarize_transcript(self, transcript: list[str]) -> str:
        logger.info(f"Summarizing transcript: {transcript}")
        response = self.model.invoke(prompt_template.format(
            transcript=transcript, summary_template=summary_template))
        return response.content
