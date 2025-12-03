# from huggingface_hub import hf_hub_download # TODO : future
import os
import logging
from langchain_ollama import ChatOllama
import glob
from .template_manager import TemplateManager

logger = logging.getLogger(__name__)


class LLMModule:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        # TODO : Download LLM model from Hugging Face in Local inference mode.
        # self.model_path = os.path.join(
        #     os.getenv("MODEL_DIR", "./models"), self.model_name)
        # if not os.path.exists(self.model_path) and os.getenv("DOWNLOAD_MODEL", "true").lower() == "true":
        #     self.download_llm_model() # TODO : Download LLM model from Hugging Face in Local inference mode.
        self.template_manager = TemplateManager(
            base_dir=os.getenv("PROMPTS_DIR", "src/prompts"))
        model_type = os.getenv("MODEL_TYPE", "ollama")
        if model_type == "ollama":
            self.model = ChatOllama(model=self.model_name, base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"))
        # TODO : Add other model types here.
        elif model_type == "chatgpt":
            raise NotImplementedError("ChatGPT model is not implemented yet.")
        elif model_type == "groq":
            raise NotImplementedError("Groq model is not implemented yet.")
        elif model_type == "gemini":
            raise NotImplementedError("Gemini model is not implemented yet.")
        elif model_type == "claude":
            raise NotImplementedError("Claude model is not implemented yet.")
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    # TODO : Download LLM model from Hugging Face in Local inference mode.
    # def download_llm_model(self) -> None:
    #     hf_token = os.getenv("HF_TOKEN")
    #     repo_id = os.getenv("HF_MODEL_REPO", "Qwen3-8B-GGUF")
    #     if not hf_token:
    #         raise ValueError("HF_TOKEN is not set")
    #     if not repo_id:
    #         raise ValueError("HF_MODEL_REPO is not set")
    #     logger.info(f"Downloading LLM model: {repo_id} / {self.model_name}")
    #     MODEL_DIR = os.getenv("MODEL_DIR", "./models")
    #     if not os.path.exists(MODEL_DIR):
    #         os.makedirs(MODEL_DIR)
    #     hf_hub_download(repo_id=repo_id,  local_dir=self.model_path, filename=self.model_name,
    #                     use_auth_token=hf_token)
    #     logger.info(f"LLM model downloaded successfully: {self.model_name}")

    def summarize_transcript(self, transcript: str, language: str) -> str:
        logger.debug(f"Summarizing transcript: {transcript}")
        chain = self.template_manager.get_composed_prompt(
            language) | self.model
        response = chain.invoke(
            {"transcript": transcript, "language": language})
        return response.content
