# from huggingface_hub import hf_hub_download # TODO : future
import os
import logging
from langchain_ollama import ChatOllama
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

    def summarize_transcript(self, transcript: str, language: str) -> str:
        logger.debug(f"Summarizing transcript: {transcript}")
        chain = self.template_manager.get_composed_prompt(
            language) | self.model
        try:
            response = chain.invoke(
                {"transcript": transcript, "language": language})
            return response.content
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate summary: {str(e)}") from e
