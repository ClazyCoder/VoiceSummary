from pathlib import Path
import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class TemplateManager:
    def __init__(self, base_dir: str = "src/prompts"):
        self.base_dir = Path(base_dir)
        self.system_prompt = ""
        self.templates = {}
        self._load_resources()

    def _load_resources(self):
        # 1. Load system prompt (use default if not found)
        sys_path = self.base_dir / "system.txt"
        logger.info(f"Loading system prompt from: {sys_path}")
        if sys_path.exists():
            self.system_prompt = sys_path.read_text(encoding="utf-8")
        else:
            logger.warning(
                f"System prompt not found at: {sys_path} using default prompt.")
            self.system_prompt = "You are a helpful assistant."

        # 2. Load language-specific templates
        template_dir = self.base_dir / "templates"
        logger.info(f"Loading templates from: {template_dir}")
        if template_dir.exists():
            for file_path in template_dir.glob("*.txt"):
                self.templates[file_path.stem] = file_path.read_text(
                    encoding="utf-8")
            logger.info(f"Loaded {len(self.templates)} templates.")
        else:
            logger.warning(
                f"Templates not found at: {template_dir} using default templates.")
            self.templates = {
                "en": "You are a helpful assistant.",
            }

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def get_summary_template(self, language: str) -> str:
        return self.templates.get(language, self.templates.get('en', ""))

    def get_composed_prompt(self, language: str) -> PromptTemplate:
        """
        Compose the system prompt with the summary template for the given language and return it
        """
        # 1. Get the template for the given language (if not found, use English, if not found, use empty string)
        target_template = self.templates.get(
            language, self.templates.get('en', ""))

        # 2. Create a LangChain PromptTemplate object
        # Note: system.txt contains {transcript}, {language}, {summary_template} variables
        prompt = PromptTemplate.from_template(
            template=self.system_prompt
        ).partial(summary_template=target_template)
        return prompt
