import json
from typing import List, Dict, Any

class Prompts:
    """Class to handle loading and accessing prompts."""

    def __init__(self, file_path: str):
        """
        Initialize the Prompts class by reading a JSON file.
        :param file_path: Path to the JSON file containing language and prompts.
        """
        self.file_path = file_path
        self.language: str = ""
        self.name = file_path
        self.prompts: List[Dict[str, Any]] = []
        self._load_file()

    def _load_file(self):
        """Private method to load JSON content into attributes."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.language = data.get("language", "")
                self.name = data.get("name")
                self.prompts = data.get("prompts", [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error reading JSON file: {e}")

    def get_language(self) -> str:
        """Return the language of the prompts."""
        return self.language

    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """Return all prompts as a list of dictionaries."""
        return self.prompts

    def get_prompt_by_id(self, prompt_id: int) -> Dict[str, Any]:
        """Return a single prompt by its ID."""
        for prompt in self.prompts:
            if prompt.get("id") == prompt_id:
                return prompt
        raise ValueError(f"Prompt with id {prompt_id} not found.")

    def get_texts(self) -> List[str]:
        """Return a list of all prompt texts."""
        return [prompt.get("text", "") for prompt in self.prompts]

    def get_tokens_by_id(self, prompt_id: int) -> List[str]:
        """Return tokens for a specific prompt by ID."""
        prompt = self.get_prompt_by_id(prompt_id)
        return prompt.get("tokens", [])
    
    def to_json(self) -> str:
        """Return the prompts as a JSON string."""
        return json.dumps({
            "name": self.name,
            "language": self.language,
            "prompts": self.prompts
        }, ensure_ascii=False, indent=4)



sv_standard_prompts = None

def get_sv_standard_prompts() -> Prompts:
    """Return a singleton instance of Prompts loaded from the standard prompts JSON file."""
    global sv_standard_prompts
    if sv_standard_prompts != None:
        return sv_standard_prompts

    sv_standard_prompts = Prompts("./assets/prompts/sv-standard-prompts.json")
    return sv_standard_prompts