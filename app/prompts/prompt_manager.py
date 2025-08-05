from pathlib import Path
from langchain.prompts import PromptTemplate
from typing import Dict, List

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._prompts = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompts from markdown files"""
        prompt_configs = {
            "anonymizer.md": ["question"],
            "planner.md": ["question"], 
            "deanonymize.md": ["plan", "mapping"],
            "queries_from_plan.md": ["question", "plan"],
            "analysis.md": ["context", "plan", "original_question", "generated_queries_from_plan"],
            "multi_query_gen.md": ["question"],
            "query_optimization.md": ["query"],
            "remove_irrelevant_content.md": ["queries", "retrieved_documents"],
            "rerun.md": ["original_input_variables", "original_prompt", "format_object", "correction"]
        }
        
        for filename, variables in prompt_configs.items():
            file_path = self.prompts_dir / filename
            if file_path.exists():
                template = file_path.read_text(encoding='utf-8')
                prompt_name = filename.replace('.md', '').replace('-', '_')
                self._prompts[prompt_name] = PromptTemplate(
                    input_variables=variables,
                    template=template
                )
    
    @property
    def anonymizer_prompt(self) -> str:
        return self._prompts['anonymizer'].template
    
    @property
    def planner_prompt(self) -> str:
        return self._prompts['planner'].template
    
    @property
    def deanonymize_prompt(self) -> str:
        return self._prompts['deanonymize'].template
    
    @property
    def queries_from_plan_prompt(self) -> str:
        return self._prompts['queries_from_plan'].template
    
    @property
    def analysis_prompt(self) -> str:
        return self._prompts['analysis'].template
    
    @property
    def multi_query_gen_prompt(self) -> str:
        return self._prompts['multi_query_gen'].template
    
    @property
    def query_optimization_prompt(self) -> str:
        return self._prompts['query_optimization'].template
    
    @property
    def remove_irrelevant_content_prompt(self) -> str:
        return self._prompts['remove_irrelevant_content'].template
    
    @property
    def rerun_prompt(self) -> str:
        return self._prompts['rerun'].template
    
_prompt_manager = PromptManager()

# Export the prompts as module-level variables for backward compatibility
anonymizer_prompt = _prompt_manager.anonymizer_prompt
planner_prompt = _prompt_manager.planner_prompt
deanonymize_prompt = _prompt_manager.deanonymize_prompt
queries_from_plan_prompt = _prompt_manager.queries_from_plan_prompt
analysis_prompt = _prompt_manager.analysis_prompt
multi_query_gen_prompt = _prompt_manager.multi_query_gen_prompt
query_optimization_prompt = _prompt_manager.query_optimization_prompt
remove_irrelevant_content_prompt = _prompt_manager.remove_irrelevant_content_prompt
rerun_prompt = _prompt_manager.rerun_prompt

