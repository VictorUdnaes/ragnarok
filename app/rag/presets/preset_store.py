from typing import Dict, Type, Callable, Any
from rag.presets.abstract.abstract_step_preset import AbstractStepPreset

class PresetStore:
    """Registry for preset classes with decorator-based registration"""
    _registry: Dict[str, Type[AbstractStepPreset]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register preset classes"""
        def decorator(preset_class: Type[AbstractStepPreset]):
            cls._registry[name] = preset_class
            return preset_class
        return decorator
    
    @classmethod
    def get(cls, name: str, llm=None, query: str = "", vectorstore=None, **kwargs) -> AbstractStepPreset:
        """Create a preset instance by name"""
        if name not in cls._registry:
            raise ValueError(f"Unknown preset: {name}. Available presets: {list(cls._registry.keys())}")
        
        preset_class = cls._registry[name]
        # Create instance with llm and query
        instance = preset_class()
        instance.setLLM(llm)
        instance.setQuery(query)
        instance.setVectorStore(vectorstore)

        # Apply any additional configuration from kwargs
        for key, value in kwargs.items():
            if hasattr(instance, f"set{key.title()}"):
                getattr(instance, f"set{key.title()}")(value)
            elif hasattr(instance, key):
                setattr(instance, key, value)
        
        return instance
    
    @classmethod
    def list_presets(cls) -> list[str]:
        """List all registered preset names"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a preset is registered"""
        return name in cls._registry

# Create singleton instance
preset_store = PresetStore()