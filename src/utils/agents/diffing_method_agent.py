from .blackbox_agent import BlackboxAgent
from typing import Any, Dict, List, Callable
from abc import abstractmethod


class DiffingMethodAgent(BlackboxAgent):
    name: str
    first_user_message_description: str
    tool_descriptions: str
    additional_conduct: str
    interaction_examples: List[str]

    @abstractmethod
    def get_method_tools(self, method: "DiffingMethod") -> Dict[str, Callable[..., Any]]:
        raise NotImplementedError

    def get_tools(self, method: "DiffingMethod") -> Dict[str, Callable[..., Any]]:
        return super().get_tools(method).update(self.get_method_tools(method))

    def get_first_user_message_description(self) -> str:
        return self.first_user_message_description
    
    def get_tool_descriptions(self) -> str:
        return super().get_tool_descriptions() + self.tool_descriptions

    def get_additional_conduct(self) -> str:
        return self.additional_conduct

    def get_interaction_examples(self) -> List[str]:
        return self.interaction_examples
