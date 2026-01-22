from typing import Protocol, TypedDict, TypeAlias, Literal, NotRequired
from PIL import Image


class VQASample(TypedDict):
    image: Image.Image
    question: str
    answer: str
    answers: NotRequired[list[str]]


Role: TypeAlias = Literal["user", "assistant", "system"]
ChatMessage: TypeAlias = dict[str, str]  # {"role": "...", "content": "..."}
ChatConversation: TypeAlias = list[ChatMessage]


class InstructSample(TypedDict):
    image: Image.Image
    conversation: ChatConversation


class ChatTokenizer(Protocol):
    """
    Specific to Qwen2.5 Instruct tokenizer usage:
    - must support apply_chat_template(tokenize=True)
    """

    pad_token_id: int | None
    eos_token_id: int | None
    pad_token: str | None
    eos_token: str | None

    def apply_chat_template(
        self,
        conversation: ChatConversation,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_tensors: None = None,
    ) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...
