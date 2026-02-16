from typing import Any, Protocol, TypeVar, TypedDict, TypeAlias, Literal, NotRequired
from PIL import Image
import torch


class VQASample(TypedDict):
    image: Image.Image
    question: str
    answer: str
    answers: NotRequired[list[str]]


Role: TypeAlias = Literal["user", "assistant", "system"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class InstructSample(TypedDict):
    image: Image.Image
    conversation: list[ChatMessage]


class CollatorBatch(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_labels: torch.Tensor
    mm_attention_mask: torch.Tensor


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
        conversation: list[ChatMessage],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool = False,
        return_tensors: None = None,
    ) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...

    def batch_decode(self, token_ids: list[list[int]], *, skip_special_tokens: bool) -> list[str]: ...


T_co = TypeVar("T_co", covariant=True)


class SizedDataset(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...

    def __len__(self) -> int: ...
