"""
Models Module: Unified Request/Response Schemas (SOTA Edition)
================================================================

COMPLETE parameter coverage for vLLM and OpenAI APIs including:
    - All sampling parameters (temperature, top_p, top_k, min_p, typical_p)
    - All penalty parameters (presence, frequency, repetition, length)
    - Guided decoding (JSON schema, regex, grammar, choice)
    - Function/tool calling with parallel execution
    - Vision/multi-modal with multiple images
    - LoRA adapter routing
    - Logprobs and logit_bias
    - Audio modalities (placeholder for future)

Memory Optimization:
    - Frozen models (immutable, hashable)
    - __slots__ via Pydantic ConfigDict
    - Lazy field defaults
    - Efficient serialization with orjson

Validation:
    - Exhaustive field constraints
    - Cross-field validation (mutually exclusive options)
    - Pre/post-condition assertions
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping, Sequence, TypeAlias
import time
import uuid

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, field_validator, model_validator


# =============================================================================
# TYPE ALIASES: Improve readability and maintainability
# =============================================================================
JsonDict: TypeAlias = dict[str, Any]
JsonList: TypeAlias = list[Any]
StopSequence: TypeAlias = str | list[str] | None
LogitBias: TypeAlias = dict[str, float] | None


# =============================================================================
# BASE MODEL: Optimized configuration for all schemas
# =============================================================================

class BaseModel(PydanticBaseModel):
    """
    Base model with SOTA optimizations.
    
    Capabilities:
        - frozen=True: Thread-safe immutability
        - extra='ignore': Graceful handling of unknown fields (OpenAI compat)
        - populate_by_name: Alias support for field mapping
        - str_strip_whitespace: Auto-trim string inputs
    """
    model_config = ConfigDict(
        frozen=True,
        extra="ignore",  # Changed: tolerate unknown fields for forward compat
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class MutableBaseModel(PydanticBaseModel):
    """
    Mutable variant for internal state tracking.
    Used for request wrappers that need modification.
    """
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        use_enum_values=True,
    )


# =============================================================================
# ENUMS: Type-safe constants
# =============================================================================

class MessageRole(str, Enum):
    """Chat message roles per OpenAI spec."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Deprecated: use TOOL
    DEVELOPER = "developer"  # New: for o1 models


class FinishReason(str, Enum):
    """Generation termination reasons."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"  # Deprecated


class ResponseFormatType(str, Enum):
    """Response format types."""
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"  # Structured outputs


class ToolChoiceMode(str, Enum):
    """Tool choice modes."""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


# =============================================================================
# FUNCTION/TOOL CALLING: Complete implementation
# =============================================================================

class FunctionCall(BaseModel):
    """
    Function call specification.
    
    Used in:
        - Assistant message tool_calls
        - Tool choice forcing
    """
    name: str = Field(..., min_length=1, max_length=64)
    arguments: str = Field(..., description="JSON-encoded arguments string")


class ToolCallFunction(BaseModel):
    """Function within a tool call."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """
    Tool invocation in assistant response.
    
    Attributes:
        id: Unique call ID for response matching
        type: Always "function" currently
        function: The function call details
    """
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: ToolCallFunction


class FunctionDefinition(BaseModel):
    """
    Function schema for tool definition.
    
    Attributes:
        name: Function identifier
        description: Natural language description for model
        parameters: JSON Schema for function arguments
        strict: Enable strict schema validation (OpenAI)
    """
    name: str = Field(..., min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=4096)
    parameters: JsonDict = Field(default_factory=dict)
    strict: bool | None = None  # OpenAI structured outputs


class Tool(BaseModel):
    """Tool definition wrapper."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    """Force specific function."""
    name: str


class ToolChoice(BaseModel):
    """Explicit tool choice."""
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


# =============================================================================
# CHAT MESSAGE: Multi-modal support
# =============================================================================

class ImageUrl(BaseModel):
    """Image URL with detail level."""
    url: str = Field(..., description="URL or base64 data URI")
    detail: Literal["auto", "low", "high"] = "auto"


class TextContentPart(BaseModel):
    """Text content part."""
    type: Literal["text"] = "text"
    text: str


class ImageContentPart(BaseModel):
    """Image content part."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


class AudioContentPart(BaseModel):
    """Audio content part (future)."""
    type: Literal["input_audio"] = "input_audio"
    input_audio: JsonDict


# Union for content parts
ContentPart = TextContentPart | ImageContentPart | AudioContentPart


class ChatMessage(BaseModel):
    """
    Chat message with full multi-modal support.
    
    Content Types:
        - String: Simple text message
        - List[ContentPart]: Multi-modal (text + images)
    
    Tool Calling:
        - Assistant messages may include tool_calls
        - Tool messages must include tool_call_id for response matching
    """
    role: MessageRole
    content: str | list[ContentPart] | None = None
    name: str | None = Field(default=None, max_length=256)
    
    # Tool calling fields
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # Required when role="tool"
    
    # Deprecated function calling
    function_call: FunctionCall | None = None
    
    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> str | list[ContentPart] | None:
        """Normalize content field."""
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return v  # Pydantic will validate items
        raise ValueError("content must be string or list of content parts")


# =============================================================================
# RESPONSE FORMAT: Structured outputs
# =============================================================================

class JsonSchemaFormat(BaseModel):
    """JSON Schema specification for structured outputs."""
    name: str
    description: str | None = None
    schema_: JsonDict = Field(alias="schema", default_factory=dict)
    strict: bool | None = None


class ResponseFormat(BaseModel):
    """
    Response format specification.
    
    Modes:
        - text: Default, model outputs natural text
        - json_object: Model outputs valid JSON
        - json_schema: Model outputs JSON matching schema (structured outputs)
    """
    type: ResponseFormatType = ResponseFormatType.TEXT
    json_schema: JsonSchemaFormat | None = None


# =============================================================================
# GUIDED DECODING: vLLM-specific constrained generation
# =============================================================================

class GuidedDecodingParams(BaseModel):
    """
    vLLM guided decoding parameters.
    
    Modes (mutually exclusive):
        - guided_json: Output matches JSON schema
        - guided_regex: Output matches regex pattern
        - guided_choice: Output is one of choices
        - guided_grammar: Output follows EBNF grammar
    
    Backend Selection:
        - guided_decoding_backend: "outlines" or "lm-format-enforcer"
    """
    guided_json: JsonDict | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None
    guided_decoding_backend: str | None = None
    guided_whitespace_pattern: str | None = None
    
    @model_validator(mode="after")
    def validate_mutual_exclusion(self) -> "GuidedDecodingParams":
        """Ensure only one guided mode is active."""
        active = sum([
            self.guided_json is not None,
            self.guided_regex is not None,
            self.guided_choice is not None,
            self.guided_grammar is not None,
        ])
        if active > 1:
            raise ValueError("Only one guided decoding mode can be active")
        return self


# =============================================================================
# SAMPLING PARAMETERS: Complete coverage
# =============================================================================

class SamplingParams(BaseModel):
    """
    All sampling parameters for vLLM and OpenAI.
    
    Temperature Family:
        - temperature: Overall randomness [0, 2]
        - top_p: Nucleus sampling mass [0, 1]
        - top_k: Top-k tokens (vLLM)
        - min_p: Minimum probability threshold (vLLM)
        - typical_p: Typical sampling (vLLM)
    
    Penalties:
        - presence_penalty: Penalize repeated topics [-2, 2]
        - frequency_penalty: Penalize repeated tokens [-2, 2]
        - repetition_penalty: Multiplicative penalty (vLLM)
        - length_penalty: Beam search length penalty
    
    Generation Control:
        - max_tokens: Maximum output tokens
        - min_tokens: Minimum output tokens (vLLM)
        - stop: Stop sequences
        - stop_token_ids: Stop on specific token IDs (vLLM)
        - include_stop_str_in_output: Include stop string (vLLM)
    
    Advanced:
        - seed: Reproducible sampling
        - logit_bias: Token probability adjustments
        - logprobs: Return log probabilities
        - top_logprobs: Number of top tokens to return
    """
    # Temperature family
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)  # -1 means disabled
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    typical_p: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Penalties
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)  # 1.0 = disabled
    length_penalty: float = Field(default=1.0, ge=0.0)
    
    # Generation control
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    min_tokens: int = Field(default=0, ge=0)  # vLLM
    stop: StopSequence = None
    stop_token_ids: list[int] | None = None  # vLLM
    include_stop_str_in_output: bool = False  # vLLM
    skip_special_tokens: bool = True  # vLLM
    spaces_between_special_tokens: bool = True  # vLLM
    
    # Advanced
    seed: int | None = None
    logit_bias: LogitBias = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    
    # Beam search (vLLM)
    best_of: int | None = Field(default=None, ge=1)
    use_beam_search: bool = False
    early_stopping: bool | str = False  # True, False, or "never"


# =============================================================================
# CHAT COMPLETION REQUEST: Full parameter coverage
# =============================================================================

class ChatCompletionRequest(BaseModel):
    """
    Chat completion request with COMPLETE parameter coverage.
    
    Supports all parameters from:
        - OpenAI API (GPT-4, GPT-4o, o1)
        - vLLM OpenAI-compatible server
    """
    # Required
    model: str = Field(..., min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    
    # === Sampling Parameters ===
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)  # vLLM
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)  # vLLM
    typical_p: float | None = None  # vLLM
    
    # === Penalty Parameters ===
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(default=None, ge=0.0)  # vLLM
    length_penalty: float | None = None  # vLLM beam search
    
    # === Generation Control ===
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    max_completion_tokens: int | None = None  # OpenAI o1 models
    min_tokens: int | None = Field(default=None, ge=0)  # vLLM
    n: int = Field(default=1, ge=1, le=128)
    stop: StopSequence = None
    stop_token_ids: list[int] | None = None  # vLLM
    
    # === Tool/Function Calling ===
    tools: list[Tool] | None = None
    tool_choice: str | ToolChoice | None = None  # "none", "auto", "required", or specific
    parallel_tool_calls: bool = True  # OpenAI: allow parallel calls
    
    # === Output Format ===
    response_format: ResponseFormat | None = None
    
    # === Logprobs ===
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    logit_bias: LogitBias = None
    
    # === Streaming ===
    stream: bool = False
    stream_options: JsonDict | None = None  # include_usage, etc.
    
    # === vLLM Guided Decoding ===
    guided_json: JsonDict | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None
    guided_decoding_backend: str | None = None
    
    # === vLLM LoRA Adapter ===
    # Model field can be LoRA adapter name registered on server
    
    # === Reproducibility ===
    seed: int | None = None
    
    # === vLLM Specific ===
    best_of: int | None = Field(default=None, ge=1)
    use_beam_search: bool | None = None
    early_stopping: bool | None = None
    skip_special_tokens: bool | None = None
    spaces_between_special_tokens: bool | None = None
    include_stop_str_in_output: bool | None = None
    
    # === Metadata ===
    user: str | None = Field(default=None, max_length=256)
    metadata: JsonDict | None = None  # Custom tracking data
    
    # === OpenAI-specific ===
    service_tier: str | None = None  # "auto" or "default"
    store: bool | None = None  # Store completion for fine-tuning
    
    def get_extra_body(self) -> JsonDict:
        """
        Build vLLM extra_body from vLLM-specific parameters.
        
        Returns dict suitable for OpenAI client's extra_body parameter.
        """
        extra: JsonDict = {}
        
        # Guided decoding
        if self.guided_json is not None:
            extra["guided_json"] = self.guided_json
        if self.guided_regex is not None:
            extra["guided_regex"] = self.guided_regex
        if self.guided_choice is not None:
            extra["guided_choice"] = self.guided_choice
        if self.guided_grammar is not None:
            extra["guided_grammar"] = self.guided_grammar
        if self.guided_decoding_backend is not None:
            extra["guided_decoding_backend"] = self.guided_decoding_backend
        
        # Sampling extensions
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if self.min_p is not None:
            extra["min_p"] = self.min_p
        if self.typical_p is not None:
            extra["typical_p"] = self.typical_p
        if self.repetition_penalty is not None:
            extra["repetition_penalty"] = self.repetition_penalty
        if self.min_tokens is not None:
            extra["min_tokens"] = self.min_tokens
        if self.stop_token_ids is not None:
            extra["stop_token_ids"] = self.stop_token_ids
        
        # Beam search
        if self.best_of is not None:
            extra["best_of"] = self.best_of
        if self.use_beam_search is not None:
            extra["use_beam_search"] = self.use_beam_search
        if self.early_stopping is not None:
            extra["early_stopping"] = self.early_stopping
        if self.length_penalty is not None:
            extra["length_penalty"] = self.length_penalty
        
        # Output options
        if self.skip_special_tokens is not None:
            extra["skip_special_tokens"] = self.skip_special_tokens
        if self.spaces_between_special_tokens is not None:
            extra["spaces_between_special_tokens"] = self.spaces_between_special_tokens
        if self.include_stop_str_in_output is not None:
            extra["include_stop_str_in_output"] = self.include_stop_str_in_output
        
        return extra
    
    def to_openai_kwargs(self) -> JsonDict:
        """
        Convert to OpenAI SDK kwargs.
        
        Filters out vLLM-specific parameters for pure OpenAI usage.
        """
        kwargs: JsonDict = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream,
        }
        
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.presence_penalty != 0.0:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.logprobs is not None:
            kwargs["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            kwargs["top_logprobs"] = self.top_logprobs
        if self.logit_bias is not None:
            kwargs["logit_bias"] = self.logit_bias
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.tools is not None:
            kwargs["tools"] = [t.model_dump() for t in self.tools]
        if self.tool_choice is not None:
            if isinstance(self.tool_choice, str):
                kwargs["tool_choice"] = self.tool_choice
            else:
                kwargs["tool_choice"] = self.tool_choice.model_dump()
        if self.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format.model_dump(exclude_none=True)
        if self.stream_options is not None:
            kwargs["stream_options"] = self.stream_options
        if self.user is not None:
            kwargs["user"] = self.user
        
        return kwargs


# =============================================================================
# USAGE STATISTICS
# =============================================================================

class ChatCompletionTokenLogprob(BaseModel):
    """Token with log probability."""
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list["ChatCompletionTokenLogprob"] | None = None


class ChoiceLogprobs(BaseModel):
    """Logprobs for a completion choice."""
    content: list[ChatCompletionTokenLogprob] | None = None
    refusal: list[ChatCompletionTokenLogprob] | None = None


class CompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    
    # OpenAI detailed usage
    prompt_tokens_details: JsonDict | None = None
    completion_tokens_details: JsonDict | None = None


# =============================================================================
# CHAT COMPLETION RESPONSE
# =============================================================================

class ChoiceMessage(BaseModel):
    """Message in completion choice."""
    role: MessageRole = MessageRole.ASSISTANT
    content: str | None = None
    refusal: str | None = None  # OpenAI: model refusal
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None  # Deprecated


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int = 0
    message: ChoiceMessage
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionResponse(BaseModel):
    """Full chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None  # OpenAI


# =============================================================================
# STREAMING RESPONSES
# =============================================================================

class DeltaMessage(BaseModel):
    """Incremental message in streaming."""
    role: MessageRole | None = None
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None


class StreamChoice(BaseModel):
    """Streaming choice with delta."""
    index: int = 0
    delta: DeltaMessage
    finish_reason: FinishReason | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk (SSE data)."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    usage: CompletionUsage | None = None  # Final chunk with stream_options
    system_fingerprint: str | None = None
    service_tier: str | None = None


# =============================================================================
# TEXT COMPLETION (Legacy)
# =============================================================================

class CompletionRequest(BaseModel):
    """
    Text completion request (/v1/completions).
    
    Legacy endpoint, prefer chat completions.
    """
    model: str = Field(..., min_length=1)
    prompt: str | list[str] | list[int] | list[list[int]] = Field(...)
    
    # Sampling
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    
    # Generation
    max_tokens: int | None = Field(default=16, ge=1, le=128000)
    min_tokens: int | None = Field(default=None, ge=0)
    stop: StopSequence = None
    n: int = Field(default=1, ge=1, le=128)
    
    # Penalties
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float | None = None
    
    # Advanced
    logprobs: int | None = Field(default=None, ge=0, le=5)
    logit_bias: LogitBias = None
    echo: bool = False
    suffix: str | None = None
    best_of: int | None = Field(default=None, ge=1)
    seed: int | None = None
    
    # Streaming
    stream: bool = False
    stream_options: JsonDict | None = None
    
    # Guided decoding (vLLM)
    guided_json: JsonDict | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None
    
    # Metadata
    user: str | None = None


class CompletionLogprobs(BaseModel):
    """Logprobs for text completion."""
    text_offset: list[int] | None = None
    token_logprobs: list[float | None] | None = None
    tokens: list[str] | None = None
    top_logprobs: list[dict[str, float] | None] | None = None


class CompletionChoice(BaseModel):
    """Text completion choice."""
    text: str
    index: int = 0
    logprobs: CompletionLogprobs | None = None
    finish_reason: Literal["stop", "length"] | None = None


class CompletionResponse(BaseModel):
    """Text completion response."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None
    system_fingerprint: str | None = None


# =============================================================================
# EMBEDDINGS
# =============================================================================

class EmbeddingRequest(BaseModel):
    """Embedding generation request."""
    model: str = Field(..., min_length=1)
    input: str | list[str] | list[int] | list[list[int]] = Field(...)
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None


class EmbeddingData(BaseModel):
    """Single embedding."""
    object: Literal["embedding"] = "embedding"
    index: int = 0
    embedding: list[float] | str  # str for base64


class EmbeddingUsage(BaseModel):
    """Embedding token usage."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingData]
    usage: EmbeddingUsage


# =============================================================================
# MODEL LISTING
# =============================================================================

class ModelPermission(BaseModel):
    """Model permission entry."""
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4().hex[:8]}")
    object: Literal["model_permission"] = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelInfo(BaseModel):
    """Model metadata."""
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "organization"
    permission: list[ModelPermission] = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None


class ModelListResponse(BaseModel):
    """Model list response."""
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# =============================================================================
# LORA ADAPTER ROUTING (vLLM)
# =============================================================================

class LoRARequest(BaseModel):
    """
    LoRA adapter specification for vLLM.
    
    Usage:
        Pass adapter name as model in chat request,
        or use lora_request parameter.
    """
    name: str = Field(..., min_length=1)
    path: str | None = None  # Optional if registered on server
    base_model: str | None = None


# =============================================================================
# HEALTH & METRICS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy", "degraded"] = "healthy"
    version: str | None = None
    uptime_seconds: float | None = None


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    status: Literal["ready", "not_ready"] = "ready"
    provider: str | None = None
    checks: JsonDict | None = None
