# models.py: A swappable model handler.

from abc import ABC, abstractmethod
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer

class LanguageModel(ABC):
    """Abstract base class for a swappable language model."""
    @abstractmethod
    def get_logits(self, token_ids):
        """
        Gets the logits for the next token prediction.

        Args:
            token_ids (torch.Tensor): A tensor of token IDs representing the input prompt.

        Returns:
            torch.Tensor: A tensor of logits for the next token.
        """
        pass

    @property
    @abstractmethod
    def tokenizer(self):
        """Returns the tokenizer instance."""
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        """Returns the vocabulary size."""
        pass

    @property
    @abstractmethod
    def device(self):
        """Returns the device the model is on (e.g. 'cpu', 'cuda')."""
        pass


class GPT2Model(LanguageModel):
    """A concrete implementation for the GPT-2 model."""
    def __init__(self, model_name='gpt2'):
        """
        Initializes the GPT-2 model and tokenizer.

        Args:
            model_name (str): The name of the GPT-2 model variant to load.
        """
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self._device}")
        
        self._model = GPT2LMHeadModel.from_pretrained(model_name).to(self._device)
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token # Set pad token for batching if needed

    def get_logits(self, token_ids):
        """Gets logits from the GPT-2 model."""
        with torch.no_grad():
            outputs = self._model(token_ids)
            # We only need the logits for the very last token in the sequence
            logits = outputs.logits[:, -1, :]
        return logits

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    @property
    def device(self):
        return self._device

class GptOssModel(LanguageModel):
    """A concrete implementation for the openai/gpt-oss-20b model."""
    def __init__(self, model_name='openai/gpt-oss-20b'):
        """
        Initializes the gpt-oss-20b model and tokenizer.
        Uses device_map="auto" to handle the large model size.
        """
        print("Loading gpt-oss-20b model. This may take a while and requires significant memory...")
        
        # Use bfloat16 for memory efficiency on compatible hardware
        self._torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._torch_dtype,
            device_map="auto", # Automatically distributes layers across devices (GPU/CPU)
            trust_remote_code=True # Often required for custom model architectures
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # The primary device is where the output logits will be, which device_map handles.
        self._device = self._model.device

    def get_logits(self, token_ids):
        """Gets logits from the gpt-oss-20b model."""
        with torch.no_grad():
            outputs = self._model(token_ids)
            logits = outputs.logits[:, -1, :]
        return logits

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    @property
    def device(self):
        return self._device

class GptOss120bModel(LanguageModel):
    """A concrete implementation for the openai/gpt-oss-120b model."""
    def __init__(self, model_name='openai/gpt-oss-120b'):
        """
        Initializes the gpt-oss-120b model and tokenizer.
        """
        print(f"Loading {model_name}. This requires an ~80GB GPU and may take several minutes...")

        self._torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._device = self._model.device

    def get_logits(self, token_ids):
        """Gets logits from the gpt-oss-120b model."""
        with torch.no_grad():
            outputs = self._model(token_ids)
            logits = outputs.logits[:, -1, :]
        return logits

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    @property
    def device(self):
        return self._device


OPT_MODEL_IDS = {
    'opt-125m': 'facebook/opt-125m',
    'opt-1.3b': 'facebook/opt-1.3b',
    'opt-2.7b': 'facebook/opt-2.7b',
    'opt-6.7b': 'facebook/opt-6.7b',
}


class OPTModel(LanguageModel):
    """A concrete implementation for Meta OPT models (no license approval needed)."""
    def __init__(self, model_name: str):
        hf_model_id = OPT_MODEL_IDS.get(model_name, model_name)
        print(f"Loading {hf_model_id}. This may take a while...")

        self._torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=self._torch_dtype,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._device = self._model.device

    def get_logits(self, token_ids):
        with torch.no_grad():
            outputs = self._model(token_ids)
            return outputs.logits[:, -1, :]

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    @property
    def device(self):
        return self._device


DEEPSEEK_MODEL_IDS = {
    'deepseek-llm-7b': 'deepseek-ai/deepseek-llm-7b-base',
}


class DeepSeekModel(LanguageModel):
    """A concrete implementation for DeepSeek LLM models."""
    def __init__(self, model_name: str):
        hf_model_id = DEEPSEEK_MODEL_IDS.get(model_name, model_name)
        print(f"Loading {hf_model_id}. This may take a while...")

        self._torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=self._torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._device = self._model.device

    def get_logits(self, token_ids):
        with torch.no_grad():
            outputs = self._model(token_ids)
            return outputs.logits[:, -1, :]

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    @property
    def device(self):
        return self._device


# Mapping from CLI shorthand to HuggingFace model ID
LLAMA_MODEL_IDS = {
    'llama-3.2-1b': 'meta-llama/Llama-3.2-1B',
    'llama-3.2-3b': 'meta-llama/Llama-3.2-3B',
    'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B',
}


class LlamaModel(LanguageModel):
    """A concrete implementation for Meta LLaMA models."""
    def __init__(self, model_name: str):
        """
        Initializes a LLaMA model and tokenizer.

        Args:
            model_name (str): CLI shorthand (e.g. 'llama-3.2-1b') or a full
                HuggingFace model ID (e.g. 'meta-llama/Llama-3.2-1B').
        """
        hf_model_id = LLAMA_MODEL_IDS.get(model_name, model_name)
        print(f"Loading {hf_model_id}. This may take a while...")
        print("Note: LLaMA models require a HuggingFace account and accepted Meta license.")
        print("      Run 'huggingface-cli login' first if you have not already.")

        self._torch_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=self._torch_dtype,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

        # LLaMA has no dedicated pad token; reuse EOS so batch encoding works.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._device = self._model.device

    def get_logits(self, token_ids):
        """Gets logits from the LLaMA model."""
        with torch.no_grad():
            outputs = self._model(token_ids)
            return outputs.logits[:, -1, :]

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    @property
    def device(self):
        return self._device
