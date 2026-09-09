"""
Single source of truth for the prompt format used everywhere in this project:
data prep, the Unsloth training notebook, evaluation, and inference.

Keeping this in one place matters: if the prompt used at training time drifts
even slightly from the prompt used at inference/eval time, the fine-tuned
model's performance looks artificially bad. Every script imports from here.
"""

FIELDS = ["company", "date", "address", "total"]

INSTRUCTION = (
    "Extract the following fields from the receipt text below and return "
    "ONLY a valid JSON object with keys \"company\", \"date\", \"address\", "
    "and \"total\". If a field is not present in the text, use null for "
    "that field. Do not include any text other than the JSON object."
)

# Alpaca-style template. This is the format Unsloth's example notebooks use
# by default, which is why we match it here rather than inventing our own.
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""


def build_prompt(input_text: str) -> str:
    """The prompt sent to the model (training input and inference input)."""
    return PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input_text=input_text.strip())


def build_training_example(input_text: str, target_json_str: str) -> str:
    """Full text (prompt + target) used as a training example for SFTTrainer."""
    return build_prompt(input_text) + target_json_str.strip()
