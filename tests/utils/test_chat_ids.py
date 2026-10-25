from unittest.mock import MagicMock

import pytest

from diffing.utils.prompts import chat_ids


def test_chat_ids_forces_list_output():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    messages = [{"role": "user", "content": "hi"}]

    assert chat_ids(tokenizer, messages, add_generation_prompt=True) == [1, 2, 3]
    tokenizer.apply_chat_template.assert_called_once_with(
        messages, tokenize=True, return_dict=False, add_generation_prompt=True
    )


def test_chat_ids_rejects_non_list():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {"input_ids": [1, 2, 3]}

    with pytest.raises(AssertionError, match="Expected list of token ids"):
        chat_ids(tokenizer, [{"role": "user", "content": "hi"}])
