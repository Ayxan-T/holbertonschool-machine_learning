#!/usr/bin/env python3
"""Module: 0-dataset"""

import transformers
from setup import load_pt2en


class Dataset:
    """Class: Dataset"""
    def __init__(self):
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt = None
        self.tokenizer_en = None

    def tokenize_dataset(self, data):
        """Create sub-word tokenizers for the Portuguese and English text."""

        def _decode_sentence(sentence):
            """Decode a tensor or bytes value into a Python string."""
            return sentence.numpy().decode('utf-8')

        def _portuguese_sentences():
            """Yield the Portuguese sentences from the dataset."""
            for pt, _ in data:
                yield _decode_sentence(pt)

        def _english_sentences():
            """Yield the English sentences from the dataset."""
            for _, en in data:
                yield _decode_sentence(en)

        # Start from pretrained BERT tokenizers and fine-tune them on the
        # dataset with a maximum vocabulary size of 2**13.
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            _portuguese_sentences(),
            vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            _english_sentences(),
            vocab_size=2**13
        )

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        return tokenizer_pt, tokenizer_en
