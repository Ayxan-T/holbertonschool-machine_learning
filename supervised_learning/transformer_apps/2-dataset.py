#!/usr/bin/env python3
"""Module: 1-dataset"""

import transformers
from setup import load_pt2en
import tensorflow as tf


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

    def encode(self, pt, en):
        """Encodes a Portuguese and English translation pair into token IDs

        with start and end sequence tokens.
        """
        # Extract string values from tf.Tensor bytes objects
        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        # Define start and end token IDs based on vocab size
        start_token = self.tokenizer_pt.vocab_size
        end_token = self.tokenizer_pt.vocab_size + 1

        # Tokenize the Portuguese sentence (without default special tokens)
        pt_tokens = [start_token] + self.tokenizer_pt.encode(
            pt_str, add_special_tokens=False
        ) + [end_token]

        # Tokenize the English sentence (without default special tokens)
        en_tokens = [start_token] + self.tokenizer_en.encode(
            en_str, add_special_tokens=False
        ) + [end_token]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        # 1. Runs Python encode logic and wraps python lists into tf.Tensors
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        
        # 2. Tell TensorFlow: "These are 1D Tensors with dynamic sequence length"
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        
        return pt_encoded, en_encoded