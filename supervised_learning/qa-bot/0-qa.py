#!/usr/bin/env python3
"""Module: 0-qa"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

def question_answer(question, reference):
    # 1. Load the required Hugging Face tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    
    # 2. Load the required TF Hub model
    model_url = "https://tfhub.dev/tensorflow/bert-uncased-tf2-qa/1"
    model = hub.KerasLayer(model_url, trainable=False)
    
    # 3. Tokenize the question and reference text together
    inputs = tokenizer(
        question, 
        reference, 
        return_tensors="tf", 
        padding=False, 
        truncation=True
    )
    
    # 4. Extract the exact arrays the TF Hub model demands
    # TF Hub expects a list of 3 tensors: [ids, mask, segment_ids]
    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_type_ids = inputs['token_type_ids']
    
    # 5. Pass the list of tensors into the TF Hub model
    # The model outputs a list containing two items: [start_logits, end_logits]
    outputs = model([input_word_ids, input_mask, input_type_ids])
    start_logits = outputs[0]
    end_logits = outputs[1]
    
    # 6. Find the word index positions with the highest scores
    # We remove the batch dimension by taking index 0
    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0]
    
    # 7. Check if an answer was actually found
    # If the start index points to token 0 ([CLS]), no answer exists
    if start_idx == 0 or end_idx < start_idx:
        return None
        
    # 8. Pull out the answer tokens and decode them back into human text
    answer_tokens = input_word_ids[0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_tokens)
    
    # 9. Clean up extra spaces and format the output
    answer = answer.strip()
    
    return answer if answer else None
