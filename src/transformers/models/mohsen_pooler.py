import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class ReprsPooling(ABC, nn.Module):
    def __init__(self):
        super(ReprsPooling, self).__init__()

    @abstractmethod
    def forward(self, last_hidden_states, attention_mask):
        """
        input:
            hidden_states: [batch_size, max_span_len, embedding_dim] ~ [16, 9, 768]
            attention_mask: [batch_size, max_span_len] ~[16, 9]
        returns:
            [16, 768]
        """
        raise NotImplementedError

class ReprsPoolingAvg(ReprsPooling):
    def forward(self, last_hidden_states, attention_mask):
        # print("MOHSEN AVG")
        # print("hidden_states", last_hidden_states.shape)
        # print("attention_mask", attention_mask.shape)

        # span_masks_shape = attention_mask.shape
        # span_masks = attention_mask.reshape(
        #     span_masks_shape[0],
        #     span_masks_shape[1],
        #     1
        # ).expand_as(last_hidden_states)
        # attention_spans = last_hidden_states * span_masks
        # sum = torch.sum(attention_spans, dim=-2)  # ~[9, 768]
        # num_words_in_batch = torch.count_nonzero(attention_mask, dim=-1).reshape(-1, 1).expand_as(sum) # ~[16, 768]
        # avg_span_repr2 = sum / num_words_in_batch

        # From https://github.com/UKPLab/sentence-transformers
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        avg_span_repr = sum_embeddings / sum_mask

        # assert(torch.allclose(avg_span_repr2, avg_span_repr, atol=1e-07))
        # print(last_hidden_states[:, :, :5])
        # print(avg_span_repr[:, :5])
        # print(avg_span_repr2[:, :5])
        return avg_span_repr

def get_pooling_module(method="avg"):
    if method == "cls":
        raise NotImplementedError
    elif method == "avg":
        return ReprsPoolingAvg()
    else:
        raise Exception("Unknown Pooling Method!")