
import random
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class ESimcseModel(nn.Module):
    def __init__(self, pretrained_model, num_classes, pooling='first-last-avg', dropout=0.3):
        super(ESimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        
        # Get embeddings
        if self.pooling == 'cls':
            embeddings = out.last_hidden_state[:, 0]
        elif self.pooling == 'pooler':
            embeddings = out.pooler_output
        elif self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            embeddings = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            embeddings = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        
        # Classification
        logits = self.classifier(embeddings)
        return embeddings, logits

class MomentumEncoder(ESimcseModel):
    def __init__(self, pretrained_model, num_classes, pooling):
        super(MomentumEncoder, self).__init__(pretrained_model, num_classes, pooling)

class MultiNegativeRankingLoss(nn.Module):
    def __init__(self):
        super(MultiNegativeRankingLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def multi_negative_ranking_loss(self, embed_src, embed_pos, embed_neg, scale=20.0):
        if embed_neg is not None:
            embed_pos = torch.cat([embed_pos, embed_neg], dim=0)

        scores = self.cos_sim(embed_src, embed_pos) * scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)

    def cos_sim(self, a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))
