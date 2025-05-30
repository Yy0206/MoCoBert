import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer

from MoCoBert_dataloader import load_cls_data, TrainDataset, TestDataset, CollateFunc
from MoCoBert_Model import ESimcseModel, MomentumEncoder, MultiNegativeRankingLoss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 计算概率p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_bert_input(source, device):
    return {k: v.to(device) for k, v in source.items()}


def train(model, momentum_encoder, train_dl, dev_dl, optimizer, contrast_loss_fn, device, save_path, gamma=0.95,
          focal_alpha=0.85,
          focal_gamma=2):
    focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)

    best_f1 = 0
    for epoch in range(5):  # 训练epoch数
        model.train()
        for (src_tokens, pos_tokens, neg_tokens), labels in tqdm(train_dl):
            # 获取数据
            labels = labels.to(device)
            src_input = get_bert_input(src_tokens, device)
            pos_input = get_bert_input(pos_tokens, device)
            neg_input = get_bert_input(neg_tokens, device) if neg_tokens else None

            # 前向传播
            src_emb, src_logits = model(**src_input)
            pos_emb, _ = model(**pos_input)
            neg_emb = momentum_encoder(**neg_input)[0] if neg_input else None

            # 计算损失
            # cls_loss = F.cross_entropy(src_logits, labels)
            cls_loss = focal_loss_fn(src_logits, labels)
            contrast_loss = contrast_loss_fn.multi_negative_ranking_loss(src_emb, pos_emb, neg_emb)
            total_loss = cls_loss + contrast_loss  # 调整权重

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 更新动量编码器
            with torch.no_grad():
                for param, m_param in zip(model.parameters(), momentum_encoder.parameters()):
                    m_param.data = gamma * m_param.data + (1 - gamma) * param.data


        if results['macro_f1'] > best_f1:
            best_f1 = results['macro_f1']
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved with F1: {best_f1:.4f}")



def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_data = load_cls_data(args.train_path)
    test_data = load_cls_data(args.test_path)

    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    model = ESimcseModel(args.pretrain_model_path, num_classes=8, pooling=args.pooler).to(device)
    momentum_encoder = MomentumEncoder(args.pretrain_model_path, num_classes=8, pooling=args.pooler).to(device)

    # 数据加载
    train_dataset = TrainDataset(train_data)
    train_collate = CollateFunc(tokenizer, max_len=args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=train_collate, num_workers=8, pin_memory=True
    )

    test_dataset = TestDataset(test_data, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    # 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    contrast_loss_fn = MultiNegativeRankingLoss()

    # 训练
    train(model, momentum_encoder, train_loader, test_loader,
          optimizer, contrast_loss_fn, device, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./cls_model.bin")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--train_path", type=str, default="./data/train.txt")
    parser.add_argument("--test_path", type=str, default="./data/test.txt")
    parser.add_argument("--pretrain_model_path", type=str, default="bert-base-chinese")
    parser.add_argument("--pooler", type=str, default="first-last-avg")
    args = parser.parse_args()

    logger.add("training.log")
    main(args)
