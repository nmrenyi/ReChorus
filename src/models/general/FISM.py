# -*- coding: UTF-8 -*-
import numpy as np
import torch.nn as nn
import torch
from models.BaseModel import SequentialModel
from typing import List


class FISM(SequentialModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        super().__init__(args, corpus)

    def _define_params(self):
        self.p_matrix = nn.Embedding(self.item_num, self.emb_size)
        self.q_matrix = nn.Embedding(self.item_num, self.emb_size)
        self.u_bias = nn.Embedding(self.user_num, 1)
        self.i_bias = nn.Embedding(self.item_num, 1)

    def forward(self, feed_dict):
        self.check_list = []

        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        # TODO: FISM is a sequential model, which extract user_rated_item from the user's history
        user_rated_item = feed_dict['user_rated_item']  # List[np.array]

        u_bias = self.u_bias(u_ids)  # [batch_size, 1]
        i_bias = self.i_bias(i_ids).squeeze(dim=-1)  # [batch_size, items]

        user_rated_emb = torch.stack([self.p_matrix(index.to('cuda:0')).sum(dim=0) for index in user_rated_item])  # [batch_size, emb_size]
        current_q_item = self.q_matrix(i_ids)  # [batch_size, items, emb_size]
        current_p_item = self.p_matrix(i_ids)  # [batch_size, items, emb_size]

        # TODO: for negative items, there shouldn't be current_p_item * current_q_item
        prediction = u_bias + i_bias + (user_rated_emb[:, None, :] * current_q_item).sum(dim=-1) - (
                    current_p_item * current_q_item).sum(dim=-1)

        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

    class Dataset(SequentialModel.Dataset):
        user_rated_item_key = 'user_rated_item'

        def _get_feed_dict(self, index: int):
            feed_dict = super()._get_feed_dict(index)
            feed_dict[self.user_rated_item_key] = np.array(list(self.corpus.user_clicked_set[feed_dict['user_id']]))
            return feed_dict

        # Collate a batch according to the list of feed dicts
        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            feed_dict = super().collate_batch(feed_dicts)
            feed_dict[self.user_rated_item_key] = [torch.from_numpy(d[self.user_rated_item_key]) for d in feed_dicts]
            return feed_dict
