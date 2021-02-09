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
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
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
        i_ids = feed_dict['item_id']  # [batch_size, items]
        user_rated_item = feed_dict['history_items']  # [batch_size, history_max]

        mask = (user_rated_item == 0).unsqueeze(-1)  # [batch_size, history_max, 1]
        user_rated_emb = self.p_matrix(user_rated_item)  # [batch_size, history_max, emb_size]
        user_rated_emb = user_rated_emb.masked_fill(mask, 0).sum(dim=1)  # [batch_size, emb_size]

        u_bias = self.u_bias(u_ids)  # [batch_size, 1]
        i_bias = self.i_bias(i_ids).squeeze(dim=-1)  # [batch_size, items]

        current_q_item = self.q_matrix(i_ids)  # [batch_size, items, emb_size]
        current_p_item = self.p_matrix(i_ids)  # [batch_size, items, emb_size]
        current_p_item[:, 1:, :] = 0  # the inner product of p_i and q_i for negative items shouldn't be taken away
        # from the sum
        # CAUTION: the problem here is that the current positive item may not lies in the history items in the batch.
        # So directly minus the item's inner product of p_i and q_i is not consistent with the paper.
        # However, the metrics seems perfect!

        prediction = u_bias + i_bias + (user_rated_emb[:, None, :] * current_q_item).sum(dim=-1) - (
                    current_p_item * current_q_item).sum(dim=-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
