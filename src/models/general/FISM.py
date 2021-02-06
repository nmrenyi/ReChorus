# -*- coding: UTF-8 -*-
import numpy as np
import torch.nn as nn

from models.BaseModel import GeneralModel


class FISM(GeneralModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

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
        user_rated_item = feed_dict['user_rated_item']

        return {'prediction': i_ids.view(feed_dict['batch_size'], -1)}

    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index: int):
            feed_dict = super()._get_feed_dict(index)
            feed_dict['user_rated_item'] = np.array(list(self.corpus.user_clicked_set[feed_dict['user_id']]))
            return feed_dict
