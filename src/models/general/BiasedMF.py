# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn

from models.BaseModel import GeneralModel


class BiasedMF(GeneralModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        super().__init__(args, corpus)

    def _define_params(self):
        self.mf_u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.mf_i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.mf_u_bias_embeddings = nn.Embedding(self.user_num, 1)
        self.mf_i_bias_embeddings = nn.Embedding(self.item_num, 1)
        self.glob_bias = torch.rand(1).cuda()

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        mf_u_vectors = self.mf_u_embeddings(u_ids)
        mf_i_vectors = self.mf_i_embeddings(i_ids)
        mf_u_bias = self.mf_u_bias_embeddings(u_ids)
        mf_i_bias = self.mf_i_bias_embeddings(i_ids)

        prediction = (mf_u_vectors * mf_i_vectors).sum(dim=-1) \
                     + mf_u_bias.view(feed_dict['batch_size'], -1) + mf_i_bias.view(feed_dict['batch_size'], -1) \
                     + self.glob_bias
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}
