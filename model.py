'''
Concatenate premise, hypothesis, label and decode one of them (learns to ignore the other).
has to have same number of stacked LSTMs for encoder and decoder for the moment.
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import util

import random
args = util.get_args()


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.d_hidden, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dp_ratio,
                        bidirectional=config.bidir)


    def forward(self, inputs):
        outputs, (ht, ct) = self.rnn(inputs)
        return ht, ct


class SNLIAutoencoder(nn.Module):
    def __init__(self, config):
        super(SNLIAutoencoder, self).__init__()
        self.config = config

        self.embed = nn.Embedding(config.n_vocab, config.d_embed)
        self.projection = nn.Linear(config.d_embed, config.d_hidden) # shared between encoder and decoder
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.encoder = Encoder(config)
        
        # because we concatenate the sentence representations and label, we need to reduce the dimension
        dim_output = 2 * config.d_hidden + 3
        if self.config.bidir:
            dim_output = 4 * config.d_hidden + 3
        self.linear_ht = nn.Linear(dim_output, config.d_hidden)
        self.linear_ct = nn.Linear(dim_output, config.d_hidden)

        self.decoder = nn.LSTM(input_size=config.d_hidden, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dp_ratio,
                        bidirectional=False)

        self.output = nn.Linear(config.d_hidden, config.n_vocab)


    def forward(self, batch, target_input, mode):
        batch_sentence1 = batch.sentence1[0]
        lens_sentence1 = batch.sentence1[1]
        batch_sentence2 = batch.sentence2[0]
        lens_sentence2 = batch.sentence2[1]

        # batch size
        assert(batch_sentence1.size(1) == batch_sentence2.size(1) and target_input.size(1) == batch_sentence1.size(1)) 
        batch_size = batch_sentence1.size(1)

        sent1_embed = self.embed(batch_sentence1)
        sent2_embed = self.embed(batch_sentence2)
        explanation_embed = self.embed(target_input)
        
        if self.config.fix_emb:
            sent1_embed = Variable(sent1_embed.data)
            sent2_embed = Variable(sent2_embed.data)
            explanation_embed = Variable(explanation_embed.data)

        sent1_embed = self.dropout(self.projection(sent1_embed))
        sent2_embed = self.dropout(self.projection(sent2_embed))
        explanation_embed = self.dropout(self.projection(explanation_embed))

        packed_sentence1, sorted_indices_sentence1 = util.create_sequence(sent1_embed, lens_sentence1, False)
        ht_sentence1, ct_sentence1 = self.encoder(packed_sentence1)
        ht_sent1 = util.recover_order_hiddens(ht_sentence1, sorted_indices_sentence1)
        ct_sent1 = util.recover_order_hiddens(ct_sentence1, sorted_indices_sentence1)

        packed_sentence2, sorted_indices_sentence2 = util.create_sequence(sent2_embed, lens_sentence2, False)
        ht_sentence2, ct_sentence2 = self.encoder(packed_sentence2)
        ht_sent2 = util.recover_order_hiddens(ht_sentence2, sorted_indices_sentence2)
        ct_sent2 = util.recover_order_hiddens(ct_sentence2, sorted_indices_sentence2)

        if self.config.bidir:
            ht_sent1 = util.arrange_bidir(ht_sent1)
            ct_sent1 = util.arrange_bidir(ct_sent1)
            ht_sent2 = util.arrange_bidir(ht_sent2)
            ct_sent2 = util.arrange_bidir(ct_sent2)

        one_hot_label = util.one_hot_vector(batch.gold_label - 1, 3).expand(self.config.n_layers, batch_size, 3)

        dec_h0 = self.linear_ht(torch.cat([ht_sent1, ht_sent2, one_hot_label], 2))
        dec_c0 = self.linear_ct(torch.cat([ct_sent1, ct_sent2, one_hot_label], 2))

        if mode == "batch":
            outputs, (ht, ct) = self.decoder(explanation_embed, (dec_h0, dec_c0))
            return self.output(outputs)
        elif mode == "for-output":
            max_T_decoder = target_input.size(0)
            out_t = []
            dec_inp_t = explanation_embed[0, :, :].unsqueeze(0)
            ht = dec_h0
            ct = dec_c0
            for t in range(max_T_decoder):
                dec_out_t, (ht, ct) = self.decoder(dec_inp_t, (ht, ct))
                out_t.append(self.output(dec_out_t))
                i_t = torch.max(out_t[-1], 2)[1]
                dec_inp_t = self.projection(self.embed(i_t))
            final_output = torch.cat(out_t, 0)
            return final_output
        elif mode == "for-output-TF":
            max_T_decoder = target_input.size(0)
            out_t = []
            ht = dec_h0
            ct = dec_c0
            for t in range(max_T_decoder):
                dec_inp_t = explanation_embed[t, :, :].unsqueeze(0)
                dec_out_t, (ht, ct) = self.decoder(dec_inp_t, (ht, ct))
                out_t.append(self.output(dec_out_t))
            final_output = torch.cat(out_t, 0)
            return final_output
        else:
            raise ValueError('The mode should be one of: "batch", "for-output", for-output-TF"')


