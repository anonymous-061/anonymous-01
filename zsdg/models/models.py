# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from zsdg.dataset.corpora import PAD, BOS, EOS, BOD
from zsdg import criterions
from zsdg.enc2dec.decoders import DecoderRNN, DecoderPointerGen
from zsdg.enc2dec.encoders import EncoderRNN, RnnUttEncoder,Semantic_guide,calculate_T
from zsdg.utils import INT, FLOAT, LONG, cast_type
from zsdg.nn_lib import IdentityConnector, Bi2UniConnector
from zsdg import nn_lib
import numpy as np
from zsdg.enc2dec.decoders import GEN
from zsdg.utils import Pack
from zsdg.models.model_bases import BaseModel


class PtrBase(BaseModel):
    def compute_loss(self, dec_outs, dec_ctx, labels):
        rnn_loss = self.nll_loss(dec_outs, labels)
        # find attention loss
        g = dec_ctx.get(DecoderPointerGen.KEY_G)
        if g is not None:
            ptr_softmax = dec_ctx[DecoderPointerGen.KEY_PTR_SOFTMAX]
            flat_ptr = ptr_softmax.view(-1, self.vocab_size)
            label_mask = labels.view(-1, 1) == self.rev_vocab[PAD]
            label_ptr = flat_ptr.gather(1, labels.view(-1, 1))
            not_in_ctx = label_ptr == 0
            mix_ptr = torch.cat([label_ptr, g.view(-1, 1)], dim=1).gather(1, not_in_ctx.long())
            # mix_ptr = g.view(-1, 1) + label_ptr
            attention_loss = -1.0 * torch.log(mix_ptr.clamp(min=1e-10))
            attention_loss.masked_fill_(label_mask, 0)

            valid_cnt = (label_mask.size(0) - torch.sum(label_mask).float()).clamp(min=1e-10)
            avg_attn_loss = torch.sum(attention_loss) / valid_cnt
        else:
            avg_attn_loss = None

        return Pack(nll=rnn_loss, attn_loss=avg_attn_loss)


class HRED(BaseModel):
    def valid_loss(self, loss, batch_cnt=None):
        return loss.nll

    def __init__(self, corpus, config):
        super(HRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        # build model here
        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout,
                                         use_attn=config.utt_type == 'attn_rnn',
                                         vocab_size=self.vocab_size,
                                         embed_dim=config.embed_size, feat_size=1)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)

        if config.bi_ctx_cell or config.num_layer > 1:
            self.connector = Bi2UniConnector(config.rnn_cell, config.num_layer,
                                             config.ctx_cell_size,
                                             config.dec_cell_size)
        else:
            self.connector = IdentityConnector()


        self.decoder = DecoderRNN(self.vocab_size, config.max_dec_len,
                                  config.embed_size, config.dec_cell_size,
                                  self.go_id, self.eos_id,
                                  n_layers=1, rnn_cell=config.rnn_cell,
                                  input_dropout_p=config.dropout,
                                  dropout_p=config.dropout,
                                  use_attention=config.use_attn,
                                  attn_size=self.ctx_encoder.output_size,
                                  attn_mode=config.attn_type,
                                  use_gpu=config.use_gpu)
        self.nll = criterions.NLLEntropy(self.pad_id, config)


    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """         
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X
        
        :param data_feed: 
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(ctx_lens)

        enc_inputs = self.utt_encoder(ctx_utts, ctx_confs)

        enc_outs, enc_last = self.ctx_encoder(enc_inputs, ctx_lens)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # pack attention context
        if self.config.use_attn:
            attn_inputs = enc_outs
        else:
            attn_inputs = None

        # create decoder initial states
        dec_init_state = self.connector(enc_last)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   attn_context=attn_inputs,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.config.beam_size)
        if mode == GEN:
            return dec_ctx, labels
        else:
            if return_latent:
                return Pack(nll=self.nll(dec_outs, labels), latent_actions=dec_init_state)
            else:
                return Pack(nll=self.nll(dec_outs, labels))


class PtrHRED(PtrBase):

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.nll + 0.01 * loss.attn_loss
        return total_loss

    def __init__(self, corpus, config):
        super(PtrHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size)

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout,
                                         use_attn=True,
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding, feat_size=1)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)

        if config.bi_ctx_cell or config.num_layer > 1:
            self.connector = Bi2UniConnector(config.rnn_cell, config.num_layer,
                                             config.ctx_cell_size,
                                             config.dec_cell_size)
        else:
            self.connector = IdentityConnector()

        self.attn_size = self.ctx_encoder.output_size

        self.decoder = DecoderPointerGen(self.vocab_size, config.max_dec_len,
                                         config.embed_size, config.dec_cell_size,
                                         self.go_id, self.eos_id,
                                         n_layers=1, rnn_cell=config.rnn_cell,
                                         input_dropout_p=config.dropout,
                                         dropout_p=config.dropout,
                                         attn_size=self.attn_size,
                                         attn_mode=config.attn_type,
                                         use_gpu=config.use_gpu,
                                         embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(ctx_lens)

        utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)

        ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # create decoder initial states
        dec_init_state = self.connector(ctx_last)

        # attention
        ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
        utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
        attn_inputs = ctx_outs + utt_outs
        flat_ctx_words = ctx_utts.view(batch_size, -1)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, attn_inputs, flat_ctx_words,
                                                   inputs=dec_inputs, init_state=dec_init_state,
                                                   mode=mode, gen_type=gen_type)
        if mode == GEN:
            return dec_ctx, labels
        else:
            results = self.compute_loss(dec_outs, dec_ctx, labels)
            if return_latent:
                results['latent_actions'] = dec_init_state
            return results


class ZeroShotHRED(PtrBase):
    def __init__(self, corpus, config):
        super(ZeroShotHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout,
                                         use_attn=config.utt_type == 'rnn_attn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding, feat_size=1)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)


        self.policy = nn_lib.Hidden2Feat(self.ctx_encoder.output_size, config.dec_cell_size,
                                         is_lstm=config.rnn_cell=='lstm')
        self.utt_policy = lambda x: x

        self.connector = nn_lib.LinearConnector(config.dec_cell_size, config.dec_cell_size,
                                                is_lstm=config.rnn_cell == 'lstm')

        self.attn_size = self.ctx_encoder.output_size

        self.decoder = DecoderRNN(self.vocab_size, config.max_dec_len,
                                  config.embed_size, config.dec_cell_size,
                                  self.go_id, self.eos_id,
                                  n_layers=1, rnn_cell=config.rnn_cell,
                                  input_dropout_p=config.dropout,
                                  dropout_p=config.dropout,
                                  use_attention=config.use_attn,
                                  attn_size=self.ctx_encoder.output_size,
                                  attn_mode=config.attn_type,
                                  use_gpu=config.use_gpu)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)
        self.l2_loss = criterions.L2Loss()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.distance + loss.nll
        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', return_latent=False):
        """
        B: batch_size, D: context_size U: utt_size, X: response_size
        1. ctx_lens: B x 1
        2. ctx_utts: B x D x U
        3. ctx_confs: B x D
        4. ctx_floors: B x D
        5. out_lens: B x 1
        6. out_utts: B x X

        :param data_feed:
        {'ctx_lens': vec_ctx_lens, 'ctx_utts': vec_ctx_utts,
         'ctx_confs': vec_ctx_confs, 'ctx_floors': vec_ctx_floors,
         'out_lens': vec_out_lens, 'out_utts': vec_out_utts}
        :param return_label
        :param dec_type
        :return: outputs
        """
        # optional fields
        ctx_lens = data_feed.get('context_lens')
        ctx_utts = self.np2var(data_feed.get('contexts'), LONG)
        ctx_confs = self.np2var(data_feed.get('context_confs'), FLOAT)
        out_acts = self.np2var(data_feed.get('output_actions'), LONG)
        domain_metas = self.np2var(data_feed.get('domain_metas'), LONG)

        # required fields
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(data_feed['outputs'])
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)

        # forward pass
        out_embedded, out_outs, _, _ = self.utt_encoder(out_utts.unsqueeze(1), out_confs, return_all=True)
        out_embedded = self.utt_policy(out_embedded.squeeze(1))

        if ctx_lens is None:
            act_embedded, act_outs, _, _ = self.utt_encoder(out_acts.unsqueeze(1), out_confs, return_all=True)
            act_embedded = act_embedded.squeeze(1)

            # create attention contexts
            attn_inputs = act_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_words = out_acts.view(batch_size, -1)
            latent_action = self.utt_policy(act_embedded)
        else:
            utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)
            ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)

            # create decoder initial states
            latent_action = self.policy(ctx_last)

            # create attention contexts
            ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
            utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_inputs = ctx_outs + utt_outs  # batch_size x num_word x attn_size
            attn_words = ctx_utts.view(batch_size, -1)  # batch_size x num_words

        dec_init_state = self.connector(latent_action)

        # mask out PAD words in the attention inputs
        attn_inputs, attn_words = self._remove_padding(attn_inputs, attn_words)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   attn_context=attn_inputs,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.config.beam_size)
        if mode == GEN:
            return dec_ctx, labels
        else:
            rnn_loss = self.nll_loss(dec_outs, labels)
            loss_pack = Pack(nll=rnn_loss)
            if return_latent:
                loss_pack['latent_actions'] = latent_action

            loss_pack['distance'] = self.l2_loss(out_embedded,latent_action)
            return loss_pack


class ZeroShotPtrHRED(PtrBase):
    def __init__(self, corpus, config):
        super(ZeroShotPtrHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.pad_id = self.rev_vocab[PAD]

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.embedding1 = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, config.dropout,
                                         use_attn=config.utt_type == 'rnn_attn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding, feat_size=1)
        self.Semantic_guide = Semantic_guide(config.utt_cell_size, config.dropout,
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding,)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      config.dropout,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=False,
                                      bidirection=config.bi_ctx_cell)

        self.policy = nn.Linear(self.ctx_encoder.output_size, config.dec_cell_size)
        self.utt_policy = lambda x: x

        self.connector = nn_lib.LinearConnector(config.dec_cell_size, config.dec_cell_size,
                                                is_lstm=config.rnn_cell == 'lstm')

        self.connector_f= nn_lib.LinearConnector_f(config.dec_cell_size, config.dec_cell_size,
                                                is_lstm=False)

        self.connector_vg = nn_lib.LinearConnector_vg(config.dec_cell_size, config.dec_cell_size,
                                                          is_lstm=False)
        self.calculate_T = calculate_T(config.utt_cell_size, config.dropout,
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding)

        self.attn_size = self.ctx_encoder.output_size
        self.decoder = DecoderPointerGen(self.vocab_size, config.max_dec_len,
                                         config.embed_size, config.dec_cell_size,
                                         self.go_id, self.eos_id,
                                         n_layers=1, rnn_cell=config.rnn_cell,
                                         input_dropout_p=config.dropout,
                                         dropout_p=config.dropout,
                                         attn_size=self.attn_size,
                                         attn_mode=config.attn_type,
                                         use_gpu=config.use_gpu,
                                         embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.pad_id, config)
        self.l2_loss = criterions.L2Loss()
        self.l2_loss1 = criterions.L2Loss()
        self.softmaxfn = nn_lib.fn_softmax(config.dec_cell_size, config.dec_cell_size,
                                                is_lstm=config.rnn_cell == 'lstm')
        self.softmax=nn.Softmax()
        self.softmax1=nn.Softmax(dim=1)


    def valid_loss(self, loss, batch_cnt=None):
        # total_loss = loss.loss_pack_warm
        # total_loss = loss.distance +loss.distance1+ loss.nll + 0.01 * loss.attn_loss + 0.01*loss.distance_semantic+
        total_loss = 0.1*loss.distance1 + loss.nll + 0.01 * loss.attn_loss + 0.1*loss.distance_semantic
        # +loss.distance1
        # +loss.loss_pack_warm
        # total_loss = loss.nll + 0.01 * loss.attn_loss + 0.1*loss.distance_semantic
        # total_loss = loss.distance + loss.nll + 0.01 * loss.attn_loss
        # total_loss = loss.nll + 0.01 * loss.attn_loss
        # total_loss = loss.distance_semantic
        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy',istrain=False, return_latent=False):
        # optional fields
        paper3=True
        similar=True
        # istrain = True
        ifwarm = data_feed.get('ifwarm')
        ctx_lens = data_feed.get('context_lens')
        ctx_utts = self.np2var(data_feed.get('contexts'), LONG)
        ctx_confs = self.np2var(data_feed.get('context_confs'), FLOAT)
        out_acts = self.np2var(data_feed.get('output_actions'), LONG)
        out_acts_semtic = self.np2var(data_feed.get('output_acts_semtic'), LONG)
        # ctx_acts_semtic = self.np2var(data_feed.get('vec_ctx_acts_semtic'), LONG)
        # print(out_acts_semtic.shape)
        # out_acts_warm_s = self.np2var(data_feed.get('vec_acts_warm_S'), LONG)
        vec_out_utts_warm_s =self.np2var(data_feed.get('vec_acts_warm_S')[0], LONG)
        vec_out_utts_warm_len_s =self.np2var(data_feed.get('vec_acts_warm_S')[3], LONG)
        vec_out_acts_warm_s =self.np2var(data_feed.get('vec_acts_warm_S')[1], LONG)
        vec_out_acts_semtic_s =self.np2var(data_feed.get('vec_acts_warm_S')[2], LONG)

        # out_acts_warm_t = self.np2var(data_feed.get('vec_acts_warm_T'), LONG)
        vec_out_utts_warm_t = self.np2var(data_feed.get('vec_acts_warm_T')[0], LONG)
        vec_out_utts_warm_len_t = self.np2var(data_feed.get('vec_acts_warm_T')[3], LONG)
        vec_out_acts_warm_t = self.np2var(data_feed.get('vec_acts_warm_T')[1], LONG)
        vec_out_acts_semtic_t = self.np2var(data_feed.get('vec_acts_warm_T')[2], LONG)
        # R(x,d)    required fields
        out_utts = self.np2var(data_feed['outputs'], LONG)
        batch_size = len(data_feed['outputs'])
        out_confs = self.np2var(np.ones((batch_size, 1)), FLOAT)

        out_embedded, out_outs, _, _ = self.utt_encoder(out_utts.unsqueeze(1), out_confs, return_all=True)

        out_embedded = self.utt_policy(out_embedded.squeeze(1))
        # ctx_lens = None

        Semantic_label = self.Semantic_guide(out_acts_semtic.unsqueeze(1))
        if ctx_lens is None:
            act_embedded, act_outs, _, _ = self.utt_encoder(out_acts.unsqueeze(1), out_confs, return_all=True)
            act_embedded = act_embedded.squeeze(1)

            # create attention contexts
            attn_inputs = act_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
            attn_words = out_acts.view(batch_size, -1)
            latent_action = self.utt_policy(act_embedded)
        else:
            utt_embedded, utt_outs, _, _ = self.utt_encoder(ctx_utts, ctx_confs, return_all=True)
            ctx_outs, ctx_last = self.ctx_encoder(utt_embedded, ctx_lens)
            pi_inputs = self._gather_last_out(ctx_outs, ctx_lens)

            # create decoder initial states
            latent_action = self.policy(pi_inputs)
            latent_action_s = latent_action

            # create attention contexts
            ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
            utt_outs = utt_outs.contiguous().view(batch_size, -1, self.utt_encoder.output_size)

            #######################################################################################################################

            # dec_init_state = self.connector(latent_action)
            Semantic_label = self.Semantic_guide(out_acts_semtic.unsqueeze(1))
            Vghidden = Semantic_label
            # if istrain:
            if True:
                # 加入语义嵌入
                fout = self.connector_f(utt_outs)
                Vgout, Vghidden = self.connector_vg(utt_outs)

                # vec_out_acts_semtic_s_all=torch.cat((vec_out_acts_semtic_s, vec_out_acts_semtic_t), 0)
                vec_out_acts_semtic_s_all=vec_out_acts_semtic_t
                semtic_s,semtic_word_embd = self.calculate_T(vec_out_acts_semtic_s_all.unsqueeze(1))
                # Semantic_label.unsqueeze(1)
                Semantic_label_new=Semantic_label.unsqueeze(1).repeat(1, semtic_s.shape[0], 1)
                # if similar:

                semtic_s_act, semtic_word_embd_act = self.calculate_T(vec_out_acts_warm_t.unsqueeze(1))
                semtic_s_act_S, semtic_word_embd_act_S = self.calculate_T(vec_out_acts_warm_s.unsqueeze(1))
                semtic_s = semtic_s.unsqueeze(0).repeat(batch_size,1, 1)
                # outputsimi =Semantic_label.mm(semtic_s.view(semtic_s.shape[1], semtic_s.shape[0]))
                outputsimi = torch.sum(Semantic_label_new.mul(semtic_s),dim=2)
                # outputsimi.unsqueeze(1).repeat(1, semtic_word_embd.shape[1], 1)
                outputsimi_soft=self.softmax1(outputsimi).unsqueeze(-1).unsqueeze(-1)\
                    .repeat(1, 1, vec_out_acts_warm_s.shape[1], Semantic_label_new.shape[2])
                semtic_word_embd_expend = semtic_word_embd_act.unsqueeze(0).repeat(Semantic_label_new.shape[0], 1,1, 1)
                semtic_word_embd_expend_S = semtic_word_embd_act_S.unsqueeze(0).repeat(Semantic_label_new.shape[0], 1,1, 1)

                # vec_out_acts_semtic_a= torch.cat((vec_out_acts_warm_s, vec_out_acts_warm_t), 0)
                vec_out_acts_semtic_a= vec_out_acts_warm_t
                self.calculate_T(vec_out_acts_semtic_a.unsqueeze(1))
                vec_out_utts_warm_test = torch.sum(outputsimi_soft.mul(semtic_word_embd_expend),dim=1)
                vec_out_utts_warm_test_S = torch.sum(outputsimi_soft.mul(semtic_word_embd_expend_S),dim=1)
                self.softmax(outputsimi)
                # outputsimi =Semantic_label_new.mm(semtic_word_embd.view(semtic_s.shape[1], semtic_s.shape[0]))
                max_index = [np.argmax(outputsimi.tolist()[i]) for i in range(len(outputsimi.tolist()))]

                idxxs = np.random.choice(len(vec_out_acts_warm_s), 10)
                idxxt = np.random.choice(len(vec_out_acts_warm_t), 10)
                vec_out_acts_warm_s_=vec_out_acts_warm_s[idxxs]
                vec_out_acts_warm_t_=vec_out_acts_warm_t[idxxt]
                vec_out_utts_warm_s_ = vec_out_utts_warm_s[idxxs]
                vec_out_utts_warm_t_ = vec_out_utts_warm_t[idxxt]
                vec_out_acts_warm = torch.cat((vec_out_acts_warm_s_,vec_out_acts_warm_t_), 0)
                vec_out_utts_warm = torch.cat((vec_out_utts_warm_s_,vec_out_utts_warm_t_), 0)
                vec_out_acts_semtic_s_all_warm = torch.cat((vec_out_acts_warm_s,vec_out_acts_warm_t), 0)[max_index]
                # vec_out_utts_warm = vec_out_utts_warm_s   vec_out_acts_warm_s
                # vec_out_acts_warm = vec_out_acts_warm_s
                # out_embedded_warm, out_outs_warm, _, _ = self.utt_encoder(vec_out_utts_warm.unsqueeze(1), out_confs,
                #                                                 return_all=True)
                out_embedded_warm, out_outs_warm, _, _ = self.utt_encoder(vec_out_utts_warm_test, out_confs,semic=True,
                                                                return_all=True)
                out_embedded_warm = self.utt_policy(out_embedded_warm.squeeze(1))

                # act_embedded_warm, act_outs_warm, _, _ = self.utt_encoder(vec_out_acts_warm.unsqueeze(1), out_confs,
                #                                                 return_all=True)

                act_embedded_warm, act_outs_warm, _, _ = self.utt_encoder(vec_out_utts_warm_test, out_confs,semic=True,
                                                                return_all=True)
                act_embedded_warm_S, act_outs_warm_S, _, _ = self.utt_encoder(vec_out_utts_warm_test_S, out_confs,semic=True,
                                                                return_all=True)


                act_embedded_warm = act_embedded_warm.squeeze(1)
                act_embedded_warm_S = act_embedded_warm_S.squeeze(1)

                # create attention contexts
                attn_inputs_warm = act_outs_warm.contiguous().view(batch_size, -1, self.utt_encoder.output_size)
                attn_words_warm = vec_out_acts_warm.view(batch_size, -1)
                latent_action_warm = self.utt_policy(act_embedded_warm)
                dec_init_state_warm = self.connector(latent_action_warm)

                attn_inputs_warm, attn_words_warm = self._remove_padding(attn_inputs_warm, attn_words_warm)

                # get decoder inputs
                labels_warm =  vec_out_utts_warm[:, 1:].contiguous()
                dec_inputs_warm = vec_out_utts_warm[:, 0:-1]

                # decode
                dec_outs_warm, dec_last_warm, dec_ctx_warm = self.decoder(batch_size, attn_inputs_warm, attn_words_warm,
                                                           inputs=dec_inputs_warm, init_state=dec_init_state_warm,
                                                           mode=mode, gen_type=gen_type)

                ha = fout.mul(Vgout)
                ha_ = np.sum(ha.tolist(), 2)
                ha_ = torch.tanh(torch.from_numpy(ha_))
                # pi_test = self.softmax(ha_)
                pi = self.softmax(ha_).unsqueeze(2).repeat(1, 1, 512)
                np.random.choice(len(vec_out_acts_warm_s), 10)                #############################################
                # ctx_outs = ctx_outs.unsqueeze(2).repeat(1, 1, ctx_utts.size(2), 1).view(batch_size, -1, self.ctx_encoder.output_size)
                #############################################
                # pi.mul(pi)
                utt_out_with_att = utt_outs.mul(pi.float().cuda())
                utt_outs_fn = utt_out_with_att+utt_outs
            if istrain:
                attn_inputs = ctx_outs + utt_outs_fn  # batch_size x num_word x attn_size
                attn_words = ctx_utts.view(batch_size, -1)  # batch_size x num_words
                latent_action = latent_action + 0.1 * latent_action.mul(act_embedded_warm)
            else:
                latent_actionS = latent_action + 0.1 * latent_action.mul(act_embedded_warm_S)
                latent_action_s = latent_actionS
                latent_action = latent_action + 0.1 * latent_action.mul(latent_action_s)
                dec_init_state = self.connector(latent_action)
                attn_inputs = ctx_outs + utt_outs  # batch_size x num_word x attn_size
                attn_words = ctx_utts.view(batch_size, -1)  # batch_size x num_words
            #######################################################################################################################
        dec_init_state = self.connector(latent_action)
        # mask out PAD words in the attention inputs
        attn_inputs, attn_words = self._remove_padding(attn_inputs, attn_words)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, attn_inputs, attn_words,
                                                   inputs=dec_inputs, init_state=dec_init_state,
                                                   mode=mode, gen_type=gen_type)

        if mode == GEN:
            return dec_ctx, labels
        else:
            if ifwarm:
                loss_pack = self.compute_loss(dec_outs, dec_ctx, labels)
                loss_pack['distance'] = self.l2_loss(out_embedded, latent_action)
                loss_pack['distance_semantic'] = torch.from_numpy(np.asarray(0.0)).cuda().float()
                loss_pack['loss_pack_warm'] = torch.from_numpy(np.asarray(0.0)).cuda().float()
                loss_pack['distance1'] = torch.from_numpy(np.asarray(0.0)).cuda().float()
                return loss_pack
            loss_pack = self.compute_loss(dec_outs, dec_ctx, labels)
            loss_pack['distance'] = self.l2_loss(out_embedded, latent_action)
            if istrain:
                loss_pack['distance_semantic'] = self.l2_loss1(Semantic_label, Vghidden)
                loss_pack_warm= self.nll_loss(dec_outs_warm, labels_warm)
                loss_pack['loss_pack_warm']=loss_pack_warm
                loss_pack['distance1'] = self.l2_loss(act_embedded_warm , latent_action)
            else:
                loss_pack['distance_semantic'] = self.l2_loss(out_embedded, latent_action)
                loss_pack['loss_pack_warm'] =self.l2_loss(out_embedded, latent_action)
                loss_pack['distance1'] = self.l2_loss(out_embedded, latent_action)

            # #########################
            # #
            # #
            # #
            # #########################
            # # loss_pack['distance2'] = self.l2_loss(out_embedded_warm , latent_action_warm)
            # # loss_pack = Pack()
            # loss_pack['distance_semantic'] = self.l2_loss1(Semantic_label, Vghidden)
            return loss_pack
