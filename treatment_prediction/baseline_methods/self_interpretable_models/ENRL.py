# coding=utf-8
import torch
import logging
from argparse import ArgumentParser
import numpy as np
import treelib

# from ..configs.constants import *
# from ..configs.settings import *
# from ..models.Model import Model
# from ..modules.nas import *
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .draw_rules import draw_rules

CTXT_MH = 'ctxt_mh'
CTXT_NU = 'ctxt_nm'

OP_LOSS = 'op_loss'
GL_INPUT = 'gl_input'
GT_LABEL = 'gt_label'
LT_LABEL = 'lt_label'
def hard_max(values):
    _, index = values.max(dim=-1)  # ?
    hard = torch.nn.functional.one_hot(index, num_classes=values.size(-1))  # ? * c
    hard = (hard - values).detach() + values  # ? * c
    return hard  # ? * c

def hard_softmax_func(logits, tau=1.0):
    probs = torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c
    return hard_max(probs)  # ? * c

def gumbel_softmax_func(logits, training, random_r=0.5, eps=1e-20, tau=1.0, hard_r=1.0, hard=True):
    if not training:
        return hard_softmax_func(logits, tau=tau) if hard \
            else torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c

    u = torch.rand(logits.size(), device=logits.device)  # ? * c
    u = -torch.log(-torch.log(u + eps) + eps)  # ? * c
    r = torch.rand(logits.size()[:-1], device=logits.device).le(random_r).unsqueeze(dim=-1)  # ? * 1
    logits = logits + u * r  # ? * c

    if hard_r >= 1.0:
        return hard_softmax_func(logits, tau=tau)  # ? * c
    if hard_r <= 0.0:
        return torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c

    hard_result = hard_softmax_func(logits, tau=tau)  # ? * c
    soft_result = torch.nn.functional.softmax(logits / tau, dim=-1)  # ? * c
    r = torch.rand(logits.size()[:-1], device=logits.device).le(hard_r).unsqueeze(dim=-1).float()  # ? * 1
    return hard_result * r + soft_result * (-r + 1)

class GumbelSoftmax(torch.nn.Module):
    def __init__(self, random_r=0.5, eps=1e-20, tau=1.0, hard_r=1.0, hard=True):
        super().__init__()
        self.random_r = random_r
        self.eps = eps
        self.tau = tau
        self.hard = hard
        self.hard_r = hard_r

    def forward(self, logits):
        return gumbel_softmax_func(logits=logits, training=self.training, random_r=self.random_r,
                                   eps=self.eps, tau=self.tau, hard_r=self.hard_r, hard=self.hard)

class ENRL(torch.nn.Module):
    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     """
    #     模型命令行参数
    #     :return:
    #     """
    #     parser = Model.add_model_specific_args(parent_parser)
    #     parser.add_argument('--rule_n', type=int, default=40,
    #                         help='Number of rules.')
    #     parser.add_argument('--rule_len', type=int, default=3,
    #                         help='Number of operators in each rule.')
    #     parser.add_argument('--gumbel_r', type=float, default=0.5,
    #                         help='Random explore ratio random_r in Gumbel softmax.')
    #     parser.add_argument('--gumbel_hr', type=float, default=0.9,
    #                         help='Hard ratio har_r in Gumbel softmax.')
    #     parser.add_argument('--op_loss', type=float, default=0.001,
    #                         help='Loss weights of operation regularizers.')
    #     parser.add_argument('--op_scale', type=float, default=10,
    #                         help='Scale before sigmoid in each operator.')
    #     parser.add_argument('--vec_size', type=int, default=64,
    #                         help='Size of feature vectors.')
    #     return parser

    def __init__(self, rule_n: int = 1, rule_len: int = 3, op_scale: int = 10,
                 multihot_f_num: int = None, multihot_f_dim: int = None,
                 numeric_f_num: int = None, numeric_f_dim: int = None,
                 gumbel_r: float = 0.1, gumbel_hr: float = 1.0, op_loss: float = 0.001, loss_sum: int = 1,loss_type: str = 'mse',
                 vec_size: int = 16, feat_range_mappings=None, mh_dict: dict = None, num_treatment=2, cont_treatment=False, has_dose=False, outcome_regression=True, num_class=None):
        # super().__init__(*args, **kwargs)
        super(ENRL, self).__init__()
        self.rule_n = rule_n
        self.rule_len = rule_len
        self.op_scale = op_scale
        # self.nu_dict = nu_dict
        # self.mh_dict = mh_dict
        self.feat_range_mappings = feat_range_mappings
        self.multihot_f_num = multihot_f_num
        self.multihot_f_dim = multihot_f_dim
        self.numeric_f_num = numeric_f_num
        self.numeric_f_dim = numeric_f_dim
        self.vec_size = vec_size
        self.gumbel_r = gumbel_r
        self.gumbel_hr = gumbel_hr
        self.op_loss = op_loss
        self.num_treatment = num_treatment
        self.cont_treatment = cont_treatment
        self.has_dose = has_dose
        self.outcome_regression= outcome_regression
        self.loss_sum = loss_sum
        self.loss_type = loss_type
        self.num_class = num_class
        self.init_modules()

    # def read_data(self, *args, **kwargs):
    #     reader = super().read_data(*args, **kwargs)
    #     data_dicts = [d for d in [reader.train_data, reader.val_data, reader.test_data] if d is not None]

    #     self.nu_dict, self.numeric_f_dim = reader.multihot_features(
    #         data_dicts, combine=CTXT_NU, k_filter=lambda x: x.endswith(INT_F))
    #     self.numeric_f_num = len(self.nu_dict)
    #     self.model_logger.info('numeric_f_num = {}'.format(self.numeric_f_num))
    #     self.model_logger.info('numeric_f_dim = {}'.format(self.numeric_f_dim))

    #     self.mh_dict, self.multihot_f_dim = reader.multihot_features(
    #         data_dicts, combine=CTXT_MH, k_filter=lambda x: x.endswith(CAT_F))
    #     self.multihot_f_num = len(self.mh_dict)
    #     self.model_logger.info('multihot_f_num = {}'.format(self.multihot_f_num))
    #     self.model_logger.info('multihot_f_dim = {}'.format(self.multihot_f_dim))
    #     return reader

    # def read_formatters(self, formatters=None) -> dict:
    #     current = {
    #         '^' + LABEL + '$': None,
    #         '^.*({}|{})$'.format(INT_F, CAT_F): None,
    #     }
    #     if formatters is None:
    #         return current
    #     return {**current, **formatters}

    # def on_train_epoch_start(self) -> None:
    #     super().on_train_epoch_start()
    #     nu_features = [f for f in self.nu_dict if f.endswith(INT_F)]
    #     if len(nu_features) > 0:
    #         gl_input = []
    #         for f in nu_features:
    #             lo, hi = self.nu_dict[f]
    #             f_ab = np.random.choice(range(lo, hi + 1), size=(len(self.train_dataset), 2), replace=True)
    #             gl_input.append(f_ab)
    #         gl_input = np.stack(gl_input, axis=1)
    #         self.train_dataset.data[GL_INPUT] = gl_input
    #     return

    # def dataset_get_item(self, dataset, index: int) -> dict:
    #     if dataset.buffer_ds > 0: return dataset.index_buffer[index]
    #     index_dict = {}
    #     index_dict[LABEL] = dataset.data[LABEL][index]
    #     if CTXT_NU in dataset.data:
    #         index_dict[CTXT_NU] = dataset.data[CTXT_NU][index]
    #     if CTXT_MH in dataset.data:
    #         index_dict[CTXT_MH] = dataset.data[CTXT_MH][index]
    #     if GL_INPUT in dataset.data:
    #         index_dict[GL_INPUT] = dataset.data[GL_INPUT][index]
    #     return index_dict

    # def dataset_collate_batch(self, dataset, batch: list) -> dict:
    #     result_dict = {}
    #     for c in [GL_INPUT, GT_LABEL, LT_LABEL]:
    #         if c in batch[0]:
    #             result_dict[c] = dataset.collate_concat([b.pop(c) for b in batch])
    #     result_dict = {**result_dict, **super().dataset_collate_batch(dataset, batch)}
    #     return result_dict

    def init_modules(self) -> None:
        # self.numeric_embeddings = torch.nn.Embedding(self.numeric_f_dim, self.vec_size)
        self.numeric_embeddings = torch.nn.Sequential(torch.nn.Linear(1, self.vec_size), torch.nn.Sigmoid())
        self.multihot_embeddings = torch.nn.Embedding(self.multihot_f_dim, self.vec_size)
        if not self.cont_treatment:
            self.treatment_embeddings = torch.nn.Embedding(self.num_treatment, self.vec_size)
        else:
            self.treatment_embeddings = torch.nn.Sequential(torch.nn.Linear(1, self.vec_size), torch.nn.Sigmoid())
        
        if self.has_dose:
            self.dose_embeddings = torch.nn.Sequential(torch.nn.Linear(1, self.vec_size), torch.nn.Sigmoid())
        
        self.ge_layers = torch.nn.Sequential(
            torch.nn.Linear(2 * self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, 1),
            torch.nn.Sigmoid()
        )
        self.le_layers = torch.nn.Sequential(
            torch.nn.Linear(2 * self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, 1),
            torch.nn.Sigmoid()
        )
        self.blto_layers = torch.nn.Sequential(
            torch.nn.Linear(2 * self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, 1),
            torch.nn.Sigmoid()
        )

        self.ge_state_layers = torch.nn.Sequential(
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, 2 * self.vec_size),
        )
        self.le_state_layers = torch.nn.Sequential(
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, 2 * self.vec_size),
        )
        self.blto_state_layers = torch.nn.Sequential(
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, 2 * self.vec_size),
        )

        self.tree_layers = [1]
        for l in range(1, self.rule_len):
            self.tree_layers.append(self.tree_layers[-1] * 2)
        self.node_n = sum(self.tree_layers)
        self.operator_n = 2

        self.ge_v = torch.nn.Parameter(
            torch.normal(0.0, 0.01, size=(self.rule_n, self.node_n, self.numeric_f_num + 1 + int(self.has_dose), self.vec_size)),
            requires_grad=True)
        self.le_v = torch.nn.Parameter(
            torch.normal(0.0, 0.01, size=(self.rule_n, self.node_n, self.numeric_f_num + 1 + int(self.has_dose), self.vec_size)),
            requires_grad=True)
        self.blto_v = torch.nn.Parameter(
            torch.normal(0.0, 0.01, size=(self.rule_n, self.node_n, self.multihot_f_num + 1 + int(self.has_dose), self.vec_size)),
            requires_grad=True)
        self.nas_w = torch.nn.Parameter(
            torch.normal(0.0, 0.01, size=(self.rule_n, self.node_n,
                                          self.operator_n * self.numeric_f_num+ 1 + int(self.has_dose) + self.multihot_f_num+ 1 + int(self.has_dose))),
            requires_grad=True)
        self.gumbel_softmax = GumbelSoftmax(random_r=self.gumbel_r, hard_r=self.gumbel_hr)
        self.rule_weight = torch.nn.Linear(self.tree_layers[-1] * self.rule_n * 2, 1, bias=False)
        
        output_dim = 1
        if self.num_class is not None:
            output_dim = self.num_class
        self.rule_weight_layers = torch.nn.Sequential(
            torch.nn.Linear(self.rule_len * self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, self.vec_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.vec_size, output_dim),
        )

        # self.dropout_layer = torch.nn.Dropout(self.dropout)
        self.register_buffer('rule_pos_cnt', torch.zeros((self.rule_n,)))  # r
        self.init_weights()
        return

    def init_weights(self) -> None:
        for n, p in self.named_parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, mean=0, std=0.01)

    def forward(self, X_num, X_cat, treatment, d=None):
        if self.training:
            return self.forward_single(X_num, X_cat, treatment, d=d)
        
        
        if not self.cont_treatment:
            full_pred = []
            for t in range(self.num_treatment):
                pred, _ = self.forward_single(X_num, X_cat, torch.ones_like(treatment)*t, d=d)
                full_pred.append(pred)
            full_pred = torch.stack(full_pred, dim=1)
            pred = full_pred[torch.arange(len(full_pred)), treatment.view(-1).long()]
            return pred, full_pred
        else:
            pred, _ = self.forward_single(X_num, X_cat, treatment, d=d)
            return pred, None

    def forward_single(self, X_num, X_cat, treatment, d=None):
        gl_input = None
        if self.training:
            gl_input = []
            for f in range(X_num.shape[1]):
                # lo, hi = self.nu_dict[f]
                lo, hi = self.feat_range_mappings[f]
                rand_vals = np.random.rand(len(X_num), 2)
                f_ab = lo + (hi - lo) * rand_vals
                # f_ab = np.random.choice(range(lo, hi + 1), size=(len(X_num), 2), replace=True)
                gl_input.append(f_ab)
            gl_input = np.stack(gl_input, axis=1)
            gl_input = torch.from_numpy(gl_input).float().to(X_num.device)
        ops, states = [], []
        # if CTXT_NU in batch:
        if self.cont_treatment:
            treatment = treatment.unsqueeze(-1)
        else:
            treatment = treatment.long()
        treatment_emd = self.treatment_embeddings(treatment)
        dose_emd = None
        if self.has_dose:
            dose_emd = self.numeric_embeddings(d.unsqueeze(-1))
        if X_num is not None:
            nu_features = self.numeric_embeddings(X_num.unsqueeze(-1))  # B * nu * v
            nu_features = torch.cat([nu_features, treatment_emd], dim=1)
            if self.has_dose:
                nu_features = torch.cat([nu_features, dose_emd], dim=1)
            
            nu_features = nu_features.unsqueeze(dim=1).unsqueeze(dim=1). \
                expand(-1, self.rule_n, self.node_n, -1, -1)  # B * r * n * nu * v
            ge_v = self.ge_v.unsqueeze(dim=0).expand_as(nu_features)  # B * r * n * nu * v
            greater_than = self.ge_layers(torch.cat([nu_features, ge_v], dim=-1)).squeeze(dim=-1)  # B * r * n * nu
            le_v = self.le_v.unsqueeze(dim=0).expand_as(nu_features)  # B * r * n * nu * v
            less_than = self.le_layers(torch.cat([nu_features, le_v], dim=-1)).squeeze(dim=-1)  # B * r * n * nu
            ops.append(greater_than)
            ops.append(less_than)
            states.append(self.ge_state_layers(self.ge_v))  # r * n * nu * 2v
            states.append(self.le_state_layers(self.le_v))  # r * n * nu * 2v
        # if CTXT_MH in batch:
        # if X_cat is not None:
        if X_cat.shape[1] > 0:
            mh_features = self.multihot_embeddings(X_cat)  # B * mh * v
            mh_features = torch.cat([mh_features, treatment_emd], dim=1)
            if self.has_dose:
                mh_features = torch.cat([mh_features, dose_emd], dim=1)
            mh_features = mh_features.unsqueeze(dim=1).unsqueeze(dim=1). \
                expand(-1, self.rule_n, self.node_n, -1, -1)  # B * r * n * mh * v
            blto_v = self.blto_v.unsqueeze(dim=0).expand_as(mh_features)  # B * r * n * mh * v
            belong_to = self.blto_layers(torch.cat([mh_features, blto_v], dim=-1)).squeeze(dim=-1)  # B * r * n * mh
            ops.append(belong_to)
            states.append(self.blto_state_layers(self.blto_v))  # r * n * mh * 2v

        ops = torch.cat(ops, dim=-1)  # B * r * n * (f*op)
        nas_w = self.gumbel_softmax(self.nas_w)  # r * n * (f*op)
        rule = (ops * nas_w.unsqueeze(dim=0)).sum(dim=-1)  # B * r * n

        states = torch.cat(states, dim=-2)  # r * n * (f*op) * 2v
        states = (states * nas_w.unsqueeze(dim=-1)).sum(dim=-2)  # r * n * 2v

        layers, layer_states = [], []
        lo, hi = 0, 0
        for l in self.tree_layers:
            lo, hi = hi, hi + l
            layers.append(rule[..., lo:hi])  # B * r * l
            layer_states.append(states[..., lo:hi, :])  # r * l * 2v

        rule, total = layers[0], 1  # B * r * l/2
        for layer in layers[1:]:  # B * r * l
            total = torch.stack([rule, -rule + total], dim=-1).flatten(start_dim=-2)  # B * r * l
            rule = layer * total  # B * r * l

        state = layer_states[0].view(self.rule_n, -1, 2, self.vec_size)  # r * l/2 * 2 * ?v
        for layer_state in layer_states[1:]:  # r * l * 2v
            state = state.flatten(start_dim=1, end_dim=2)  # r * l * ?v
            layer_state = layer_state.view(self.rule_n, -1, 2, self.vec_size)  # r * l * 2 * v
            state = state.unsqueeze(dim=-2).expand(-1, -1, 2, -1)  # r * l * 2 * ?v
            state = torch.cat([state, layer_state], dim=-1)  # r * l * 2 * (?+1)v

        state = state.flatten(start_dim=1, end_dim=2)  # r * 2l * (?+1)v
        rule_weight = self.rule_weight_layers(state).flatten().unsqueeze(dim=0)  # 1 * (r*2l)

        rule = torch.stack([rule, -rule + total], dim=-1).flatten(start_dim=-2)  # B * r * 2l
        rule = hard_max(rule)  # B * r * 2l
        rule_out = rule.flatten(start_dim=-2)  # B * (r*2l)

        # prediction = self.rule_weight(rule_out).sigmoid().flatten()  # B
        # prediction = (rule_weight * rule_out).sum(dim=-1).sigmoid().flatten()  # B
        prediction = (rule_weight * rule_out).sum(dim=-1).flatten()  # B
        
        return prediction, self.operation_loss(gl_input, treatment_emd, dose_emd)
        # return self.format_output(batch, {PREDICTION: prediction, OP_LOSS: self.operation_loss(batch)})
    def get_all_rules(self, bin_dict):
        all_rules = {}
        for _, r, t in draw_rules(self, bin_dict):
            if r not in all_rules:
                all_rules[r] = []
            all_rules[r].append(t)
        self.all_rules = all_rules
    def set_explaining(self):
        self.explaining = True
    def set_rules(self, rule_weight, rule_out):
        rule_out_w = rule_weight * rule_out
        sat_rules = torch.argmax(rule_out_w, dim=1).tolist()
        contra_rules = torch.argmin(rule_out_w, dim=1).tolist()

        def get_rule(r_p):
            r = r_p // (2 ** self.rule_len)
            p = r_p % (2 ** self.rule_len)
            
            return self.all_rules[r][p]

        for sat_r_p, contra_r_p in zip(sat_rules, contra_rules):
            self.sat_rules.append(get_rule(sat_r_p))
            self.contra_rules.append(get_rule(contra_r_p))
            
    def operation_loss(self, weight, treatment_emd, dose_emd):
        if not self.training or self.op_loss <= 0:
            return 0.0
        loss = 0.0
        if self.numeric_f_num > 0:
            loss_func = torch.nn.MSELoss(reduction='sum' if self.loss_sum == 1 else 'mean')
            if self.loss_type == 'bce':
                loss_func = torch.nn.BCELoss(reduction='sum' if self.loss_sum == 1 else 'mean')

            gl_a, gl_b = weight[:, :, 0], weight[:, :, 1]  # B * nu
            gl_a_vectors = self.numeric_embeddings(gl_a.unsqueeze(-1))  # B * nu * v
            gl_b_vectors = self.numeric_embeddings(gl_b.unsqueeze(-1))  # B * nu * v
            gl_a_vectors = torch.cat([gl_a_vectors, treatment_emd], dim=1)
            gl_b_vectors = torch.cat([gl_b_vectors, treatment_emd], dim=1)
            if self.has_dose:
                gl_a_vectors = torch.cat([gl_a_vectors, dose_emd], dim=1)
                gl_b_vectors = torch.cat([gl_b_vectors, dose_emd], dim=1)
            gl_vectors = torch.stack([gl_a_vectors, gl_b_vectors], dim=-2)  # B * nu * 2 * v

            ridx = torch.randint(self.rule_n, size=(gl_vectors.size(0), 2), device=gl_vectors.device)
            nidx = torch.randint(self.node_n, size=(gl_vectors.size(0), 2), device=gl_vectors.device)
            glidx = torch.randint(2, size=(gl_vectors.size(0), 2), device=gl_vectors.device)
            all_gl_v = torch.stack([self.ge_v, self.le_v], dim=2)  # r * n * 2 * nu * v
            gl_v = all_gl_v[ridx, nidx, glidx]  # B * 2 * nu * v
            gl_v_a, gl_v_b = gl_v[:, 0], gl_v[:, 1]  # B * nu * v
            gl_v = torch.stack([gl_v_a, gl_v_b], dim=-2)  # B * nu * 2 * v

            # basic
            ge_output = self.ge_layers(gl_vectors.flatten(start_dim=-2))  # B * nu * 1
            le_output = self.le_layers(gl_vectors.flatten(start_dim=-2))  # B * nu * 1
            ge_loss = loss_func(ge_output[:,0:gl_a.shape[-1]].flatten(), gl_a.ge(gl_b).float().flatten())
            le_loss = loss_func(le_output[:,0:gl_a.shape[-1]].flatten(), gl_a.le(gl_b).float().flatten())
            basic_loss = ge_loss + le_loss
            loss = basic_loss + loss

            # # reflexive
            ge_output = self.ge_layers(torch.cat([gl_v, gl_v], dim=-1))  # B * nu * 2 * 1
            le_output = self.le_layers(torch.cat([gl_v, gl_v], dim=-1))  # B * nu * 2 * 1
            ge_loss = loss_func(ge_output.flatten(), torch.ones(ge_output.flatten().size(), device=ge_output.device))
            le_loss = loss_func(le_output.flatten(), torch.ones(le_output.flatten().size(), device=le_output.device))
            reflexive_loss = ge_loss + le_loss
            loss = reflexive_loss + loss

            # # antisymmetric
            # ge_loss1 = (ge_ab * ge_ba * (1 - torch.nn.functional.cosine_similarity(gl_vectors, gl_v, dim=-1))).sum()

            av = torch.cat([gl_vectors, gl_v_a.unsqueeze(dim=-2)], dim=-2)  # B * nu * 3 * v
            bv = torch.cat([gl_v, gl_v_b.unsqueeze(dim=-2)], dim=-2)  # B * nu * 3 * v
            ge_ab = self.ge_layers(torch.cat([av, bv], dim=-1))  # B * nu * 3 * 1
            ge_ba = self.ge_layers(torch.cat([bv, av], dim=-1))  # B * nu * 3 * 1
            ge_loss = ge_ab * ge_ba * (av - bv).square()
            ge_loss = ge_loss.sum() if self.loss_sum == 1 else ge_loss.mean()

            le_ab = self.le_layers(torch.cat([av, bv], dim=-1))  # B * nu * 3 * 1
            le_ba = self.le_layers(torch.cat([bv, av], dim=-1))  # B * nu * 3 * 1
            le_loss = le_ab * le_ba * (av - bv).square()
            le_loss = le_loss.sum() if self.loss_sum == 1 else le_loss.mean()
            antisymmetric_loss = ge_loss + le_loss
            loss = antisymmetric_loss + loss

            # # transitive
            loss_func = torch.nn.MSELoss(reduction='none')
            if self.loss_type == 'bce':
                loss_func = torch.nn.BCELoss(reduction='none')

            av = torch.stack([gl_a_vectors, gl_v_a], dim=-2)  # B * nu * 2 * v
            bv = torch.stack([gl_v_a, gl_a_vectors], dim=-2)  # B * nu * 2 * v
            cv = torch.stack([gl_b_vectors, gl_v_b], dim=-2)  # B * nu * 2 * v

            ge_ab = self.ge_layers(torch.cat([av, bv], dim=-1))  # B * nu * 2 * 1
            ge_bc = self.ge_layers(torch.cat([bv, cv], dim=-1))  # B * nu * 2 * 1
            ge_ac = self.ge_layers(torch.cat([av, cv], dim=-1))  # B * nu * 2 * 1
            ge_loss = ge_ab * ge_bc * loss_func(ge_ac, torch.ones(ge_ac.size(), device=ge_ac.device))  # B * nu * 2 * 1
            ge_loss = ge_loss.sum() if self.loss_sum == 1 else ge_loss.mean()

            le_ab = self.le_layers(torch.cat([av, bv], dim=-1))  # B * nu * 2 * 1
            le_bc = self.le_layers(torch.cat([bv, cv], dim=-1))  # B * nu * 2 * 1
            le_ac = self.le_layers(torch.cat([av, cv], dim=-1))  # B * nu * 2 * 1
            le_loss = le_ab * le_bc * loss_func(le_ac, torch.ones(le_ac.size(), device=le_ac.device))  # B * nu * 2 * 1
            le_loss = le_loss.sum() if self.loss_sum == 1 else le_loss.mean()
            transitive_loss = ge_loss + le_loss
            loss = transitive_loss + loss

        if self.multihot_f_num > 0:
            loss_func = torch.nn.MSELoss(reduction='sum' if self.loss_sum == 1 else 'mean')
            if self.loss_type == 'bce':
                loss_func = torch.nn.BCELoss(reduction='sum' if self.loss_sum == 1 else 'mean')
            random_vectors = torch.empty(self.blto_v.size(), device=self.blto_v.device). \
                normal_(0, 0.01)  # r * n * mh * v
            bl = self.blto_layers(torch.cat([random_vectors, self.blto_v], dim=-1))  # r * n * mh * 1
            bl_loss = loss_func(bl.flatten(), torch.zeros(bl.flatten().size(), device=bl.device))
            loss = bl_loss + loss
        return loss

    # def format_output(self, batch, out_dict, *args, **kwargs):
    #     result = {LABEL: batch[LABEL].flatten()}
    #     return {**result, **out_dict}

    # def loss_func(self, batch, out_dict, *args, **kwargs):
    #     prediction, label = out_dict[PREDICTION].flatten(), out_dict[LABEL].flatten().float()

    #     loss_func = torch.nn.MSELoss(reduction='sum' if self.loss_sum == 1 else 'mean')
    #     if self.loss_type == 'bce':
    #         loss_func = torch.nn.BCELoss(reduction='sum' if self.loss_sum == 1 else 'mean')
    #     loss = loss_func(prediction, label)

    #     # op_loss = self.operation_loss(batch)
    #     return loss + self.op_loss * out_dict[OP_LOSS]

    # def on_train_epoch_end(self, outputs) -> None:
    #     super().on_train_epoch_end(outputs)
    #     self.model_logger.info(outputs[0][OP_LOSS])
