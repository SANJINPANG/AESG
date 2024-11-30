from typing import Dict, List, Optional, Tuple
import copy
import torch
from torch import nn, Tensor
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import random
import json
from numpy.core.numeric import Inf
import math
import torch.nn as nn

#新引入的包
from tqdm import tqdm
from kbc.src.models import ComplEx
import os

class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len=20):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=True)

    def forward(self, batch_len, start, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([list(range(start + 1, start + seq_len + 1)) for _ in range(batch_len)]).cuda()
        return self.position_encoding(input_pos).transpose(0, 1)

class PositionalEncoding1(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, args, dictionary, true_triples=None):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
        except:
            raise ImportError('Transformer module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp = args.embedding_dim
        self.args = args
        self.pos_encoder = PositionalEncoding(self.ninp)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.embedding_dim, nhead=4, dim_feedforward=args.hidden_size, dropout=args.dropout)
        self.enencoder = nn.TransformerEncoder(encoder_layers, args.num_layers)
        self.ntoken = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.encoder = nn.Embedding(self.ntoken, self.ninp)
        self.fc = torch.nn.Linear(self.ninp, self.ninp)
        self.dictionary = dictionary
        self.glue = GELU()
        self.label_smooth = args.label_smooth

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        xavier_normal_(self.encoder.weight.data)

    def logits(self, source, prev_outputs, **unused):
        bsz, src_len = source.shape
        #获取源序列批次大小和源序列长度
        out_len = prev_outputs.size(1)
        #获取目标序列长度
        device = source.device
        source = source.transpose(0, 1)
        #源序列组行列转换
        source = self.encoder(source)#验证时，source维度从（3，16）-----》（3，16，256）
        #进行编码
        source += self.pos_encoder(bsz, 0, src_len)
        #将位置编码添加进源序列
        mask = self._generate_square_subsequent_mask(prev_outputs.size(-1))
        #掩码生成
        prev_outputs = prev_outputs.transpose(0, 1)#验证时，pre_outputs维度从（16，1）————————》（1，16）  
        prev_outputs = self.encoder(prev_outputs)#验证时，pre_outputs维度从（1，16）————————》（1，16，256）
        #目标序列转换 编码
        prev_outputs += self.pos_encoder(bsz, src_len, out_len)
        #将位置编码添加进目标序列
        if self.args.encoder:
            enmask = torch.zeros(out_len + src_len, out_len + src_len)
            enmask[:, src_len:] = float("-inf")
            enmask[src_len:, src_len:] = mask 
            enmask = enmask.to(device)
            imput = torch.cat((source, prev_outputs), dim=0)
            #创建一个全是0的掩码 对角线上的元素为负无穷
            output = self.enencoder(imput, mask=enmask)[src_len:, :, :].transpose(0, 1)
            # output = self.enencoder(torch.cat((source, prev_outputs), dim=0), mask=enmask)[src_len:, :, :].transpose(0, 1)
            #首先将源序列和目标序列沿着0维度进行拼接
            #使用掩码遮挡 使用TransformerEncoder进行编码
            #选择输出目标序列编码
            #转换行列
        else:
            mask = mask.to(device)
            output = self.endecoder(source, prev_outputs, tgt_mask=mask).transpose(0, 1)
            #没有参与程序运行 论文中没有transformer解码器参与
        logits = torch.mm(self.glue(self.fc(output)).view(-1, self.ninp), self.encoder.weight.transpose(0, 1)).view(bsz, out_len, -1)
        #批次大小 目标序列长度 最后一个维度的大小由前面数据的计算获得
        return logits

    def get_loss(self, source, prev_outputs, target, mask, **unused):
        device = source.device
        bsz = prev_outputs.size(0)
        seq_len = prev_outputs.size(1)
        logits = self.logits(source, prev_outputs)
        # label-smoothing
        lprobs = F.log_softmax(logits, dim=-1)
        loss = -(self.label_smooth * torch.gather(input=lprobs, dim=-1, index=target.unsqueeze(-1)).squeeze() \
            + (1 - self.label_smooth) / (self.ntoken - 1) * lprobs.sum(dim=-1)) * mask
        loss = loss.sum() / mask.sum()
        return loss
    




#参数需要进行调整 

# 模型需要的参数 
# kbcModel = KGReasoning(args, device, adj_list, query_name_dict, name_answer_dict)
# query_name_dict, name_answer_dict

class KGReasoning(nn.Module):
    def __init__(self, args, device, adj_list):
        super(KGReasoning, self).__init__()
        self.nentity = args.nentity#实体数量
        self.nrelation = args.nrelation#关系数量
        self.device = device#运行设备
        self.relation_embeddings = list()#存储关系嵌入
        self.fraction = args.fraction#注释为将实体分段减少对GPU内存的使用
        # self.query_name_dict = query_name_dict#查询形式 不需要
        # self.name_answer_dict = name_answer_dict#答案形式 不需要
        self.neg_scale = args.neg_scale#负样本缩放因子
        #数据集的名称不同不再使用“-”进行分割,这里的代码需要重写#################################
        dataset_name = args.dataset
        # if args.data_path.split('/')[1].split('-')[1] == "237":
        #     dataset_name += "-237"
        filename = 'neural_adj/'+dataset_name+'_'+str(args.fraction)+'_'+str(args.thrshd)+'.pt'
        #这里的thrshd参数是神经矩阵阈值
        #这里新建的是一个pytorch模型的序列化文件，用来保存模型
        ####################################################################################
        #对于内存过大的数据集采用邻接矩阵的方式存储实体之间的关系


        #这里将模型的最佳轮次加载进模型对后续的尾实体集合进行重新排序
        if args.test:
            kbc_model = load_kbc(args.kbc_path, device, args.nentity, args.nrelation)

        if os.path.exists(filename):
            self.relation_embeddings = torch.load(filename, map_location=device)
            #如果这个模型文件存在，加载关系嵌入
            #这里一定是不存在的
        else:
            #加载对不同数据集训练的模型 --kbc_path kbc/FB15K/best_valid.model其中一个参数的示例
            #这里的实体和关系数量由主函数为参数赋值
            kbc_model = load_kbc(args.kbc_path, device, args.nentity, args.nrelation)
            #将训练好的模型载入
            for i in tqdm(range(args.nrelation)):
                relation_embedding = neural_adj_matrix(kbc_model, i, args.nentity, device, args.thrshd, adj_list[i])
                #为每一个关系r构建一个nentity X nentity大小的张量存储 由KGE模型计算出的 两个实体之间的关系r的概率 低于thrshd的被舍弃
                relation_embedding = (relation_embedding>=1).to(torch.float) * 0.9999 + (relation_embedding<1).to(torch.float) * relation_embedding
                #标准化概率分数 大于1的变成0.9999 小于1的不变 都是浮点数
                for (h, t) in adj_list[i]: 
                    relation_embedding[h, t] = 1.
                #将关系嵌入中的存在的头实体和尾实体对设置为1
                # add fractional
                fractional_relation_embedding = []#用于存储分段的关系嵌入
                dim = args.nentity // args.fraction#每个分段维度大小
                rest = args.nentity - args.fraction * dim#计算分段后剩余实体数量
                for i in range(args.fraction):
                    s = i * dim #s为分段的启示索引
                    t = (i+1) * dim#t为分段的结束索引
                    if i == args.fraction - 1:
                        t += rest #最后一段索引将剩余量加上
                    fractional_relation_embedding.append(relation_embedding[s:t, :].to_sparse().to(self.device))
                    #将当前分段的关系嵌入添加到列表中，并转换为稀疏张量格式，移动到指定的设备。
                self.relation_embeddings.append(fractional_relation_embedding)
                #将分段的关系嵌入添加到 self.relation_embeddings 列表中
            torch.save(self.relation_embeddings, filename)
            #
            #这里初始化模型后为每一个关系构建了一个实体X实体大小的概率矩阵 存在内存不足问题

    def relation_projection(self, embedding, r_embedding, is_neg=False):
        dim = self.nentity // self.fraction
        rest = self.nentity - self.fraction * dim
        new_embedding = torch.zeros_like(embedding).to(self.device)
        r_argmax = torch.zeros(self.nentity).to(self.device)
        for i in range(self.fraction):
            s = i * dim
            t = (i+1) * dim
            if i == self.fraction - 1:
                t += rest
            fraction_embedding = embedding[:, s:t]
            if fraction_embedding.sum().item() == 0:
                continue
            nonzero = torch.nonzero(fraction_embedding, as_tuple=True)[1]
            fraction_embedding = fraction_embedding[:, nonzero]
            fraction_r_embedding = r_embedding[i].to_dense()[nonzero, :].unsqueeze(0)
            if is_neg:
                fraction_r_embedding = torch.minimum(torch.ones_like(fraction_r_embedding).to(torch.float), self.neg_scale*fraction_r_embedding)
                fraction_r_embedding = 1. - fraction_r_embedding
            fraction_embedding_premax = fraction_r_embedding * fraction_embedding.unsqueeze(-1)
            fraction_embedding, tmp_argmax = torch.max(fraction_embedding_premax, dim=1)
            tmp_argmax = nonzero[tmp_argmax.squeeze()] + s
            new_argmax = (fraction_embedding > new_embedding).to(torch.long).squeeze()
            r_argmax = new_argmax * tmp_argmax + (1-new_argmax) * r_argmax
            new_embedding = torch.maximum(new_embedding, fraction_embedding)
        return new_embedding, r_argmax.cpu().numpy()
    
    def intersection(self, embeddings):
        return torch.prod(embeddings, dim=0)

    def union(self, embeddings):
        return (1. - torch.prod(1.-embeddings, dim=0))

    def embed_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        exec_query = []
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                bsz = queries.size(0)
                embedding = torch.zeros(bsz, self.nentity).to(torch.float).to(self.device)
                embedding.scatter_(-1, queries[:, idx].unsqueeze(-1), 1)
                exec_query.append(queries[:, idx].item())
                idx += 1
            else:
                embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[0], idx)
                exec_query.append(pre_exec_query)
            r_exec_query = []
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    r_exec_query.append('n')
                else:
                    r_embedding = self.relation_embeddings[queries[0, idx]]
                    if (i < len(query_structure[-1]) - 1) and query_structure[-1][i+1] == 'n':
                        embedding, r_argmax = self.relation_projection(embedding, r_embedding, True)
                    else:
                        embedding, r_argmax = self.relation_projection(embedding, r_embedding, False)
                    r_exec_query.append((queries[0, idx].item(), r_argmax))
                    r_exec_query.append('e')
                idx += 1
            r_exec_query.pop()
            exec_query.append(r_exec_query)
            exec_query.append('e')
        else:
            embedding_list = []
            union_flag = False
            for ele in query_structure[-1]:
                if ele == 'u':
                    union_flag = True
                    query_structure = query_structure[:-1]
                    break
            for i in range(len(query_structure)):
                embedding, idx, pre_exec_query = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                exec_query.append(pre_exec_query)
            if union_flag:
                embedding = self.union(torch.stack(embedding_list))
                idx += 1
                exec_query.append(['u'])
            else:
                embedding = self.intersection(torch.stack(embedding_list))
            exec_query.append('e')
        
        return embedding, idx, exec_query

    def find_ans(self, exec_query, query_structure, anchor):
        ans_structure = self.name_answer_dict[self.query_name_dict[query_structure]]
        return self.backward_ans(ans_structure, exec_query, anchor)

    def backward_ans(self, ans_structure, exec_query, anchor):
        if ans_structure == 'e': # 'e'
            return exec_query, exec_query

        elif ans_structure[0] == 'u': # 'u'
            return ['u'], 'u'
        
        elif ans_structure[0] == 'r': # ['r', 'e', 'r']
            cur_ent = anchor
            ans = []
            for ele, query_ele in zip(ans_structure[::-1], exec_query[::-1]):
                if ele == 'r':
                    r_id, r_argmax = query_ele
                    ans.append(r_id)
                    cur_ent = int(r_argmax[cur_ent])
                elif ele == 'n':
                    ans.append('n')
                else:
                    ans.append(cur_ent)
            return ans[::-1], cur_ent

        elif ans_structure[1][0] == 'r': # [[...], ['r', ...], 'e']
            r_ans, r_ent = self.backward_ans(ans_structure[1], exec_query[1], anchor)
            e_ans, e_ent = self.backward_ans(ans_structure[0], exec_query[0], r_ent)
            ans = [e_ans, r_ans, anchor]
            return ans, e_ent
            
        else: # [[...], [...], 'e']
            ans = []
            for ele, query_ele in zip(ans_structure[:-1], exec_query[:-1]):
                ele_ans, ele_ent = self.backward_ans(ele, query_ele, anchor)
                ans.append(ele_ans)
            ans.append(anchor)
            return ans, ele_ent
        

#增添的新函数
def load_kbc(model_path, device, nentity, nrelation):
    model = ComplEx(sizes=[nentity, nrelation, nentity], rank=1000, init_size=1e-3)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

@torch.no_grad()
def neural_adj_matrix(model, rel, nentity, device, thrshd, adj_list):
    bsz = 100
    softmax = nn.Softmax(dim=1)
    relation_embedding = torch.zeros(nentity, nentity).to(torch.float)
    r = torch.LongTensor([rel]).to(device)
    num = torch.zeros(nentity, 1).to(torch.float).to(device)
    for (h, t) in adj_list:
        num[h, 0] += 1
    num = torch.maximum(num, torch.ones(nentity, 1).to(torch.float).to(device))
    for s in range(0, nentity, bsz):
        t = min(nentity, s+bsz)
        h = torch.arange(s, t).to(device)
        score = kge_forward(model, h, r, device, nentity)
        normalized_score = softmax(score) * num[s:t, :]
        mask = (normalized_score >= thrshd).to(torch.float)
        normalized_score = mask * normalized_score
        relation_embedding[s:t, :] = normalized_score.to('cpu')
    return relation_embedding

@torch.no_grad()
def kge_forward(model, h, r, device, nentity):
    bsz = h.size(0)
    r = r.unsqueeze(-1).repeat(bsz, 1)
    h = h.unsqueeze(-1)
    positive_sample = torch.cat((h, r, h), dim=1)
    score = model(positive_sample, score_rhs=True, score_rel=False, score_lhs=False)
    return score[0]
