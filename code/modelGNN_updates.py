import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import scipy.sparse as sp

att_op_dict = {
    'sum': 'sum',
    'mul': 'mul',
    'concat': 'concat'
}

"""GatedGNN with residual connection"""
class ReGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op='mul', alpha_weight=1.0):
        super(ReGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            if self.residual:
                x = x + self.gatedGNN(x.double(), adj.double()) * mask.double()  # add residual connection, can use a weighted sum
            else:
                x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum/mean and max pooling

        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings  

"""GCNs with residual connection"""
class ReGCN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu,
                 residual=True, att_op="mul", alpha_weight=1.0):
        super(ReGCN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.gnnlayers.append(GraphConvolution(feature_dim_size, hidden_size, dropout, act=act))
            else:
                self.gnnlayers.append(GraphConvolution(hidden_size, hidden_size, dropout, act=act))
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act
       
    def forward(self, inputs, adj, mask):
        x = inputs
        for idx_layer in range(self.num_GNN_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](x, adj) * mask
            else:
                if self.residual:
                    x = x + self.gnnlayers[idx_layer](x, adj) * mask  # Residual Connection, can use a weighted sum
                else:
                    x = self.gnnlayers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x.double()).double())
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
      
        return graph_embeddings


"""GatedGNN"""
class GGGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, dropout, act=nn.functional.relu):
        super(GGGNN, self).__init__()
        self.num_GNN_layers = num_GNN_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_GNN_layers):
            x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        return x


""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.double(), self.weight.double())
        output = torch.matmul(adj.double(), support.double())
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


weighted_graph = False
print('using default unweighted graph')


# build graph function
def build_graph(shuffle_doc_words_list, word_embeddings, window_size=3):
    # print('using window size = ', window_size)
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    vocab_set = set()

    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p_id = window[p]
                    word_q_id = window[q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[p])
            col.append(doc_word_id_map[q])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        x_adj.append(adj)

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k])
        x_feature.append(features)

    return x_adj, x_feature


# another way to build graph from text
def build_graph_text(shuffle_doc_words_list, word_embeddings, window_size=3):

    # print('using window size = ', window_size)
    x_adj = []
    x_feature = []
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        doc_len = len(doc_words)

        row = []
        col = []
        weight = []
        features = []

        if doc_len > window_size:
            for j in range(doc_len - window_size + 1):
                for p in range(j + 1, j + window_size):
                    for q in range(j, p):
                        row.append(p)
                        col.append(q)
                        weight.append(1.)
                        #
                        row.append(q)
                        col.append(p)
                        weight.append(1.)
        else:  # doc_len < window_size
            for p in range(1, doc_len):
                for q in range(0, p):
                    row.append(p)
                    col.append(q)
                    weight.append(1.)
                    #
                    row.append(q)
                    col.append(p)
                    weight.append(1.)

        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_len, doc_len))
        if weighted_graph == False:
            adj[adj > 1] = 1.
        x_adj.append(adj)
        #
        for word in doc_words:
            feature = word_embeddings[word]
            features.append(feature)
        x_feature.append(features)

    return x_adj, x_feature