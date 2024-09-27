import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.disease_dim = config['disease_dim']
        self.drug_dim = config['drug_dim']
        self.disease_size = config['disease_size']
        self.drug_size = config['drug_size']
        self.latent_dim = config['latent_dim']
        self.attention_flag = config['attention_flag']
        self.atten_dim = config['atten_dim']
        self.l2 = config['l2']
        self.mlp_layer_num = config['mlp_layer_num']


        self.disease_embedding = nn.Embedding(self.disease_size, self.disease_dim)
        self.drug_embedding = nn.Embedding(self.drug_size, self.drug_dim)

     
        self.disease_gcn_w = nn.Linear(self.drug_dim, self.latent_dim)
        self.disease_gcn_b = nn.Parameter(torch.zeros(self.latent_dim))
        self.drug_gcn_w = nn.Linear(self.disease_dim, self.latent_dim)
        self.drug_gcn_b = nn.Parameter(torch.zeros(self.latent_dim))

  
        if self.attention_flag:
            self.disease_attention_w1 = nn.Linear(self.disease_dim + self.drug_dim, self.atten_dim)
            self.disease_attention_w2 = nn.Linear(self.atten_dim, 1)
            self.drug_attention_w1 = nn.Linear(self.disease_dim + self.drug_dim, self.atten_dim)
            self.drug_attention_w2 = nn.Linear(self.atten_dim, 1)

        if self.config['disease_knn_number'] > 0:
            self.disease_w2 = nn.Linear(self.disease_dim, self.latent_dim)
        if self.config['drug_knn_number'] > 0:
            self.drug_w3 = nn.Linear(self.drug_dim, self.latent_dim)


        mlp_layers = []
        input_dim = self.latent_dim  
        for _ in range(self.mlp_layer_num):
            mlp_layers.append(nn.Linear(input_dim, self.disease_dim))
            mlp_layers.append(nn.SELU())
            input_dim = self.disease_dim 
        self.mlp = nn.Sequential(*mlp_layers)

   
        self.prediction_layer = nn.Linear(self.disease_dim, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def attention_gcn(self, adj, node_embedding_other, node_embedding_self, type='disease'):
        query = node_embedding_self.unsqueeze(1).expand(-1, node_embedding_other.size(0), -1)
        key = node_embedding_other.unsqueeze(0).expand(node_embedding_self.size(0), -1, -1)
        key_query = torch.cat([key, query], dim=-1)
        key_query = key_query.view(-1, key_query.size(-1))
        if type == 'disease':
            alpha = F.relu(self.disease_attention_w1(key_query))
            alpha = F.relu(self.disease_attention_w2(alpha))
        else:
            alpha = F.relu(self.drug_attention_w1(key_query))
            alpha = F.relu(self.drug_attention_w2(alpha))
        alpha = alpha.view(node_embedding_self.size(0), node_embedding_other.size(0))
        alpha = alpha * adj
        alpha_exps = F.softmax(alpha, dim=1)
        e_r = torch.matmul(alpha_exps, node_embedding_other)
        return e_r

    def simple_gcn(self, adj, node_embedding_other):
        edges = torch.matmul(adj, node_embedding_other)
        return edges

    def forward(self, e_p_adj, e_e_adj, p_p_adj, input_disease, input_drug, labels):
        disease_embeds = self.disease_embedding.weight
        drug_embeds = self.drug_embedding.weight

        if self.attention_flag:

            h_e_gcn = self.attention_gcn(e_p_adj, drug_embeds, disease_embeds, type='disease')
            h_e_gcn = self.disease_gcn_w(h_e_gcn) + self.disease_gcn_b
            h_p_gcn = self.attention_gcn(e_p_adj.t(), disease_embeds, drug_embeds, type='drug')
            h_p_gcn = self.drug_gcn_w(h_p_gcn) + self.drug_gcn_b
        else:

            h_e_gcn = self.simple_gcn(e_p_adj, drug_embeds)
            h_e_gcn = self.disease_gcn_w(h_e_gcn) + self.disease_gcn_b
            h_p_gcn = self.simple_gcn(e_p_adj.t(), disease_embeds)
            h_p_gcn = self.drug_gcn_w(h_p_gcn) + self.drug_gcn_b


        if self.config['disease_knn_number'] > 0:
            disease_edges = e_e_adj.sum(dim=1, keepdim=True)
            disease_edges = disease_edges + 1e-8 
            ave_disease_edges = torch.matmul(e_e_adj, disease_embeds) / disease_edges
            h_e_e = self.disease_w2(ave_disease_edges)
            h_e = F.selu(h_e_gcn + h_e_e)
        else:
            h_e = F.selu(h_e_gcn)

        if self.config['drug_knn_number'] > 0:
            drug_edges = p_p_adj.sum(dim=1, keepdim=True)
            drug_edges = drug_edges + 1e-8  
            ave_drug_edges = torch.matmul(p_p_adj, drug_embeds) / drug_edges
            h_p_p = self.drug_w3(ave_drug_edges)
            h_p = F.selu(h_p_gcn + h_p_p)
        else:
            h_p = F.selu(h_p_gcn)

        h_e_1 = h_e[input_disease]
        h_p_1 = h_p[input_drug]

        input_temp = h_e_1 * h_p_1

        input_temp = self.mlp(input_temp)

        z = self.prediction_layer(input_temp).squeeze()

        loss = self.criterion(z, labels.float())

        return loss, torch.sigmoid(z)
