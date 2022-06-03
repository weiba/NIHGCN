import torch.nn.functional as fun
from abc import ABC
import torch.optim as optim
from myutils import *


class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cpu"):
        super(ConstructAdjMatrix, self).__init__()
        self.adj = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1)+1, -0.5))
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0)+1, -0.5))

        agg_cell_lp = torch.mm(torch.mm(d_x, self.adj), d_y)
        agg_drug_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)

        d_c = torch.pow(torch.sum(self.adj, dim=1)+1, -1)
        self_cell_lp = torch.diag(torch.add(d_c, 1))
        d_d = torch.pow(torch.sum(self.adj, dim=0)+1, -1)
        self_drug_lp = torch.diag(torch.add(d_d, 1))
        return agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp


class LoadFeature(nn.Module, ABC):
    def __init__(self, cell_exprs, drug_finger, device="cpu"):
        super(LoadFeature, self).__init__()
        cell_exprs = torch.from_numpy(cell_exprs).to(device)
        self.cell_feat = torch_z_normalized(cell_exprs,dim=1).to(device)
        self.drug_feat = torch.from_numpy(drug_finger).to(device)

    def forward(self):
        cell_feat = self.cell_feat
        drug_feat = self.drug_feat
        return cell_feat, drug_feat
 
class GEncoder(nn.Module, ABC):
    def __init__(self, agg_c_lp, agg_d_lp, self_c_lp, self_d_lp, cell_feat, drug_feat, layer_size, alpha):
        super(GEncoder, self).__init__()
        self.agg_c_lp = agg_c_lp
        self.agg_d_lp = agg_d_lp
        self.self_c_lp = self_c_lp
        self.self_d_lp = self_d_lp
        self.layers = layer_size
        self.alpha = alpha
        self.cell_feat = cell_feat
        self.drug_feat = drug_feat

        self.fc_cell = nn.Linear(self.cell_feat.shape[1], layer_size[0], bias=True)
        self.fc_drug = nn.Linear(self.drug_feat.shape[1], layer_size[0], bias=True)
        self.lc = nn.BatchNorm1d(layer_size[0])
        self.ld = nn.BatchNorm1d(layer_size[0])
        self.lm_cell = nn.Linear(layer_size[0], layer_size[1], bias=True)
        self.lm_drug = nn.Linear(layer_size[0], layer_size[1], bias=True)

    def forward(self):
        cell_fc = self.lc(self.fc_cell(self.cell_feat))
        drug_fc = self.ld(self.fc_drug(self.drug_feat))

        cell_gcn = torch.mm(self.self_c_lp, cell_fc)+torch.mm(self.agg_c_lp, drug_fc)
        drug_gcn = torch.mm(self.self_d_lp, drug_fc)+torch.mm(self.agg_d_lp, cell_fc)

        cell_ni = torch.mul(cell_gcn, cell_fc)
        drug_ni = torch.mul(drug_gcn, drug_fc)
        
        cell_emb = fun.relu(self.lm_cell((1-self.alpha)*cell_gcn + self.alpha*cell_ni))
        drug_emb = fun.relu(self.lm_drug((1-self.alpha)*drug_gcn + self.alpha*drug_ni))
        return cell_emb, drug_emb
        
class GDecoder(nn.Module, ABC):
    def __init__(self,gamma):
        super(GDecoder, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        Corr = torch_corr_x_y(cell_emb, drug_emb)
        output = scale_sigmoid(Corr, alpha=self.gamma)
        return output


class nihgcn(nn.Module, ABC):
    def __init__(self, adj_mat, cell_exprs, drug_finger, layer_size, alpha, gamma,
                 device="cpu"):
        super(nihgcn, self).__init__()
        construct_adj_matrix = ConstructAdjMatrix(adj_mat, device=device)
        loadfeat = LoadFeature(cell_exprs, drug_finger, device=device)

        agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp = construct_adj_matrix()
        cell_feat,drug_feat = loadfeat()
        self.encoder = GEncoder(agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp,
                                cell_feat, drug_feat, layer_size, alpha)
        self.decoder = GDecoder(gamma=gamma)

    def forward(self):
        cell_emb, drug_emb = self.encoder()
        output = self.decoder(cell_emb, drug_emb)
        return output


class Optimizer(nn.Module, ABC):
    def __init__(self, model, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device="cpu"):
        super(Optimizer, self).__init__()
        self.model = model.to(device)
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        best_predict = 0
        best_auc = 0
        for epoch in torch.arange(self.epochs):
            predict_data = self.model()
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)
            if auc > best_auc:
                best_auc = auc
                best_predict = torch.masked_select(predict_data, self.test_mask)
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc)
        print("Fit finished.")
        return true_data, best_predict

