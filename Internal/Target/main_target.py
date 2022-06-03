# coding: utf-8
import argparse
from load_data import load_data
from model import nihgcn, Optimizer
from sklearn.model_selection import KFold
from sampler import TargetSampler
from myutils import *

parser = argparse.ArgumentParser(description="Run NIHGCN")
parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='gdsc', help='Dataset{gdsc or ccle}')
parser.add_argument('--lr', type=float,default=0.001,
                    help="the learning rate")
parser.add_argument('--wd', type=float,default=1e-5,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--layer_size', nargs='?', default=[1024,1024],
                    help='Output sizes of every layer')
parser.add_argument('--alpha', type=float,default=0.25,
                    help="the scale for balance gcn and ni")
parser.add_argument('--gamma', type=float,default=8,
                    help="the scale for sigmod")
parser.add_argument('--epochs', type=float,default=1000,
                    help="the epochs for model")
args = parser.parse_args()

if args.data == "gdsc":
    target_drug_cids = np.array([5330286, 11338033, 24825971])
    # 加载靶点药物索引
    data_dir = dir_path(k=1) + "NIHGCN/Data/GDSC/"
    cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    cell_drug.columns = cell_drug.columns.astype(np.int32)
    drug_cids = cell_drug.columns.values
    cell_target_drug = np.array(cell_drug.loc[:, target_drug_cids], dtype=np.float32)
    target_pos_num = sp.coo_matrix(cell_target_drug).data.shape[0]
    target_indexes = common_data_index(drug_cids, target_drug_cids)

elif args.data == "ccle":
    target_drug_cids = np.array([5330286])
    # 加载靶点药物索引
    data_dir = dir_path(k=1) + "NIHGCN/Data/CCLE/"
    cell_drug = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    cell_drug.columns = cell_drug.columns.astype(np.int32)
    drug_cids = cell_drug.columns.values
    cell_target_drug = np.array(cell_drug.loc[:, target_drug_cids], dtype=np.float32)
    target_pos_num = sp.coo_matrix(cell_target_drug).data.shape[0]
    target_indexes = common_data_index(drug_cids, target_drug_cids)

#load data
res, drug_finger, exprs, null_mask, pos_num, args = load_data(args)


true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()

k = 5
n_kfolds = 5
for fold in range(n_kfolds):
    kfold = KFold(n_splits=k, shuffle=True, random_state=fold)
    for train_index, test_index in kfold.split(np.arange(target_pos_num)):
        sampler = TargetSampler(response_mat=res, null_mask=null_mask, target_indexes=target_indexes,
                          pos_train_index=train_index, pos_test_index=test_index)
        model = nihgcn(adj_mat=sampler.train_data, cell_exprs=exprs, drug_finger=drug_finger,
                       layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma,  device=args.device)
        opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                        roc_auc, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)
        true_data, predict_data = opt()
        true_datas = true_datas.append(translate_result(true_data))
        predict_datas = predict_datas.append(translate_result(predict_data))
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")
