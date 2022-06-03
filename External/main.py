# coding: utf-8
import argparse
from load_data import load_data
from sampler import ExterSampler
from model import nihgcn,Optimizer
from myutils import *

parser = argparse.ArgumentParser(description="Run NIHGCN")
parser.add_argument('-device', type=str, default="cuda:0",
                    help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='pdx',
                    help='Dataset{pdx or tcga}')
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


#load data
res, drug_finger, exprs, null_mask, train_num, args = load_data(args)

true_datas = pd.DataFrame()
predict_datas = pd.DataFrame()
n_kfolds = 25
for n_kfold in range(n_kfolds):
    train_index = np.arange(train_num)
    test_index = np.arange(res.shape[0]-train_num) + train_num
    sampler = ExterSampler(res, null_mask, train_index, test_index)
    model = nihgcn(adj_mat=sampler.train_data, cell_exprs=exprs, drug_finger=drug_finger,
                   layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device)
    opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask,
                    roc_auc, lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device).to(args.device)
    true_data, predict_data = opt()
    true_datas = true_datas.append(translate_result(true_data))
    predict_datas = predict_datas.append(translate_result(predict_data))
pd.DataFrame(true_datas).to_csv("./result_data/true_data.csv")
pd.DataFrame(predict_datas).to_csv("./result_data/predict_data.csv")

