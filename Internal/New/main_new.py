# coding: utf-8
import argparse
from load_data import load_data
from myutils import *
from Internal.New.NIHGCN_New import nihgcn_new

parser = argparse.ArgumentParser(description="Run NIHGCN")
parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
parser.add_argument('-data', type=str, default='gdsc', help='Dataset{gdsc,ccle}')
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

res, drug_finger, exprs, null_mask, pos_num, args = load_data(args)
cell_sum = np.sum(res, axis=1)
drug_sum = np.sum(res, axis=0)

target_dim = [0, 1]

n_kfold = 1
for dim in target_dim:
    for target_index in np.arange(res.shape[dim]):
        print(dim)
        if dim:
            if drug_sum[target_index] < 10:
                continue
        else:
            if cell_sum[target_index] < 10:
                continue
        epochs = []
        true_data_s = pd.DataFrame()
        predict_data_s = pd.DataFrame()
        for fold in range(n_kfold):
            true_data, predict_data = nihgcn_new(cell_exprs=exprs, drug_finger=drug_finger,
                                                                res_mat=res,null_mask=null_mask,
                                                                target_dim=dim,target_index=target_index,
                                                                evaluate_fun=roc_auc, args=args)
            true_data_s = true_data_s.append(translate_result(true_data))
            predict_data_s = predict_data_s.append(translate_result(predict_data))
        if dim:
            true_data_s.to_csv("./result_data/drug_" + str(target_index) + "_true_data.csv")
            predict_data_s.to_csv("./result_data/drug_" + str(target_index) + "_predict_data.csv")
        else:
            true_data_s.to_csv("./result_data/cell_" + str(target_index) + "_true_data.csv")
            predict_data_s.to_csv("./result_data/cell_" + str(target_index) + "_predict_data.csv")
