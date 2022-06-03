# coding: utf-8
import argparse
from load_data import load_data
from sklearn.model_selection import KFold
from Internal.Single.NIHGCN_Single import nihgcn_single
from myutils import *

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
drug_sum = np.sum(res, axis=0)

k = 5
n_kfolds = 5

for target_index in np.arange(res.shape[1]):
    times = []
    true_data_s = pd.DataFrame()
    predict_data_s = pd.DataFrame()
    target_pos_index = np.where(res[:, target_index] == 1)[0]
    if drug_sum[target_index] < 10:
        continue
    for fold in range(n_kfolds):
        kfold = KFold(n_splits=k, shuffle=True, random_state=fold)
        start = time.time()
        for train, test in kfold.split(target_pos_index):
            train_index = target_pos_index[train]
            test_index = target_pos_index[test]
            true_data, predict_data = nihgcn_single(cell_exprs=exprs,
                                                              drug_finger=drug_finger, res_mat=res,
                                                              null_mask=null_mask, target_index=target_index,
                                                              train_index=train_index, test_index=test_index,
                                                              evaluate_fun=roc_auc, args=args)
            true_data_s = true_data_s.append(translate_result(true_data))
            predict_data_s = predict_data_s.append(translate_result(predict_data))
        end = time.time()
        times.append(end - start)
    true_data_s.to_csv("./pan_result_data/noparallel_drug_" + str(target_index) + "_" + "true_data.csv")
    predict_data_s.to_csv("./pan_result_data/noparallel_drug_" + str(target_index) + "_" + "predict_data.csv")

