from model import nihgcn, Optimizer
from sampler import SingleSampler


def nihgcn_single(cell_exprs, drug_finger, res_mat, null_mask, target_index,
                         train_index, test_index, evaluate_fun, args):

    sample = SingleSampler(res_mat, null_mask, target_index, train_index, test_index)
    model = nihgcn(adj_mat=sample.train_data, cell_exprs=cell_exprs, drug_finger=drug_finger,
    layer_size=args.layer_size, gamma=args.gamma, alpha=args.alpha, device=args.device)
    opt = Optimizer(model, sample.train_data, sample.test_data, sample.test_mask, sample.train_mask, evaluate_fun,
                    lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device)
    true_data, predict_data = opt()
    return true_data, predict_data
