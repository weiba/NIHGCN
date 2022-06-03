from model import nihgcn, Optimizer
from sampler import NewSampler

                            
def nihgcn_new(cell_exprs, drug_finger, res_mat, null_mask, target_dim, target_index,
               evaluate_fun, args):

    sampler = NewSampler(res_mat, null_mask, target_dim, target_index)
    model = nihgcn(sampler.train_data, cell_exprs=cell_exprs, drug_finger=drug_finger,
                   layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, device=args.device)
    opt = Optimizer(model, sampler.train_data, sampler.test_data, sampler.test_mask, sampler.train_mask, evaluate_fun,
                    lr=args.lr, wd=args.wd, epochs=args.epochs, device=args.device)
    true_data, predict_data,auc_data = opt()
    return true_data, predict_data
