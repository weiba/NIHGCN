from myutils import *


def load_data(args):
    if args.data == 'gdsc':
        return _load_gdsc(args)
    elif args.data == 'ccle':
        return _load_ccle(args)
    elif args.data == 'pdx':
        return _load_pdx(args)
    elif args.data == 'tcga':
        return _load_tcga(args)
    else:
        raise NotImplementedError

def _load_gdsc(args):
    args.alpha = 0.25
    args.layer_size = [1024,1024]
    data_dir = dir_path(k=1) + "NIHGCN/Data/GDSC/"
    # 加载细胞系-药物矩阵
    res = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    res = np.array(res, dtype=np.float32)
    pos_num = sp.coo_matrix(res).data.shape[0]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # 加载细胞系-基因特征矩阵
    exprs = pd.read_csv(data_dir + "cell_exprs.csv", index_col=0, header=0)
    exprs = np.array(exprs, dtype=np.float32)

    # 加载null_mask
    null_mask = pd.read_csv(data_dir + "null_mask.csv", index_col=0, header=0)
    null_mask = np.array(null_mask, dtype=np.float32)
    return res, drug_feature, exprs, null_mask, pos_num, args

def _load_ccle(args):
    args.alpha = 0.45
    args.layer_size = [512,512]
    data_dir = dir_path(k=1) + 'NIHGCN/Data/CCLE/'

    # 加载细胞系-药物矩阵
    res = pd.read_csv(data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    res = np.array(res, dtype=np.float32)
    pos_num = sp.coo_matrix(res).data.shape[0]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # 加载细胞系-基因特征矩阵
    exprs = pd.read_csv(data_dir + "cell_exprs.csv", index_col=0, header=0)
    exprs = np.array(exprs, dtype=np.float32)

    # 加载null_mask
    null_mask = np.zeros(res.shape, dtype=np.float32)
    return res, drug_feature, exprs, null_mask, pos_num, args


def _load_pdx(args):
    args.alpha = 0.15
    args.layer_size = [1024,1024]
    pdx_data_dir = dir_path(k=1) + "NIHGCN/Data/PDX/"
    gdsc_data_dir = dir_path(k=1) + "NIHGCN/Data/GDSC/"

    # 加载GDSC细胞系-药物矩阵
    gdsc_res = pd.read_csv(gdsc_data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    gdsc_res = np.array(gdsc_res, dtype=np.float32)
    # 加载PDX病人-药物矩阵
    pdx_res = pd.read_csv(pdx_data_dir + "pdx_response.csv", index_col=0, header=0)
    pdx_res = np.array(pdx_res, dtype=np.float32)
    # 合并GDSC-PDX反应矩阵
    res = np.concatenate((gdsc_res, pdx_res), axis=0)
    train_row = gdsc_res.shape[0]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(gdsc_data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # 加载GDSC细胞系-基因特征矩阵
    gdsc_exprs = pd.read_csv(gdsc_data_dir + "cell_exprs.csv", index_col=0, header=0)
    # 加载PDX病人-基因特征矩阵
    pdx_exprs = pd.read_csv(pdx_data_dir + "pdx_exprs.csv", index_col=0, header=0)
    # 取GDSC-PDX共同基因
    common_gene_gdsc = gdsc_exprs.columns.isin(pdx_exprs.columns)
    common_gene_tcga = pdx_exprs.columns.isin(gdsc_exprs.columns)
    gdsc_exprs = gdsc_exprs.loc[:, common_gene_gdsc]
    gdsc_exprs = np.array(gdsc_exprs, dtype=np.float32)
    pdx_exprs = pdx_exprs.loc[:, common_gene_tcga]
    pdx_exprs = np.array(pdx_exprs, dtype=np.float32)
    # 合并GDSC-PDX基因特征矩阵
    exprs = np.concatenate((gdsc_exprs, pdx_exprs), axis=0)

    # 加载GDSC null_mask
    gdsc_null_mask = pd.read_csv(gdsc_data_dir + "null_mask.csv", index_col=0, header=0)
    gdsc_null_mask = np.array(gdsc_null_mask, dtype=np.float32)
    # 加载PDX null_mask
    pdx_null_mask = pd.read_csv(pdx_data_dir + "pdx_null_mask.csv", index_col=0, header=0)
    pdx_null_mask = np.array(pdx_null_mask, dtype=np.float32)
    # 合并GDSC-PDX null_mask
    null_mask = np.concatenate((gdsc_null_mask, pdx_null_mask), axis=0)
    return res, drug_feature, exprs, null_mask, train_row, args


def _load_tcga(args):
    args.alpha = 0.1
    args.layer_size = [1024,1024]
    tcga_data_dir = dir_path(k=1) + "NIHGCN/Data/TCGA/"
    gdsc_data_dir = dir_path(k=1) + "NIHGCN/Data/GDSC/"

    # 加载GDSC细胞系-药物矩阵
    gdsc_res = pd.read_csv(gdsc_data_dir + "cell_drug_binary.csv", index_col=0, header=0)
    gdsc_res = np.array(gdsc_res, dtype=np.float32)
    # 加载TCGA病人-药物矩阵
    tcga_res = pd.read_csv(tcga_data_dir + "patient_drug_binary.csv", index_col=0, header=0)
    tcga_res = np.array(tcga_res, dtype=np.float32)
    # 合并GDSC-TCGA反应矩阵
    res = np.concatenate((gdsc_res, tcga_res), axis=0)
    train_row = gdsc_res.shape[0]

    # 加载药物-指纹特征矩阵
    drug_feature = pd.read_csv(gdsc_data_dir + "drug_feature.csv", index_col=0, header=0)
    drug_feature = np.array(drug_feature, dtype=np.float32)

    # 加载GDSC细胞系-基因特征矩阵
    gdsc_exprs = pd.read_csv(gdsc_data_dir + "cell_exprs.csv", index_col=0, header=0)
    # 加载TCGA病人-基因特征矩阵
    patient_gene = pd.read_csv(tcga_data_dir + "tcga_exprs.csv", index_col=0, header=0)
    # 取GDSC-TCGA共同基因
    common_gene_gdsc = gdsc_exprs.columns.isin(patient_gene.columns)
    common_gene_tcga = patient_gene.columns.isin(gdsc_exprs.columns)
    gdsc_exprs = gdsc_exprs.loc[:, common_gene_gdsc]
    patient_gene = patient_gene.loc[:, common_gene_tcga]
    gdsc_exprs = np.array(gdsc_exprs, dtype=np.float32)
    patient_gene = np.array(patient_gene, dtype=np.float32)
    # 合并GDSC-TCGA基因特征矩阵
    exprs = np.concatenate((gdsc_exprs, patient_gene), axis=0)

    # 加载GDSC null_mask
    gdsc_null_mask = pd.read_csv(gdsc_data_dir + "null_mask.csv", index_col=0, header=0)
    gdsc_null_mask = np.array(gdsc_null_mask, dtype=np.float32)
    # 加载TCGA null_mask
    tcga_null_mask = pd.read_csv(tcga_data_dir + "tcga_null_mask.csv", index_col=0, header=0)
    tcga_null_mask = np.array(tcga_null_mask, dtype=np.float32)
    # 合并GDSC-TCGA null_mask
    null_mask = np.concatenate((gdsc_null_mask, tcga_null_mask), axis=0)
    return res, drug_feature, exprs, null_mask, train_row, args


