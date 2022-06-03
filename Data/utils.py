import os
import re
import math
import zipfile
import pandas as pd


class CsvLimitSizeZip(object):
    def __init__(self, path: str, size: int or float = 50):
        self.path = path
        self.size = size

    def __get_file_name(self):
        return self.path.split(sep='/')[-1].rstrip('.csv')

    def __get_file_size(self, flag: str = "M"):
        unit = {'B': 0, 'K': 10, 'M': 20, 'G': 30}
        size = os.path.getsize(filename=self.path)
        return size / math.pow(2, unit[flag])

    def __get_file_save_dir(self):
        dir_name = self.path.rstrip('.csv')
        if not os.path.exists(dir_name):
            os.mkdir(path=dir_name)
        return dir_name + "/"

    @staticmethod
    def __split_seq(seq: list, number: int):
        result = []
        length = len(seq)
        each_num = length // number
        start_num = 0
        end_num = start_num + each_num
        while number > 1:
            sub_seq = seq[start_num: end_num]
            result.append(sub_seq)
            length -= each_num
            number -= 1
            each_num = length // number
            start_num = end_num
            end_num = start_num + each_num
        result.append(seq[start_num:])
        return result

    @staticmethod
    def __split_row_data(data: pd.DataFrame, number: int):
        seq = data.index.values
        result = CsvLimitSizeZip.__split_seq(seq=list(seq), number=number)
        datas = [data.loc[result[i], :] for i in range(number)]
        return datas

    @staticmethod
    def __split_col_data(data: pd.DataFrame, number: int):
        seq = data.columns.values
        result = CsvLimitSizeZip.__split_seq(seq=list(seq), number=number)
        datas = [data.loc[:, result[i]] for i in range(number)]
        return datas

    @staticmethod
    def __zip_file(path, data: pd.DataFrame):
        data.to_csv(path)
        zip_file = zipfile.ZipFile(str(path) + ".zip", "w", zipfile.ZIP_DEFLATED)
        zip_file.write(path, path.split(sep='/')[-1])
        zip_file.close()
        os.remove(path=path)

    @staticmethod
    def __uzip_file(path_dir: str, file_name: str):
        path = path_dir + file_name
        zip_file = zipfile.ZipFile(path, 'r', zipfile.ZIP_DEFLATED)
        zip_file.extractall(path_dir)
        zip_file.close()
        data = pd.read_csv(path.rstrip('.zip'), index_col=0, header=0)
        return data

    @staticmethod
    def _uzip_file(path_dir: str, file_names: list):
        file_names = [name for name in file_names if name.endswith('.zip')]
        datas = [CsvLimitSizeZip.__uzip_file(path_dir=path_dir, file_name=name) for name in file_names]
        return datas

    @staticmethod
    def __get_dim_from_name(path_dir: str):
        file_names = os.listdir(path_dir)
        file_names = [name for name in file_names if name.endswith('.zip')]
        rule = 'dim_(?P<dim>[0|1])'
        dim = int(re.search(rule, file_names[0]).group('dim'))
        return dim

    def split_file(self):
        file_size = self.__get_file_size()
        if file_size > self.size:
            split_number = math.ceil(file_size/self.size)
            data = pd.read_csv(self.path, index_col=0, header=0)
            dim = 0 if data.shape[0] > data.shape[1] else 1
            path = self.__get_file_save_dir() + self.__get_file_name()
            fun_map = {0: CsvLimitSizeZip.__split_row_data, 1: CsvLimitSizeZip.__split_col_data}
            datas = fun_map[dim](data=data, number=split_number)
            paths = [path + '_dim_{}_num_{}.csv'.format(dim, i) for i in range(split_number)]
            for ph, dt in zip(paths, datas):
                CsvLimitSizeZip.__zip_file(path=ph, data=dt)

    @staticmethod
    def merge_file(path_dir: str):
        file_names = os.listdir(path_dir)
        datas = CsvLimitSizeZip._uzip_file(path_dir=path_dir, file_names=file_names)
        dim = CsvLimitSizeZip.__get_dim_from_name(path_dir=path_dir)
        data = pd.concat(datas, axis=dim)
        return data


if __name__ == '__main__':
    gdsc_dir = './GDSC/processed_data/'
    ccle_dir = './CCLE/processed_data/'
    for name in os.listdir(gdsc_dir):
        path = gdsc_dir + name
        print(path)
        zip_trans = CsvLimitSizeZip(path=path)
        zip_trans.split_file()
    for name in os.listdir(ccle_dir):
        path = ccle_dir + name
        print(path)
        zip_trans = CsvLimitSizeZip(path=path)
        zip_trans.split_file()

    # zip_file = CsvLimitSizeZip(path=path)
    # data = zip_file.merge_file(path_dir=path.rstrip('.csv') + '/')






    
