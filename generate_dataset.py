import os
import argparse
import abc
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from typing import Union
from tqdm import tqdm

ONE_DAY = 24 * 60 * 60

class DataGenerator(metaclass=abc.ABCMeta):
    def __init__(self, config):
        self.config = config
        self.feature_path = self.config.feature_path
        self.train_data_path = self.config.train_data_path
        self.test_data_path = self.config.test_data_path
        self.ticket_path = self.config.ticket_path
        self.temp_path = os.path.join(self.config.train_data_path, "temp")  # 临时目录
        os.makedirs(self.temp_path, exist_ok=True)

        self.train_start_date = self._datetime_to_timestamp(self.config.train_date_range[0])
        self.train_end_date = self._datetime_to_timestamp(self.config.train_date_range[1])
        self.test_start_date = self._datetime_to_timestamp(self.config.test_date_range[0])
        self.test_end_date = self._datetime_to_timestamp(self.config.test_date_range[1])

        ticket = pd.read_csv(self.ticket_path)
        self.ticket = ticket[ticket["alarm_time"] <= self.train_end_date]
        self.ticket_sn_map = dict(zip(self.ticket["sn_name"], self.ticket["alarm_time"]))

        os.makedirs(self.config.train_data_path, exist_ok=True)
        os.makedirs(self.config.test_data_path, exist_ok=True)

    @staticmethod
    def _datetime_to_timestamp(date: str) -> int:
        return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

    def _get_data(self) -> pd.DataFrame:
        file_list = os.listdir(self.feature_path)
        file_list = [x for x in file_list if x.endswith(".feather")]
        file_list.sort()

        batch_size = 1000
        data_all = []
        cnt = 0

        for i in tqdm(range(0, len(file_list), batch_size), desc="Processing batches"):
            batch_files = file_list[i:i + batch_size]
            with Pool(processes=self.config.number_workers) as pool:
                results = list(pool.imap(self._process_file, batch_files))
            batch_data = [result for result in results if result is not None]
            if batch_data:
                cnt += len(batch_data)
                batch_df = pd.concat(batch_data)
                data_all.append(batch_df)
                
        data_all = pd.concat(data_all)
        print("--- Number of valid files: {}/{}".format(cnt, len(file_list)))
        print("--- Number of valid data points: ", data_all.shape[0])
        print("--- Number of features: ", data_all.shape[1])
        return data_all

    @abc.abstractmethod
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        pass

    @abc.abstractmethod
    def generate_and_save_data(self):
        pass
    
    def load_data(self, path):
        data = pd.read_feather(path)
        # drop col with "6h" in the col names
        data.drop(columns=[col for col in data.columns if "6h" in col], inplace=True)
        return data
    
class PositiveDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        sn_name = os.path.splitext(sn_file)[0]
        end_time = self.ticket_sn_map.get(sn_name)
        if not end_time:
            return None
        start_time = end_time - 7 * ONE_DAY
        data = self.load_data(os.path.join(self.feature_path, sn_file))
        data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
        if data.empty:
            return None
        
        data["label"] = 1
        data.index = pd.MultiIndex.from_tuples([(sn_name, log_time) for log_time in data["LogTime"]])
        data = data.drop(columns=["LogTime"])
        return data

    def generate_and_save_data(self):
        """
        Generate and save positive sample data.
        """
        print("Generating positive data...")
        data_all = self._get_data()
        save_path = os.path.join(self.train_data_path, "positive_train.feather")
        # save use pandas's 
        data_all.to_feather(save_path)
        print("Positive data generated and saved to {}".format(save_path))

class NegativeDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        Process a single file to extract negative sample data.

        :param sn_file: File name
        :return: Processed DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        if not self.ticket_sn_map.get(sn_name):
            data = self.load_data(os.path.join(self.feature_path, sn_file))
            end_time = self.train_end_date - 7 * ONE_DAY
            start_time = self.train_end_date - 14 * ONE_DAY
            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            if data.empty:
                return None
            data["label"] = 0

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            # drop LogTime column
            data = data.drop(columns=["LogTime"])
            return data
        else:
            return None

    def generate_and_save_data(self):
        """
        Generate and save negative sample data.
        """
        print("Generating negative data...")
        data_all = self._get_data()
        save_path = os.path.join(self.train_data_path, "negative_train.feather")
        data_all.to_feather(save_path)
        print("Negative data generated and saved to {}".format(save_path))


class TestDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        Process a single file to extract test data.

        :param sn_file: File name
        :return: Processed DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        data = self.load_data(os.path.join(self.feature_path, sn_file))
        data = data[data["LogTime"] >= self.test_start_date]
        data = data[data["LogTime"] <= self.test_end_date]
        if data.empty:
            return None

        index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
        data.index = pd.MultiIndex.from_tuples(index_list)
        data = data.drop(columns=["LogTime"])
        return data

    def generate_and_save_data(self):
        """
        Generate and save test data.
        """
        print("Generating test data...")
        data_all = self._get_data()
        save_path = os.path.join(self.test_data_path, "test_data.feather")
        data_all.to_feather(save_path)
        print("Test data generated and saved to {}".format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart Memory')
    parser.add_argument('--test_stage', type=int, default=0)
    parser.add_argument('--path2features', type=str, default='/home/fei/research/smartmem/dataset/features')
    parser.add_argument('--path2dataset', type=str, default='/home/fei/research/smartmem/dataset/train_test_data')
    parser.add_argument('--ticket_path', type=str, default='/home/fei/research/smartmem/dataset/failure_ticket.csv')
    parser.add_argument('--number_workers', type=int, default=100, help='number of workers')
    args = parser.parse_args()
    
    args.train_date_range = ("2024-01-01", "2024-06-01")
    if args.test_stage == 0:
        args.test_date_range = ("2024-06-01", "2024-08-01")
    elif args.test_stage == 1:
        args.test_date_range = ("2024-08-01", "2024-10-01")
    else:
        raise ValueError("Invalid test stage {}".format(args.test_stage))
    
    for sn_type in ["A","B"]:
        args.feature_path = os.path.join(args.path2features, "type_{}".format(sn_type))
        args.train_data_path = os.path.join(args.path2dataset, "train_data", "type_{}".format(sn_type))
        args.test_data_path = os.path.join(args.path2dataset, "test_data", "type_{}_{}".format(sn_type, args.test_stage))
    
        os.makedirs(args.train_data_path, exist_ok=True)
        os.makedirs(args.test_data_path, exist_ok=True)
        
        positive_generator = PositiveDataGenerator(args)
        positive_generator.generate_and_save_data()
        
        negative_generator = NegativeDataGenerator(args)
        negative_generator.generate_and_save_data()
        
        test_generator = TestDataGenerator(args)
        test_generator.generate_and_save_data()