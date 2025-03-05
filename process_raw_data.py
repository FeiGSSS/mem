import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging

# 常量定义
IMPUTE_VALUE: int = -1
FEATURE_INTERVAL: int = 3600
TIME_RELATED_LIST: List[int] = [15 * 60, 60 * 60, 24 * 60 * 60]
TIME_WINDOW_SIZE_MAP: Dict[int, str] = {15 * 60: "15m", 60 * 60: "1h", 24 * 60 * 60: "1d"}
DQ_COUNT: int = 4
BURST_COUNT: int = 8

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Preprocess the Raw Data")

class FeatureFactory:
    def __init__(self, raw_data_path: str, save_feather_path: str, number_workers: int):
        """初始化 FeatureFactory 类。

        Args:
            raw_data_path (str): 原始数据路径。
            save_feather_path (str): 特征保存路径。
            number_workers (int): 多进程工作数。
        """
        self.raw_data_path = raw_data_path
        self.save_feather_path = save_feather_path
        self.number_workers = number_workers

    def process_all_sn(self) -> None:
        """处理所有 SN 文件，使用多进程并行执行。"""
        sn_files = list(Path(self.raw_data_path).rglob("*.feather"))
        sn_files.sort()
        logger.info(f"Preprocessing {len(sn_files)} SN files...")

        # with Pool(processes=self.number_workers) as pool:
        #     for sn_file in sn_files:
        #         pool.apply_async(
        #             self.process_single_sn,
        #             (sn_file,),
        #             error_callback=lambda e: logger.error(f"Error processing {sn_file}: {e}")
        #         )
        #     pool.close()
        #     pool.join()
        # 利用 imap + tqdm 进度条
        with Pool(processes=self.number_workers) as pool:
            for _ in tqdm(pool.imap_unordered(self.process_single_sn, sn_files), total=len(sn_files)):
                pass

    def process_single_sn(self, sn_file: str) -> None:
        """处理单个 SN 文件，提取不同时间窗口的特征并保存。

        Args:
            sn_file (str): SN 文件路径。
        """
        raw_df = pd.read_feather(sn_file).sort_values("LogTime").reset_index(drop=True)
        processed_df = self.preprocess_data(raw_df)
        processed_df = self.process_parity_info(processed_df)
        time_windows = self.calculate_time_windows(processed_df)
        feature_list = self.extract_features(processed_df, time_windows)
        self.save_features(feature_list, sn_file)

    def preprocess_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """预处理原始数据，提取必要列并生成新特征。

        Args:
            raw_df (pd.DataFrame): 原始 DataFrame。

        Returns:
            pd.DataFrame: 预处理后的 DataFrame。
        """
        df = raw_df[["LogTime", "deviceID", "BankId", "RowId", "ColumnId", "MciAddr", 
                     "RetryRdErrLogParity", "error_type_full_name"]].copy()
        df["deviceID"] = df["deviceID"].fillna(IMPUTE_VALUE).astype(int)
        df["error_type_is_READ_CE"] = (df["error_type_full_name"] == "CE.READ").astype(int)
        df["error_type_is_SCRUB_CE"] = (df["error_type_full_name"] == "CE.SCRUB").astype(int)
        df["CellId"] = df["RowId"].astype(str) + "_" + df["ColumnId"].astype(str)
        df["position_and_parity"] = df.apply(
            lambda row: "_".join(map(str, [row["deviceID"], row["BankId"], row["RowId"], 
                                          row["ColumnId"], row["RetryRdErrLogParity"] or ""])),
            axis=1
        )
        return df

    def process_parity_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理 parity 信息，生成 bit/dq/burst 相关特征。

        Args:
            df (pd.DataFrame): 预处理后的 DataFrame。

        Returns:
            pd.DataFrame: 添加 parity 特征的 DataFrame。
        """
        parity_array = df["RetryRdErrLogParity"].fillna(0).replace("", 0).astype(np.int64)
        unique_parities = np.unique(parity_array)
        parity_dict = {p: self._get_bit_dq_burst_info(p) for p in unique_parities}
        parity_info = [parity_dict[p] for p in parity_array]
        parity_df = pd.DataFrame(parity_info, columns=["bit_count", "dq_count", "burst_count", 
                                                       "max_dq_interval", "max_burst_interval"])
        return pd.concat([df, parity_df], axis=1)

    def calculate_time_windows(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """计算时间窗口的起止索引和结束时间。

        Args:
            df (pd.DataFrame): 处理后的 DataFrame。

        Returns:
            List[Tuple[int, int, float]]: 每个窗口的 (开始索引, 结束索引, 结束时间)。
        """
        df["time_index"] = (df["LogTime"] // FEATURE_INTERVAL).astype(int)
        time_groups = df.groupby("time_index")["LogTime"].max()
        window_end_times = time_groups.values
        max_window_size = max(TIME_RELATED_LIST)
        window_start_times = window_end_times - max_window_size
        log_times = df["LogTime"].values
        start_indices = np.searchsorted(log_times, window_start_times, side="left")
        end_indices = np.searchsorted(log_times, window_end_times, side="right")
        return list(zip(start_indices, end_indices, window_end_times))

    def extract_features(self, df: pd.DataFrame, time_windows: List[Tuple[int, int, float]]) -> List[Dict]:
        """提取每个时间窗口的特征。

        Args:
            df (pd.DataFrame): 处理后的 DataFrame。
            time_windows (List[Tuple[int, int, float]]): 时间窗口列表。

        Returns:
            List[Dict]: 每个窗口的特征字典列表。
        """
        feature_list = []
        for start_idx, end_idx, end_time in time_windows:
            window_df = df.iloc[start_idx:end_idx].copy()
            window_df["Count"] = window_df.groupby("position_and_parity")["position_and_parity"].transform("count")
            window_df = window_df.drop_duplicates(subset="position_and_parity", keep="first")
            log_times = window_df["LogTime"].values
            end_logtime = log_times.max()
            features = {"LogTime": end_logtime}
            
            for window_size in TIME_RELATED_LIST:
                idx = np.searchsorted(log_times, end_logtime - window_size, side="left")
                subset_df = window_df.iloc[idx:]
                temporal = self._get_temporal_features(subset_df, window_size)
                spatio = self._get_spatio_features(subset_df)
                parity = self._get_err_parity_features(subset_df)
                suffix = TIME_WINDOW_SIZE_MAP[window_size]
                for feat_dict in [temporal, spatio, parity]:
                    for key, value in feat_dict.items():
                        features[f"{key}_{suffix}"] = value
            feature_list.append(features)
        return feature_list

    def save_features(self, feature_list: List[Dict], sn_file: str) -> None:
        """将特征保存为 CSV 文件。

        Args:
            feature_list (List[Dict]): 特征字典列表。
            sn_file (str): 原始 SN 文件路径。
        """
        feature_df = pd.DataFrame(feature_list)
        output_file = os.path.join(self.save_feather_path, os.path.basename(sn_file))
        # 文件夹不存在则创建
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # 保存为 feather 文件
        feature_df.to_feather(output_file)

    @staticmethod
    def _get_bit_dq_burst_info(parity: np.int64) -> Tuple[int, int, int, int, int]:
        """提取 parity 的 bit/dq/burst 信息。

        Args:
            parity (np.int64): Parity 值。

        Returns:
            Tuple[int, int, int, int, int]: (bit_count, dq_count, burst_count, max_dq_interval, max_burst_interval)。
        """
        bin_parity = bin(parity)[2:].zfill(32)
        bit_count = bin_parity.count("1")
        binary_row_array = [bin_parity[i:i+4].count("1") for i in range(0, 32, 4)]
        binary_row_array_indices = [idx for idx, value in enumerate(binary_row_array) if value > 0]
        burst_count = len(binary_row_array_indices)
        max_burst_interval = binary_row_array_indices[-1] - binary_row_array_indices[0] if binary_row_array_indices else 0
        binary_column_array = [bin_parity[i::4].count("1") for i in range(4)]
        binary_column_array_indices = [idx for idx, value in enumerate(binary_column_array) if value > 0]
        dq_count = len(binary_column_array_indices)
        max_dq_interval = binary_column_array_indices[-1] - binary_column_array_indices[0] if binary_column_array_indices else 0
        return bit_count, dq_count, burst_count, max_dq_interval, max_burst_interval

    def _get_temporal_features(self, window_df: pd.DataFrame, time_window_size: int) -> Dict[str, float]:
        """提取时间相关特征。

        Args:
            window_df (pd.DataFrame): 时间窗口内的数据。
            time_window_size (int): 时间窗口大小（秒）。

        Returns:
            Dict[str, float]: 时间特征字典。
        """
        error_type_read = window_df["error_type_is_READ_CE"]
        error_type_scrub = window_df["error_type_is_SCRUB_CE"]
        ce_count = window_df["Count"]
        return {
            "read_ce_log_num": error_type_read.sum(),
            "scrub_ce_log_num": error_type_scrub.sum(),
            "all_ce_log_num": len(window_df),
            "read_ce_count": (error_type_read * ce_count).sum(),
            "scrub_ce_count": (error_type_scrub * ce_count).sum(),
            "all_ce_count": ce_count.sum(),
            "log_happen_frequency": len(window_df) / time_window_size,
            "ce_storm_count": self._calculate_ce_storm_count(window_df["LogTime"].values)
        }

    def _get_spatio_features(self, window_df: pd.DataFrame) -> Dict[str, int]:
        """提取空间相关特征。

        Args:
            window_df (pd.DataFrame): 时间窗口内的数据。

        Returns:
            Dict[str, int]: 空间特征字典。
        """
        spatio_features = {
            "fault_mode_others": 0, "fault_mode_device": 0, "fault_mode_bank": 0,
            "fault_mode_row": 0, "fault_mode_column": 0, "fault_mode_cell": 0,
            "fault_row_num": 0, "fault_column_num": 0
        }
        unique_devices = window_df["deviceID"].nunique()
        unique_banks = window_df["BankId"].nunique()
        unique_rows = window_df["RowId"].nunique()
        unique_columns = window_df["ColumnId"].nunique()
        unique_cells = window_df["CellId"].nunique()

        if unique_devices > 1:
            spatio_features["fault_mode_others"] = 1
        elif unique_banks > 1:
            spatio_features["fault_mode_device"] = 1
        elif unique_columns > 1 and unique_rows > 1:
            spatio_features["fault_mode_bank"] = 1
        elif unique_columns > 1:
            spatio_features["fault_mode_row"] = 1
        elif unique_rows > 1:
            spatio_features["fault_mode_column"] = 1
        elif unique_cells == 1:
            spatio_features["fault_mode_cell"] = 1

        row_pos_dict = defaultdict(list)
        col_pos_dict = defaultdict(list)
        for _, row in window_df.iterrows():
            current_row = "_".join(map(str, [row["deviceID"], row["BankId"], row["RowId"]]))
            current_col = "_".join(map(str, [row["deviceID"], row["BankId"], row["ColumnId"]]))
            row_pos_dict[current_row].append(row["ColumnId"])
            col_pos_dict[current_col].append(row["RowId"])

        spatio_features["fault_row_num"] = sum(1 for row in row_pos_dict if len(set(row_pos_dict[row])) > 1)
        spatio_features["fault_column_num"] = sum(1 for col in col_pos_dict if len(set(col_pos_dict[col])) > 1)
        return spatio_features

    def _get_err_parity_features(self, window_df: pd.DataFrame) -> Dict[str, int]:
        """提取 parity 相关特征。

        Args:
            window_df (pd.DataFrame): 时间窗口内的数据。

        Returns:
            Dict[str, int]: Parity 特征字典。
        """
        err_parity_features = {
            "error_bit_count": window_df["bit_count"].sum(),
            "error_dq_count": window_df["dq_count"].sum(),
            "error_burst_count": window_df["burst_count"].sum(),
            "max_dq_interval": window_df["max_dq_interval"].max(),
            "max_burst_interval": window_df["max_burst_interval"].max()
        }
        dq_counts = window_df["dq_count"].value_counts().to_dict()
        burst_counts = window_df["burst_count"].value_counts().to_dict()

        for dq in range(1, DQ_COUNT + 1):
            err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)
        for burst in range(1, BURST_COUNT + 1):
            err_parity_features[f"burst_count={burst}"] = burst_counts.get(burst, 0)
        return err_parity_features

    @staticmethod
    def _calculate_ce_storm_count(log_times: np.ndarray, ce_storm_interval_seconds: int = 60, 
                                 ce_storm_count_threshold: int = 10) -> int:
        """计算 CE storm 数量。

        Args:
            log_times (np.ndarray): 日志时间数组。
            ce_storm_interval_seconds (int): CE storm 间隔阈值（秒）。
            ce_storm_count_threshold (int): CE storm 日志数量阈值。

        Returns:
            int: CE storm 数量。
        """
        if len(log_times) < ce_storm_count_threshold:
            return 0
        log_times = sorted(log_times)
        ce_storm_count = 0
        consecutive_count = 1
        for i in range(1, len(log_times)):
            if log_times[i] - log_times[i-1] <= ce_storm_interval_seconds:
                consecutive_count += 1
                if consecutive_count == ce_storm_count_threshold:
                    ce_storm_count += 1
            else:
                consecutive_count = 1
        return ce_storm_count
    

if __name__ == "__main__":
    raw_data_path = "/home/fei/research/smartmem/dataset/raw_data"
    save_feather_path = "/home/fei/research/smartmem/dataset/features"
    number_workers = 100
    for sn_type in ["A", "B"]:
        ft = FeatureFactory(os.path.join(raw_data_path, f"type_{sn_type}"), 
                            os.path.join(save_feather_path, f"type_{sn_type}"), number_workers)
        ft.process_all_sn()
        logger.info(f"Finish processing type {sn_type} data.")