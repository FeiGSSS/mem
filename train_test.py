import logging
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import argparse
import os
import joblib  # 用于保存和加载模型

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SmartMemory")

class MemoryFailurePredictor:
    def __init__(self, sn_type, model_dir="models"):
        self.sn_type = sn_type
        self.model_dir = model_dir+"_"+sn_type
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "model_ensemble.joblib")

    def _init_model_ensemble(self, X, y):
        """Initialize a list of different machine learning models with optimized parameters"""
        
        # preprocessor = ColumnTransformer(transformers=[('scaler', RobustScaler(), X.columns)],
        #                                  remainder='passthrough').set_output(transform="pandas")
        
        if self.sn_type == "A":
            model1 = RandomForestClassifier(random_state=42,
                                            n_jobs=50,
                                            n_estimators=1063,
                                            max_depth=7,
                                            min_samples_leaf=5,
                                            min_samples_split=3,
                                            max_leaf_nodes=902,
                                            class_weight=None,
                                            criterion='gini')
            
            model2 = XGBClassifier(objective='binary:logistic',
                                   eval_metric='logloss',
                                   n_jobs=50,                 
                                   random_state=42,
                                   max_depth=5,
                                   learning_rate=0.062,
                                   n_estimators=604,
                                   subsample=0.727,
                                   colsample_bytree=0.636,
                                   gamma=2.83,
                                   reg_alpha=2.183,
                                   reg_lambda=6.29e-08)
            
            model3 = LGBMClassifier(objective='binary',
                                    verbose=-1,
                                    n_jobs=50,
                                    random_state=42,
                                    force_row_wise=True,
                                    max_depth = 4,
                                    num_leaves=8,
                                    learning_rate=0.02961318405320819,
                                    n_estimators=1052,
                                    min_child_samples=33,
                                    reg_alpha=0.02000822696886964,
                                    reg_lambda=0.016658086959063234,
                                    feature_fraction=0.7130697388172327,
                                    bagging_fraction=0.7615571226273274,
                                    bagging_freq=5,
                                    class_weight=None)
            
        elif self.sn_type == "B":
            model1 = RandomForestClassifier(random_state=42,
                                            n_jobs=50,
                                            n_estimators=362,
                                            max_depth=3,
                                            min_samples_split=14,
                                            min_samples_leaf=1,
                                            max_leaf_nodes=601,
                                            class_weight='balanced',
                                            criterion='entropy')
            
            model2 = XGBClassifier(objective='binary:logistic',
                                   eval_metric='logloss',
                                   n_jobs=50,                 
                                   random_state=42,
                                   scale_pos_weight = sum(y == 0) / sum(y == 1),
                                   max_depth=3,
                                   learning_rate=0.0011,
                                   n_estimators=1038,
                                   subsample=0.746,
                                   colsample_bytree=0.7,
                                   gamma=0.804,
                                   reg_alpha=5.0706897823449304e-06,
                                   reg_lambda=4.4243674081009865e-06)
            model3 = LGBMClassifier(objective='binary',
                                    verbose=-1,
                                    n_jobs=50,
                                    random_state=42,
                                    force_row_wise=True,
                                    class_weight='balanced',
                                    max_depth=4,
                                    num_leaves=11,
                                    learning_rate=0.0014,
                                    n_estimators=753,
                                    min_child_samples=8,
                                    reg_alpha=8e-5,
                                    reg_lambda=2e-6)
        else:
            raise ValueError(f"Unknown sn_type: {self.sn_type}")
        
        # return [Pipeline([('preprocessor', preprocessor), ('model', model1)]),
        #         Pipeline([('preprocessor', preprocessor), ('model', model2)]),
        #         Pipeline([('preprocessor', preprocessor), ('model', model3)])]
        
        return [model1, model2, model3]
                                    
                                   
        

    def train(self, train_positive, train_negative, threshold):
        """Train the model ensemble"""
        logger.info("=" * 50)
        logger.info("开始训练模型...")

        train_data = pd.concat([train_positive, train_negative])

        # Prepare features and labels
        X = train_data.drop(columns=["label"])
        y = train_data["label"]

        logger.info("数据大小，特征数: {}, 样本数: {}".format(X.shape[1], X.shape[0]))
        logger.info(f"正样本数: {len(train_positive)}, 负样本数: {len(train_negative)}, "
                    f"正负比例: {len(train_positive) / (len(train_negative) + 1e-10):.4f}")

        # Initialize and train the model ensemble
        models = self._init_model_ensemble(X, y)
        for i, model in enumerate(models):
            logger.info(f"训练模型 {i+1}...")
            model.fit(X, y)

        # Save the model ensemble
        joblib.dump(models, self.model_path)
        logger.info(f"模型集成已保存至: {self.model_path}")

        logger.info("模型训练完成")
        logger.info("=" * 50)

        # Evaluate on training set
        logger.info("在训练集上评估模型...")
        probas_list = [model.predict_proba(X)[:, 1] for model in models]
        mean_probas = np.mean(probas_list, axis=0)
        final_pred = (mean_probas > threshold).astype(int)
        
        auc = roc_auc_score(y, mean_probas)
        f1 = f1_score(y, final_pred)
        cm = confusion_matrix(y, final_pred)
        
        logger.info(f"训练集上的AUC: {auc:.4f}, F1: {f1:.4f}")
        logger.info(f"混淆矩阵: \n{cm}")

    def predict(self, test_data, threshold):
        """Predict using majority voting"""
        logger.info("=" * 50)
        logger.info("开始预测...")

        X = test_data.drop(columns=["label"]) if "label" in test_data.columns else test_data
        X = X.fillna(0)  # Fill missing values with 0

        # Load the model ensemble
        logger.info(f"加载模型集成: {self.model_path}")
        models = joblib.load(self.model_path)

        # Get predictions from each model
        probas_list = [model.predict_proba(X)[:, 1] for model in models]
        mean_probas = np.mean(probas_list, axis=0)
        final_pred = (mean_probas > threshold).astype(int)

        sn_names = test_data.index.get_level_values(0)
        log_times = test_data.index.get_level_values(1)

        logger.info(f"正样本数: {sum(final_pred)}, 负样本数: {len(final_pred) - sum(final_pred)}, "
                    f"正负比例: {sum(final_pred) / (len(final_pred) - sum(final_pred) + 1e-10):.4f}")

        return sn_names, log_times, final_pred  # Return probas to maintain interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="内存故障预测系统")
    parser.add_argument("--data_path", type=str, default="/home/fei/research/smartmem/dataset/train_test_data/train_data")
    parser.add_argument("--test_data_path", type=str, default="/home/fei/research/smartmem/dataset/train_test_data/test_data")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--train", action="store_true", help="运行训练阶段")
    parser.add_argument("--test", action="store_true", help="运行测试阶段")
    args = parser.parse_args()

    logger.info("内存故障预测系统启动")

    if args.train:
        logger.info("进入训练阶段...")
        logger.info("加载训练数据...")
        for sn_type in ["A", "B"]:
            logger.info(f"开始训练 {sn_type} 类型的数据...")
            model = MemoryFailurePredictor(sn_type)
            train_dir = os.path.join(args.data_path, f"type_{sn_type}")
            train_positive_path = os.path.join(train_dir, "positive_train.feather")
            train_negative_path = os.path.join(train_dir, "negative_train.feather")
            
            positive = pd.read_feather(train_positive_path)
            negative = pd.read_feather(train_negative_path)
            
        
            model.train(positive, negative, args.threshold)

    if args.test:
        logger.info("进入测试阶段...")
        submission = []
        for sn_type in ["A", "B"]:
            logger.info(f"开始测试 {sn_type} 类型的数据...")
            model = MemoryFailurePredictor(sn_type)
            test_dir = os.path.join(args.test_data_path, f"type_{sn_type}_0")
            test_data = pd.read_feather(os.path.join(test_dir, "test_data.feather"))            
            sn_names, log_times, pred = model.predict(test_data, threshold=args.threshold)
        
            for sn, lt, p in zip(sn_names, log_times, pred):
                if p == 1:
                    submission.append([sn, lt, sn_type])

        submission_df = pd.DataFrame(submission,
                                     columns=["sn_name", "prediction_timestamp", "serial_number_type"])
        
        output_path = "/home/fei/research/smartmem/submission/submission.csv"
        submission_df.to_csv(output_path, index=False)
        logger.info(f"结果已保存至 {output_path}，共 {len(submission)} 条记录")
        logger.info("测试阶段完成")

    logger.info("流程结束")