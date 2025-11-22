#%%
from datetime import datetime, timedelta
import gc
import os
import json
from getopt import long_has_args

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col, ceil, to_json
from pyspark.sql.functions import col, concat, lit, explode, array, struct, posexplode, format_string
from pyspark.sql.types import StringType, ArrayType,IntegerType, FloatType,StructType,StructField,BooleanType,DoubleType
from aibrain_common.utils.date_convert_utils import DateConvertUtils
from aibrain_common.component import tools
import uuid
from aibrain_common.data.dataset_builder import DatasetBuilder
from aibrain_common.utils import env_utils

from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader, RandomSampler

# %matplotlib notebook


import logging

logger = logging.getLogger(__name__)

import gc


spark = SparkSession.builder.\
        config('spark.executor.memory', '8g').\
        config('spark.executor.cores', '6').\
        config('spark.driver.memory','6g').\
        config('spark.executor.instances', '10').\
        config('spark.driver.maxResultSize', '50000m').\
        appName('ebike_net_pred').\
        enableHiveSupport().getOrCreate()


time_str_formats = {
    "hour": "%Y%m%d%H",
    "day": "%Y%m%d",
}

date_converter = DateConvertUtils()
date_converter.set_biz_date("20250919")
today = date_converter.parse_data_date("${yyyymmdd}")
logger.warning(f"today is : {today}")
# print("today is :%s" % today)
db = 'turing_dev'
db0 = 'turing'
tomorrow = (datetime.strptime(today, '%Y%m%d') + timedelta(days=1)).date().strftime('%Y%m%d') # 预测的日期（t+1）
yesterday = (datetime.strptime(today, '%Y%m%d') + timedelta(days=-1)).date().strftime('%Y%m%d') 
pt = tomorrow

    

def printbar():
    t = datetime.datetime.now()
    print('==========='*8 + str(t))


    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # [N, 38, 14*24]->[N, 300, 14*24]
        # n_inputs=38, n_outputs=1
        # weight_norm(
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.GroupNorm(1, n_outputs)  #加一层试试

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # [N, 300, 14*24]->[N, 300, 14*24]
        # weight_norm(
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.GroupNorm(1, n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        

        self.net = nn.Sequential(self.conv1, self.chomp1, self.norm1, self.relu1, self.dropout1,
                                  self.conv2, self.chomp2, self.norm2, self.relu2, self.dropout2)

        # [N, 38, 14*24]->[N, 300, 14*24]
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None # 1x1 conv
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        x: n*emb_size*seq_len
        out: n*layer_outchannel* seq_len"""
        # [N, 38, 14*24]->[N, 1, 14*24]
        out = self.net(x)
        # [N, 38, 14*24]->[N, 1, 14*24]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # [N, 1, 14*24]


class TemporalConvNet(nn.Module):
    # num_inputs=38, out_channels=[300, 200, 100, 50, 1]
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # dilation_sizes = [1,4,16,24]
        for i in range(num_levels):
            """dilated conv"""
            dilation_size = 2 ** i   #认为此处不合理，待改                                                    
            # dilation_size = dilation_sizes[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # [N, 300, 14*24] + [N, 200, 14*24] + [N, 100, 14*24] + [N, 50, 14*24] + [N, 1, 14*24]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
 


    
#%%
class PeakHuberLoss(nn.Module):
    def __init__(self):
        super(PeakHuberLoss, self).__init__()
    def forward(self, y_pred, y_true, delta = 5):
        error = y_true - y_pred
        peak_mask = (y_true >= 5)
        # 峰值用Huber Loss，非峰值用MAE
        peak_loss = torch.where(torch.abs(error[peak_mask]) <= delta, 
                               0.5 * error[peak_mask]**2, 
                               delta * (torch.abs(error[peak_mask]) - 0.5 * delta)).mean() if torch.any(peak_mask) else 0.0
        non_peak_loss = torch.abs(error[~peak_mask]).mean() if torch.any(~peak_mask) else 0.0
        return peak_loss * 2 + non_peak_loss  # 峰值损失权重加倍
    

class MultiTaskPHLoss(nn.Module):
    def __init__(self, loss_weights=None):
        super(MultiTaskPHLoss, self).__init__()
        self.peakhuberloss = PeakHuberLoss()
        self.loss_weights = loss_weights
    
    def forward(self, predictions, targets, delta = 5):
        total_loss = 0
        losses = {}
        
        for scale, pred in predictions.items():
            target = targets[scale]
            scale_loss = self.peakhuberloss(pred, target, delta = delta)
            
            # 应用权重（如果有）
            weight = self.loss_weights[scale] if self.loss_weights else 1.0
            weighted_loss = weight * scale_loss
            
            losses[scale] = scale_loss.item()
            total_loss += weighted_loss
        
        return total_loss, losses
    


#%%
def batch_predict_from_spark_table(table_name, pred_date, model, output_sizes, batch_size=2500, 
                                 seq_len=336, feature_count=24, device='cuda:1'):
    """
    从Spark表批量加载数据并进行模型预测
    """
    
    # 1. 为数据添加行号以便分批处理
    df = spark.sql(f"""
        SELECT *, 
               ROW_NUMBER() OVER (ORDER BY site_guid) as row_id
        FROM {table_name} 
        WHERE pt = {pred_date}
    """)
    
    # 获取总行数
    total_count = df.count()
    logger.warning(f"总共需要处理 {total_count} 个点位")
    
    # 2. 计算批次数
    num_batches = (total_count + batch_size - 1) // batch_size
    
    results = []
    
    for batch_idx in range(num_batches):
        start_row = batch_idx * batch_size + 1  # row_number从1开始
        end_row = min((batch_idx + 1) * batch_size, total_count)
        
        logger.info(f"处理批次 {batch_idx + 1}/{num_batches}, 行范围: {start_row}-{end_row}")
        
        # 3. 获取当前批次数据
        batch_df = df.filter(F.col('row_id').between(start_row, end_row)
                            ).select('site_guid','features')
#         batch_df = spark.sql(f"""
#             SELECT site_guid, features
#             FROM (
#                 SELECT *, ROW_NUMBER() OVER (ORDER BY site_guid) as row_id
#                 FROM {table_name} 
#                 WHERE prediction_pt = '{pred_date}'
#             ) t
#             WHERE row_id >= {start_row} and row_id <= {end_row}
#         """)
        
        # 4. 收集当前批次数据到Driver
        batch_data = batch_df.collect()
        # batch_df.show(10)
        # print(batch_data[0:10,:])
        if not batch_data:
            continue
            
        # 5. 转换为numpy数组
        batch_features, site_guids = prepare_batch_features(
            batch_data, seq_len, feature_count
        )
        
        # 6. 模型预测
        batch_predictions = predict_batch(model, batch_features, output_sizes, device)
        
        # 7. 保存结果 - 多目标版本
        for i, site_guid in enumerate(site_guids):
            # 获取该点位的所有目标预测
            pred_dict = {}
            for target_idx in output_sizes.keys():
                # 获取该目标的预测数组 [24, 1]
                target_pred = batch_predictions[target_idx][i]
                # 展平为24个值的列表
                pred_list = target_pred.flatten().tolist()[:24]  # 取前24个元素
                
                # 如果长度不足24，用0填充
                if len(pred_list) < 24:
                    pred_list.extend([0.0] * (24 - len(pred_list)))
                
                pred_dict[f'target_{target_idx}'] = pred_list
            
            # 添加站点信息和预测结果
            result_item = {'id': site_guid}
            result_item.update(pred_dict)
            results.append(result_item)

        # 8. 清理内存
        del batch_data, batch_features, batch_predictions
        gc.collect()
        if batch_idx % 10 == 0:  # 每10个批次打印一次进度
            logger.info(f"已完成 {batch_idx + 1}/{num_batches} 批次")

    # 最后将结果转换为DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

def prepare_batch_features(batch_data, seq_len, feature_count):
    """
    解码JSON格式的特征数据
    """
    import json
    
    batch_features = []
    site_guids = []
    
    for row in batch_data:
        site_guid = row['site_guid']
        features_json = row['features']
        
        # print(f"处理 {site_guid}")
        # print(f"JSON数据类型: {type(features_json)}")
        
        try:
            # 解析JSON
            if isinstance(features_json, str):
                features_list = json.loads(features_json)
                # print("JSON解析成功")
            else:
                features_list = features_json
                logger.info("直接使用原始数据")
            
            # print(f"解析后数据类型: {type(features_list)}")
            # print(f"解析后数据长度: {len(features_list) if features_list else 0}")
            
            # if features_list and len(features_list) > 0:
                # print(f"第一个时间步: {features_list[0]}")
            
            # 转换为numpy数组
            if features_list and len(features_list) == seq_len:
                feature_matrix = []
                
                for i, time_step in enumerate(features_list):
                    if isinstance(time_step, (list, tuple)) and len(time_step) == feature_count:
                        row_data = [float(x) for x in time_step]
                        feature_matrix.append(row_data)
                    else:
                        logger.warning(f"时间步 {i} 格式错误: {type(time_step)}, 长度: {len(time_step) if hasattr(time_step, '__len__') else 'N/A'}")
                        break
                else:
                    # 所有时间步都正确处理
                    feature_array = np.array(feature_matrix, dtype=np.float32)
                    # print(f"转换后shape: {feature_array.shape}")
                    
                    if feature_array.shape == (seq_len, feature_count):
                        batch_features.append(feature_array)
                        site_guids.append(site_guid)
                        # print("✓ 成功添加")
                    else:
                        logger.warning(f"✗ 维度不匹配: {feature_array.shape} vs ({seq_len}, {feature_count})")
            else:
                logger.warning(f"✗ 序列长度不匹配: {len(features_list) if features_list else 0} vs {seq_len}")
                
        # except json.JSONDecodeError as e:
            # print(f"✗ JSON解析错误 {site_guid}: {e}")
            # print(f"原始数据: {features_json[:200]}...")
        except Exception as e:
            logger.warning(f"✗ 转换错误 {site_guid}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 转换为3D数组 (batch_size, seq_len, feature_count)
    if batch_features:
        batch_features = np.stack(batch_features, axis=0)
        logger.warning(f"最终批次形状: {batch_features.shape}")
        # print(f"数据范围: min={batch_features.min():.4f}, max={batch_features.max():.4f}")
    else:
        batch_features = np.empty((0, seq_len, feature_count))
        # print("警告: 没有有效的特征数据")
    
    return batch_features, site_guids

def predict_batch(model, batch_features, output_sizes, device='cuda'):
    """
    使用多目标模型进行批量预测
    """
    
    # 转换为torch tensor
    batch_tensor = torch.FloatTensor(batch_features).to(device)
    
    with torch.no_grad():
        predictions = model(batch_tensor)
        # 转回CPU并转为numpy
        pred_results = {}
        for target_idx in output_sizes.keys():
            # 确保形状正确 [batch_size, output_size, 1]
            pred = predictions[target_idx]
#             if pred.dim() == 2:  # 如果是 [batch_size, output_size]
#                 pred = pred.unsqueeze(-1)  # 添加最后一维
            pred_results[target_idx] = pred.cpu().numpy()
    
    return pred_results
#%%
output_sizes = {0: 24, 1: 24, 2: 24}  # 3/6/12小时预测，输出对应长度
num_channels = [64, 128, 32, 3]  # TCN隐藏层维度
device = 'cpu'
# 加载模型，预测
tcn_save_model = MultiTaskTCN(input_size=24, input_len=14*24, output_sizes=output_sizes, num_channels=num_channels, kernel_size=3, dropout=0.25, emb_dropout=0.2, tied_weights=False).to(device)
# batch*length*size 输入， batch = 32个点位，len = 7天*24小时，size = 8个特征 
tcn_save_model.load_state_dict(torch.load('net_ebik_multitask_2.pth'))
tcn_save_model.eval()
logger.warning(f"model loaded")
#%%
# 执行批量预测
table_name = "turing_dev.turing_ebike_site_fixtime_predict_features_df"
# pred_date = "20250826"
# today = DateConvertUtils().parse_data_date('${yyyymmdd}')
# logger.warning(f"today is : {today}")
# pred_date =  (datetime.strptime(today, '%Y%m%d') + timedelta(days=1)).date().strftime('%Y%m%d') # 预测的日期（t+1）
logger.warning(f"predict date is : {tomorrow}")

results = batch_predict_from_spark_table(
    table_name=table_name,
    pred_date=tomorrow,
    model=tcn_save_model,
    output_sizes = output_sizes,
    batch_size=20000,  # 根据内存情况调整
    seq_len=14*24,
    feature_count=24,
    device=device
)

logger.warning(f"预测完成，共处理 {len(results)} 个点位")
#%%
##%%
targets = ["pred_10","pred_16","pred_21"]
results.rename(columns={'target_0': 'pred_10','target_1': 'pred_16','target_2': 'pred_21'},inplace=True)
for target in targets:
    results[target] = results[target].apply(lambda x: [max(0, int(np.ceil(xx))) for xx in x])
spark_output = spark.createDataFrame(results)
spark_output.createOrReplaceTempView("my_data")

# 分别炸开每个目标列
exploded_dfs = {}

for target_col in targets:
    # 为每个目标列创建炸开的DataFrame
    exploded_df = spark_output.select(
        F.col("id").alias("parking_guid"),
        F.posexplode(target_col).alias("hour_idx", target_col)
    ).withColumn(
        "hour", 
        F.format_string("%02d", F.col("hour_idx"))
    ).drop("hour_idx")
    
    exploded_dfs[target_col] = exploded_df

# 通过parking_guid和hour连接所有炸开的DataFrame
# 从第一个目标开始
spark_output2 = exploded_dfs[targets[0]]

# 依次连接其他目标
for target_col in targets[1:]:
    spark_output2 = spark_output2.join(
        exploded_dfs[target_col],
        ["parking_guid", "hour"],
        "inner"
    )


# 添加city列
city_table = spark.table('dim.dim_spt_fence_info'
                         ).filter((F.col('pt') == yesterday) & (F.col('area_status') == 5)
                         ).select('fence_id', 'city_guid')
spark_output2 = spark_output2.join(
    city_table,
    spark_output2.parking_guid == city_table.fence_id, 
    'left'
).select('parking_guid', 'city_guid', ceil('pred_10'), ceil('pred_16'), ceil('pred_21'), 'hour')

spark_output2.createOrReplaceTempView("new_format_res")

# 插入到目标表
table_name2 = 'turing.ebike_site_period_net_out_predict_di'
spark.sql(f"insert overwrite table {table_name2} partition(pt='{tomorrow}') select * from new_format_res")
logger.warning(f"{tomorrow}预测结果已保存: {table_name2}")
#%%


# 定义target列名和对应的JSON特征名
target_columns = {
    'pred_10': 'period_10_netout',
    'pred_16': 'period_16_netout', 
    'pred_21': 'period_21_netout'
}

# 为每个target列创建JSON结构
for target_col, feature_name in target_columns.items():
    # 确保列存在
    if target_col in spark_output.columns:
        # 将列表转换为JSON字符串，并指定特征名
        spark_output = spark_output.withColumn(
            f"{feature_name}_json",
            to_json(struct(col(target_col).alias(feature_name)))
        )
    else:
        print(f"Warning: Column {target_col} not found in DataFrame")

        
spark_output = spark_output.select('id','period_10_netout_json','period_16_netout_json','period_21_netout_json')
spark_output.createOrReplaceTempView("my_data")
table_name3 = 'turing.ebike_site_net_out_pred_json_fea_di'
spark.sql(f"insert overwrite table {table_name3} partition(pt='{tomorrow}') select * from my_data")
logger.warning(f"{tomorrow}json特征已保存: {table_name3}")
