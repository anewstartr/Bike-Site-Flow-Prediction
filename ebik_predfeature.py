#%%
from datetime import datetime, timedelta
import gc
import os
import json
import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, count, row_number, collect_list
from pyspark.sql.functions import avg, from_json, to_json, concat_ws
from pyspark.sql.functions import col, concat, lit, explode, array, struct
from pyspark.sql.types import StringType, ArrayType,IntegerType, FloatType,StructType,StructField,BooleanType

from aibrain_common.utils.date_convert_utils import DateConvertUtils
from aibrain_common.component import tools
import uuid
from aibrain_common.data.dataset_builder import DatasetBuilder
from aibrain_common.utils import env_utils

from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


# %matplotlib notebook


import logging

logger = logging.getLogger(__name__)

import gc


spark = SparkSession.builder.\
        config('spark.executor.memory', '12g').\
        config('spark.executor.cores', '6').\
        config('spark.driver.memory','10g').\
        config('spark.executor.instances', '10').\
        config('spark.driver.maxResultSize', '50000m').\
        appName('ebikepredfea').\
        enableHiveSupport().getOrCreate()


time_str_formats = {
    "hour": "%Y%m%d%H",
    "day": "%Y%m%d",
}

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
#%%
@F.udf(StringType())
def generate_datetime_hour(date_str, hour_int):
    return f"{date_str}{hour_int:02d}"

@F.udf(StringType())
def generate_datetime_hour_minus2(date_str, hour_int):
    date_m2 = (datetime.strptime(date_str, '%Y%m%d') + timedelta(days=-2)).date().strftime('%Y%m%d')
    return f"{date_m2}{hour_int:02d}"

def datetime2str(date: datetime, rtype="hour"):
    if rtype not in time_str_formats:
        raise ValueError("rtype Error!")
    else:
        return date.strftime(time_str_formats[rtype])


def str2datetime(s, itype="hour"):
    return datetime.strptime(s, time_str_formats[itype])
def add_delta(time_str: str, delta: dict, itype="day", rtype="day"):
    target_time = str2datetime(time_str, itype) + timedelta(**delta)
    target_time = datetime2str(target_time, rtype)
    return target_time
#%%
date_converter = DateConvertUtils()
date_converter.set_biz_date("20250917")
pt = date_converter.parse_data_date("${yyyymmdd}")
today = pt
logger.warning(f"today is : {today}")
print("today is :%s" % today)
db = 'turing_dev'
db0 = 'turing'
# 运行当天t
week = 2
tomorrow = add_delta(today, {'days': 1}, "day", "day")  # 预测的日期（t+1）
yesterday = add_delta(today, {'days': -1}, "day", "day")  # 前一天，用于找大点


day1 = - (week * 7 - 1)  # 差值为13，获取14天的数据特征
end_date = add_delta(today, {'days': - 2}, "day", "day")  # 能够获取到的最新的真值的日期为t-2(流入流出特征需要两天真值来计算)
start_date = add_delta(end_date, {'days': day1}, "day", "day")  # 预测需要用到前14天特征，
twoweek_ago_date = add_delta(end_date, {'days': - 13}, "day", "day")
# start_date = '20250708'
start_date_minus11 = (datetime.strptime(start_date, '%Y%m%d') + timedelta(days=-11)).date().strftime(
    '%Y%m%d')  # 还需要+11天得到lag14d特征
start_date_add2 = (datetime.strptime(start_date, '%Y%m%d') + timedelta(days=2)).date().strftime('%Y%m%d')

end_date_add2 = (datetime.strptime(end_date, '%Y%m%d') + timedelta(days=2)).date().strftime('%Y%m%d')
end_date_2 = (datetime.strptime(end_date, '%Y%m%d') + timedelta(days=-2)).date().strftime('%Y%m%d')

#%%
df_flow = spark.sql(f'''
with 
    bike_start_order as (
        select 
            bike_start_park_guid as parking_guid,
            count(order_id) as daily_order_cnt
        from dwd.dwd_trd_ord_ebik_order_ent_di
        where pt between '{twoweek_ago_date}' and '{end_date}'
        group by bike_start_park_guid,pt
    ),
    
    site_max_orders as (
        select 
            parking_guid,
            max(daily_order_cnt) as max_daily_orders
        from bike_start_order
        group by parking_guid
        having max_daily_orders >= 5
    ),
    
    filtered_streets as (
        select parking_guid from site_max_orders
    ),

    net_data as (
        select 
            site_guid, city_guid, label_10, label_16, label_21, hour, pt
        from turing.ebike_site_period_net_out_label_di
        where pt between '{start_date_minus11}' and '{end_date}'
    )

select net_data.* from net_data
where net_data.site_guid in (select parking_guid from filtered_streets)

''')



#%%

# 定义窗口规范
window_spec = Window.partitionBy("site_guid","hour").orderBy("date")
# 定义滞后天数列表
lags = [1, 2, 3, 4, 11]

# 2. lag24数据 (t-4日)
long_df = df_flow.withColumn("dt", generate_datetime_hour("pt", "hour")  # 如pt=20250701 + hour=12 → dt=2025070112
    ).withColumn('date', F.to_date(F.col('dt').cast('string'), 'yyyyMMddHH')
    ).withColumn('day_of_week', F.dayofweek('date'))

# 为每个特征生成滞后列
for k in lags:
    # 计算滞后的小时数
    # lag_hours = 24 * k
    long_df = long_df \
        .withColumn(f"lag{k}d_10", F.lag("label_10", k).over(window_spec)) \
        .withColumn(f"lag{k}d_16", F.lag("label_16", k).over(window_spec)) \
        .withColumn(f"lag{k}d_21", F.lag("label_21", k).over(window_spec)) \

long_df = long_df.filter(F.col("pt").between(start_date, end_date)).fillna(0)
long_df.cache()
#%%

ebik_park_hf = spark.read.format("iceberg").load("dwb.dwb_veh_ebik_park_hf") \
    .filter(F.col("pt").between(start_date,end_date)&
        (F.col("min") == "00")
    ).select("parking_guid","put_veh_cnt","hr","pt"
    ).withColumn("dt2", F.concat(F.col("pt"), F.col("hr"))
    )

long_df = long_df.join(ebik_park_hf, (long_df.site_guid==ebik_park_hf.parking_guid) & (long_df.dt==ebik_park_hf.dt2), 'left'
                ).select(long_df.site_guid,long_df.city_guid,"label_10","label_16","label_21","put_veh_cnt",
        "lag1d_10","lag1d_16","lag1d_21","lag2d_10","lag2d_16","lag2d_21",
        "lag3d_10","lag3d_16","lag3d_21","lag4d_10","lag4d_16","lag4d_21",
        "lag11d_10","lag11d_16","lag11d_21","hour",long_df.pt,"day_of_week","dt")



# 2. 获取数据
wtw = spark.table('turing_dev.turing_net_pred_wea_temp_wkd_feature') \
    .select('city_guid','forecast_date','cycle_weather_level','temperature_avg_val','workday_level') \
    .filter(F.col('forecast_date').between(start_date_add2, end_date_add2)) \
    .groupBy('city_guid', 'forecast_date') \
    .agg(
        F.first('cycle_weather_level').alias('cycle_weather_level'),
        F.first('temperature_avg_val').alias('temperature_avg_val'),
        F.first('workday_level').alias('workday_level')

    )

long_df = long_df.join(wtw,
                       (long_df.pt==wtw.forecast_date) &
                       (long_df.city_guid==wtw.city_guid), 'left'
        ).select(long_df.site_guid,long_df.city_guid,"label_10","label_16","label_21","put_veh_cnt",
        "lag1d_10","lag1d_16","lag1d_21","lag2d_10","lag2d_16","lag2d_21",
        "lag3d_10","lag3d_16","lag3d_21","lag4d_10","lag4d_16","lag4d_21",
        "lag11d_10","lag11d_16","lag11d_21","hour","pt","day_of_week","dt",
        wtw.cycle_weather_level,wtw.temperature_avg_val,wtw.workday_level
        )

print("小时粒度数据完成...")
#%%
fill_dict = {
    "label_10": 0,
    "label_16": 0,
    "label_21": 0,
    "put_veh_cnt": 0,
    "lag1d_10": 0,
    "lag1d_16": 0,
    "lag1d_21": 0,
    "lag2d_10": 0,
    "lag2d_16": 0,
    "lag2d_21": 0,
    "lag3d_10": 0,
    "lag3d_16": 0,
    "lag3d_21": 0,
    "lag4d_10": 0,
    "lag4d_16": 0,
    "lag4d_21": 0,
    "lag11d_10": 0,
    "lag11d_16": 0,
    "lag11d_21": 0,
    'cycle_weather_level': 2,  # 假设填充2
    'workday_level': 1,
    'temperature_avg_val': 25.00,  # 假设填充25
}

long_df = long_df.fillna(fill_dict)
long_df = long_df.select(
    'dt',
    'site_guid',
    "label_10",
    "label_16",
    "label_21",
    "put_veh_cnt",
    "lag1d_10",
    "lag1d_16",
    "lag1d_21",
    "lag2d_10",
    "lag2d_16",
    "lag2d_21",
    "lag3d_10",
    "lag3d_16",
    "lag3d_21",
    "lag4d_10",
    "lag4d_16",
    "lag4d_21",
    "lag11d_10",
    "lag11d_16",
    "lag11d_21",
    'cycle_weather_level',
    'temperature_avg_val',
    'workday_level',
    'day_of_week',
    'hour'
).sort('site_guid', 'dt')

long_df.cache()
#%%
def create_optimized_intermediate_table_v2(long_df, pred_date, seq_len, output_table_name, id_col="site_guid", time_col="dt", feature_cols=None):
    """
    修复版本：避免复杂UDF，使用更直接的方法
    """
    if feature_cols is None:
        feature_cols = [col for col in long_df.columns if col not in [id_col, time_col]]


    print(f"特征列: {feature_cols}")

   # 数据预处理
    window_spec = Window.partitionBy(id_col).orderBy(F.desc(time_col))
    windowed_df = long_df.withColumn("row_num", row_number().over(window_spec)) \
                         .filter(F.col("row_num") <= seq_len) \
                         .orderBy(id_col, time_col)

    # 创建特征向量
    windowed_df = windowed_df.withColumn(
        "feature_vector",
        array(*[F.col(feat).cast("double") for feat in feature_cols])
    )

    # 分组并收集
    grouped_df = windowed_df.groupBy(id_col).agg(
        collect_list("feature_vector").alias("features_array"),
        F.count("*").alias("seq_count")
    ).filter(F.col("seq_count") == seq_len)

    # 方案A: 存储为JSON字符串（推荐）
    final_df = grouped_df.withColumn("features_json", to_json(F.col("features_array"))
                                    ).select(id_col, "features_json")
    # final_df.show(2, truncate=False)

# 保存
    final_df.createOrReplaceTempView("final_feature_df")
    spark.sql(f"""
        INSERT OVERWRITE TABLE {output_table_name} 
        PARTITION(pt='{pred_date}') 
        SELECT {id_col}, features_json
        FROM final_feature_df
    """)

    print(f"优化的中间表已保存: {output_table_name}")
    print(f"序列长度: {seq_len}, 特征数量: {len(feature_cols)}")


    return final_df
outputtable = 'turing_dev.turing_ebike_site_fixtime_predict_features_df'
final_df = create_optimized_intermediate_table_v2(long_df, tomorrow, 14*24, output_table_name = outputtable, id_col="site_guid", time_col="dt", feature_cols=None)
#%%

#%%
# spark.sql("""
# CREATE TABLE turing_dev.turing_ebike_site_fixtime_predict_features_df(
#   site_guid string COMMENT '街区id',
#   features string COMMENT '深度模型需要的特征')
# COMMENT '单车街区定终点模型每日预测特征数据表'
# PARTITIONED BY (
#   pt string COMMENT '预测的时间')
# """)
#%%

