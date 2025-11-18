from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

def create_spark_session():
    spark = SparkSession.builder \
        .appName("导出FineBI数据") \
        .master("local[*]") \
        .enableHiveSupport() \
        .getOrCreate()
    
    spark.sql("USE dbestate")
    return spark

def create_dashboard_summaries(spark):
    """创建仪表板汇总数据 - 修复中文字段名问题"""
    
    # 1. 城市维度汇总（使用英文字段名，后面在导出时添加中文标题）
    city_summary = spark.sql("""
        SELECT 
            city,
            COUNT(*) as house_count,
            ROUND(AVG(price_num), 0) as avg_price,
            ROUND(AVG(area_avg), 1) as avg_area,
            SUM(CASE WHEN is_subway = 1 THEN 1 ELSE 0 END) as subway_count,
            ROUND(SUM(CASE WHEN is_subway = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as subway_ratio,
            SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as accuracy_rate
        FROM ml_predictions
        WHERE price_num > 1000 AND price_num < 200000
          AND area_avg > 20 AND area_avg < 400
        GROUP BY city
        ORDER BY avg_price DESC
    """)
    
    print("\n城市汇总数据:")
    city_summary.show()
    
    # 转换为带中文标题的DataFrame（用于CSV导出）
    city_summary_cn = city_summary.select(
        col("city").alias("城市"),
        col("house_count").alias("房源数量"),
        col("avg_price").alias("平均价格_元每平米"),
        col("avg_area").alias("平均面积_平米"),
        col("subway_count").alias("地铁房数量"),
        col("subway_ratio").alias("地铁房比例_百分比"),
        col("correct_predictions").alias("预测正确数量"),
        col("accuracy_rate").alias("预测准确率_百分比")
    )
    
    # 导出城市汇总
    city_summary_cn.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .option("encoding", "UTF-8") \
        .csv("/tmp/finebi_final/city_summary")
    
    print("城市汇总数据已导出")
    
    # 2. 混淆矩阵数据
    confusion_matrix = spark.sql("""
        SELECT 
            price_level as actual_level,
            prediction as predicted_level,
            COUNT(*) as count_num
        FROM ml_predictions
        WHERE price_num > 1000 AND price_num < 200000
          AND area_avg > 20 AND area_avg < 400
        GROUP BY price_level, prediction
        ORDER BY price_level, prediction
    """)
    
    # 添加中文标签
    confusion_matrix_cn = confusion_matrix.select(
        when(col("actual_level") == 0, "低价房")
        .when(col("actual_level") == 1, "中价房")
        .otherwise("高价房").alias("实际等级"),
        when(col("predicted_level") == 0, "低价房")
        .when(col("predicted_level") == 1, "中价房")
        .otherwise("高价房").alias("预测等级"),
        col("count_num").alias("数量")
    )
    
    print("\n混淆矩阵:")
    confusion_matrix_cn.show()
    
    # 导出混淆矩阵
    confusion_matrix_cn.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .option("encoding", "UTF-8") \
        .csv("/tmp/finebi_final/confusion_matrix")
    
    print("混淆矩阵数据已导出")

def create_feature_analysis(spark):
    """创建特征分析数据"""
    print("\n=== 创建特征分析数据 ===")
    
    # 特征对价格的影响分析
    feature_analysis = spark.sql("""
        SELECT 
            '面积影响' as feature_type,
            CASE 
                WHEN area_avg < 80 THEN '小户型_80以下'
                WHEN area_avg < 120 THEN '中户型_80到120'
                ELSE '大户型_120以上'
            END as feature_value,
            COUNT(*) as count_num,
            ROUND(AVG(price_num), 0) as avg_price,
            SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as accuracy_rate
        FROM ml_predictions
        WHERE price_num > 1000 AND price_num < 200000
          AND area_avg > 20 AND area_avg < 400
        GROUP BY 
            CASE 
                WHEN area_avg < 80 THEN '小户型_80以下'
                WHEN area_avg < 120 THEN '中户型_80到120'
                ELSE '大户型_120以上'
            END
        
        UNION ALL
        
        SELECT 
            '地铁影响' as feature_type,
            CASE WHEN is_subway = 1 THEN '地铁房' ELSE '非地铁房' END as feature_value,
            COUNT(*) as count_num,
            ROUND(AVG(price_num), 0) as avg_price,
            SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as accuracy_rate
        FROM ml_predictions
        WHERE price_num > 1000 AND price_num < 200000
          AND area_avg > 20 AND area_avg < 400
        GROUP BY is_subway
        
        UNION ALL
        
        SELECT 
            '停车影响' as feature_type,
            CASE WHEN has_parking = 1 THEN '有停车位' ELSE '无停车位' END as feature_value,
            COUNT(*) as count_num,
            ROUND(AVG(price_num), 0) as avg_price,
            SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as accuracy_rate
        FROM ml_predictions
        WHERE price_num > 1000 AND price_num < 200000
          AND area_avg > 20 AND area_avg < 400
        GROUP BY has_parking
        
        ORDER BY feature_type, avg_price DESC
    """)
    
    # 转换字段名
    feature_analysis_cn = feature_analysis.select(
        col("feature_type").alias("特征类型"),
        col("feature_value").alias("特征值"),
        col("count_num").alias("数量"),
        col("avg_price").alias("平均价格"),
        col("correct_predictions").alias("预测正确数"),
        col("accuracy_rate").alias("预测准确率")
    )
    
    print("特征影响分析:")
    feature_analysis_cn.show()
    
    # 导出特征分析
    feature_analysis_cn.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .option("encoding", "UTF-8") \
        .csv("/tmp/finebi_final/feature_analysis")
    
    print("特征分析数据已导出")

def create_price_distribution(spark):
    """创建价格分布分析"""
    print("\n=== 创建价格分布分析 ===")
    
    price_distribution = spark.sql("""
        SELECT 
            city,
            CASE 
                WHEN price_num < 20000 THEN '1万以下'
                WHEN price_num < 30000 THEN '1万到3万'
                WHEN price_num < 50000 THEN '3万到5万'
                WHEN price_num < 80000 THEN '5万到8万'
                ELSE '8万以上'
            END as price_range,
            COUNT(*) as count_num,
            ROUND(AVG(area_avg), 1) as avg_area
        FROM ml_predictions
        WHERE price_num > 1000 AND price_num < 200000
          AND area_avg > 20 AND area_avg < 400
        GROUP BY city, 
            CASE 
                WHEN price_num < 20000 THEN '1万以下'
                WHEN price_num < 30000 THEN '1万到3万'
                WHEN price_num < 50000 THEN '3万到5万'
                WHEN price_num < 80000 THEN '5万到8万'
                ELSE '8万以上'
            END
        ORDER BY city, count_num DESC
    """)
    
    # 转换字段名
    price_distribution_cn = price_distribution.select(
        col("city").alias("城市"),
        col("price_range").alias("价格区间"),
        col("count_num").alias("房源数量"),
        col("avg_area").alias("平均面积")
    )
    
    print("价格分布分析:")
    price_distribution_cn.show()
    
    # 导出价格分布
    price_distribution_cn.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .option("encoding", "UTF-8") \
        .csv("/tmp/finebi_final/price_distribution")
    
    print("价格分布数据已导出")

def main():
    import os
    os.makedirs("/tmp/finebi_final", exist_ok=True)
    
    spark = create_spark_session()
    
    try:
        print("开始创建FineBI分析数据包")
        
        # 1. 创建汇总数据
        create_dashboard_summaries(spark)
        
        # 2. 创建特征分析数据
        create_feature_analysis(spark)
        
        # 3. 创建价格分布数据
        create_price_distribution(spark)
        
        print("\n" + "="*60)
        print("FineBI完整分析数据包创建完成!")
        print("导出位置: /tmp/finebi_final/")
        print("包含数据表:")
        print("   • analysis_data/ - 详细分析数据")
        print("   • city_summary/ - 城市汇总统计")
        print("   • confusion_matrix/ - 预测准确性矩阵")
        print("   • feature_analysis/ - 特征影响分析")
        print("   • price_distribution/ - 价格分布分析")
        print("\n这些数据可以直接导入FineBI进行可视化分析!")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
