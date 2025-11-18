from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when, lit
from pyspark.sql.types import DoubleType

def create_spark_session():
    spark = SparkSession.builder \
        .appName("58数据分析") \
        .master("local[*]") \
        .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
        .enableHiveSupport() \
        .getOrCreate()
    
    # 切换到正确的数据库
    spark.sql("USE dbestate")
    return spark

def analyze_data(spark):
    """分析数据"""
    
    print("=== 检查dbestate数据库中的数据 ===")
    
    # 显示当前数据库
    current_db = spark.sql("SELECT current_database()").collect()[0][0]
    print(f"当前数据库: {current_db}")
    
    # 显示所有表
    tables = spark.sql("SHOW TABLES")
    print("当前数据库中的表:")
    tables.show()
    
    # 检查每个城市的房屋数据
    cities = ['beijing', 'shanghai', 'guangzhou', 'shenzhen', 'chongqing', 'taiyuan']
    
    print("\n=== 各城市房屋数据统计 ===")
    for city in cities:
        try:
            count = spark.sql(f"SELECT COUNT(*) FROM house_{city}").collect()[0][0]
            print(f"house_{city}: {count} 条记录")
        except Exception as e:
            print(f"house_{city}: 查询失败 - {e}")
    
    print("\n=== 各城市工作数据统计 ===")
    for city in cities:
        try:
            count = spark.sql(f"SELECT COUNT(*) FROM job_{city}").collect()[0][0]
            print(f"job_{city}: {count} 条记录")
        except Exception as e:
            print(f"job_{city}: 查询失败 - {e}")

def read_and_clean_house_data(spark):
    """读取和清理房屋数据"""
    
    print("\n=== 开始清理房屋数据 ===")
    
    cities = ['beijing', 'shanghai', 'guangzhou', 'shenzhen', 'chongqing', 'taiyuan']
    all_data = []
    
    for city in cities:
        print(f"\n处理 {city} 房屋数据...")
        
        try:
            # 读取数据并过滤表头
            df = spark.sql(f"""
                SELECT * FROM house_{city}
                WHERE house_name != '房源名称' 
                  AND house_name IS NOT NULL
                  AND price IS NOT NULL
                  AND square IS NOT NULL
            """)
            
            # 添加城市列
            df = df.withColumn("city", lit(city))
            
            # 提取价格数字
            df = df.withColumn(
                "price_num", 
                regexp_extract(col("price"), r"(\d+)", 1).cast(DoubleType())
            )
            
            # 提取面积数字（取范围平均值）
            df = df.withColumn(
                "area_min", 
                regexp_extract(col("square"), r"(\d+)", 1).cast(DoubleType())
            ).withColumn(
                "area_max", 
                regexp_extract(col("square"), r"(\d+)-(\d+)", 2).cast(DoubleType())
            ).withColumn(
                "area_avg", 
                when(col("area_max").isNull(), col("area_min"))
                .otherwise((col("area_min") + col("area_max")) / 2)
            )
            
            # 提取房间数
            df = df.withColumn(
                "room_count",
                when(col("type").contains("4室"), 4)
                .when(col("type").contains("3室"), 3)
                .when(col("type").contains("2室"), 2)
                .when(col("type").contains("1室"), 1)
                .otherwise(2)
            )
            
            # 是否地铁房
            df = df.withColumn(
                "is_subway", 
                when(col("trait").contains("轨交房"), 1).otherwise(0)
            )
            
            # 是否有停车位
            df = df.withColumn(
                "has_parking", 
                when(col("trait").contains("车位充足"), 1).otherwise(0)
            )
            
            # 过滤有效数据
            df_clean = df.filter(
                (col("price_num") > 0) & 
                (col("area_avg") > 0) &
                (col("price_num") < 200000) &
                (col("area_avg") < 500) &
                (col("area_avg") > 20)
            )
            
            count = df_clean.count()
            print(f"{city} 清理后数据量: {count}")
            
            if count > 0:
                print(f"{city} 数据示例:")
                df_clean.select("house_name", "address", "price_num", "area_avg", "room_count").show(3, truncate=False)
                all_data.append(df_clean)
            
        except Exception as e:
            print(f"{city} 处理失败: {e}")
    
    if all_data:
        # 合并所有数据
        combined_df = all_data[0]
        for df in all_data[1:]:
            combined_df = combined_df.union(df)
        
        total_count = combined_df.count()
        print(f"\n总计处理 {total_count} 条有效房屋数据")
        
        return combined_df
    
    return None

def create_ml_features(df):
    """创建机器学习特征"""
    
    print("\n=== 创建机器学习特征 ===")
    
    # 创建价格等级标签（这是我们要预测的目标）
    df_ml = df.withColumn(
        "price_level",
        when(col("price_num") >= 50000, 2)  # 高价
        .when(col("price_num") >= 30000, 1)  # 中价
        .otherwise(0)                        # 低价
    ).withColumn(
        "price_label",
        when(col("price_num") >= 50000, "high")
        .when(col("price_num") >= 30000, "medium")
        .otherwise("low")
    )
    
    # 创建面积等级特征
    df_ml = df_ml.withColumn(
        "area_level",
        when(col("area_avg") >= 120, 2)  # 大户型
        .when(col("area_avg") >= 80, 1)   # 中户型
        .otherwise(0)                     # 小户型
    )
    
    print("各城市数据分布:")
    df_ml.groupBy("city").count().show()
    
    print("价格等级分布:")
    df_ml.groupBy("price_label").count().show()
    
    print("各城市价格等级分布:")
    df_ml.groupBy("city", "price_label").count().show()
    
    # 选择机器学习所需的特征列
    ml_features = df_ml.select(
        "city", "house_name", "address",
        "area_avg", "room_count", "is_subway", "has_parking",
        "price_num", "price_level", "price_label", "area_level"
    )
    
    print("机器学习特征数据示例:")
    ml_features.show(10)
    
    return ml_features

def save_results(spark, ml_df):
    """保存结果到新的Hive表"""
    
    print("\n=== 保存处理结果 ===")
    
    try:
        # 保存清理后的数据
        ml_df.write.mode("overwrite").saveAsTable("dbestate.house_ml_features")
        print("机器学习特征数据已保存到 dbestate.house_ml_features 表")
        
        # 创建统计汇总表
        summary_df = ml_df.groupBy("city", "price_label").count() \
            .withColumnRenamed("count", "house_count")
        
        summary_df.write.mode("overwrite").saveAsTable("dbestate.city_price_summary")
        print("城市价格汇总已保存到 dbestate.city_price_summary 表")
        
        print("\n保存的表:")
        spark.sql("SHOW TABLES").show()
        
    except Exception as e:
        print(f"保存失败: {e}")

def main():
    spark = create_spark_session()
    
    try:
        # 1. 分析现有数据
        analyze_data(spark)
        
        # 2. 读取和清理房屋数据
        clean_df = read_and_clean_house_data(spark)
        
        if clean_df is not None:
            # 3. 创建机器学习特征
            ml_df = create_ml_features(clean_df)
            
            # 4. 保存结果
            save_results(spark, ml_df)
            
            print("\n 数据预处理完成！")
        else:
            print("数据读取失败")
    
    except Exception as e:
        print(f"处理过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
