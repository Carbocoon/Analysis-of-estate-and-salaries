from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when

def create_spark_session():
    spark = SparkSession.builder \
        .appName("工作薪资预测模型") \
        .enableHiveSupport() \
        .getOrCreate()
    
    spark.sql("USE dbestate")
    return spark

def load_job_data(spark):
    """加载工作数据"""
    print("=== 加载工作数据 ===")
    
    df = spark.sql("""
        SELECT 
            city, job_category, salary_avg, salary_level, salary_label,
            has_accommodation, has_insurance, is_urgent, 
            is_daily_pay, no_experience_required, city_level
        FROM job_ml_features
        WHERE salary_level IS NOT NULL
    """)
    
    print(f"总工作数据量: {df.count()}")
    return df

def prepare_job_features(df):
    """准备工作特征"""
    print("\n=== 准备工作特征 ===")
    
    # 字符串特征编码
    city_indexer = StringIndexer(inputCol="city", outputCol="city_index")
    category_indexer = StringIndexer(inputCol="job_category", outputCol="category_index")
    
    # 独热编码
    city_encoder = OneHotEncoder(inputCol="city_index", outputCol="city_vec")
    category_encoder = OneHotEncoder(inputCol="category_index", outputCol="category_vec")
    
    # 特征向量组装
    feature_cols = [
        "city_vec", "category_vec", "city_level",
        "has_accommodation", "has_insurance", "is_urgent", 
        "is_daily_pay", "no_experience_required"
    ]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    return city_indexer, category_indexer, city_encoder, category_encoder, assembler

def train_salary_model(df):
    """训练薪资预测模型"""
    print("\n=== 训练薪资预测模型 ===")
    
    # 准备特征处理步骤
    city_indexer, category_indexer, city_encoder, category_encoder, assembler = prepare_job_features(df)
    
    # 随机森林分类器
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="salary_level",
        numTrees=150,
        maxDepth=6,
        seed=42
    )
    
    # 创建机器学习管道
    pipeline = Pipeline(stages=[
        city_indexer, category_indexer, 
        city_encoder, category_encoder, 
        assembler, rf
    ])
    
    # 分割数据
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    print(f"训练数据: {train_data.count()}, 测试数据: {test_data.count()}")
    
    # 训练模型
    print("开始训练模型...")
    model = pipeline.fit(train_data)
    
    # 预测
    predictions = model.transform(test_data)
    
    return model, predictions, train_data, test_data

def evaluate_job_model(predictions):
    """评估工作薪资预测模型"""
    print("\n=== 模型评估 ===")
    
    # 计算准确率
    evaluator = MulticlassClassificationEvaluator(
        labelCol="salary_level",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    accuracy = evaluator.evaluate(predictions)
    print(f"薪资预测准确率: {accuracy:.2%}")
    
    # 显示预测结果
    print("\n预测结果示例:")
    predictions.select(
        "city", "job_category", "salary_avg", 
        "salary_level", "prediction", "salary_label"
    ).show(10, truncate=False)
    
    # 混淆矩阵
    print("\n预测效果分析:")
    confusion = predictions.groupBy("salary_level", "prediction").count()
    confusion.orderBy("salary_level", "prediction").show()
    
    # 各薪资等级准确率
    print("\n各薪资等级预测准确率:")
    for level in [0, 1, 2]:
        level_data = predictions.filter(col("salary_level") == level)
        if level_data.count() > 0:
            correct = level_data.filter(col("prediction") == level).count()
            total = level_data.count()
            level_accuracy = correct / total
            level_name = ["低薪", "中薪", "高薪"][level]
            print(f"{level_name}工作准确率: {level_accuracy:.2%} ({correct}/{total})")

def job_recommendation_analysis(model, spark):
    """工作推荐分析"""
    print("\n=== 工作推荐分析 ===")
    
    # 创建不同工作场景
    job_scenarios = [
        ("beijing", "司机", 2, 1, 0, 0, 0, 0),      # 北京司机，有住宿
        ("shanghai", "外卖配送", 1, 1, 1, 1, 1, 1), # 上海外卖，全福利
        ("guangzhou", "按摩理疗", 1, 1, 0, 0, 0, 0), # 广州按摩
        ("shenzhen", "销售营销", 2, 0, 1, 0, 0, 1),  # 深圳销售
        ("taiyuan", "普工包装", 0, 1, 0, 1, 0, 1),   # 太原普工
    ]
    
    scenario_df = spark.createDataFrame(
        job_scenarios,
        ["city", "job_category", "city_level", "has_accommodation", 
         "has_insurance", "is_urgent", "is_daily_pay", "no_experience_required"]
    )
    
    # 预测薪资等级
    scenario_predictions = model.transform(scenario_df)
    
    # 添加薪资标签
    result = scenario_predictions.withColumn(
        "predicted_salary_label",
        when(col("prediction") == 0, "低薪(2000-8000)")
        .when(col("prediction") == 1, "中薪(8000-12000)")
        .otherwise("高薪(12000+)")
    )
    
    print("不同工作场景薪资预测:")
    result.select(
        "city", "job_category", "has_accommodation", 
        "has_insurance", "predicted_salary_label"
    ).show(truncate=False)

def save_job_model_results(model, predictions, spark):
    """保存工作模型结果"""
    print("\n=== 保存工作预测结果 ===")
    
    try:
        # 保存预测结果
        predictions.select(
            "city", "job_category", "salary_avg", 
            "salary_level", "prediction", "salary_label",
            "has_accommodation", "has_insurance"
        ).write.mode("overwrite").saveAsTable("dbestate.job_salary_predictions")
        
        print("工作薪资预测结果已保存到 dbestate.job_salary_predictions")
        
        # 保存准确率统计
        accuracy_stats = predictions.groupBy("city", "job_category", "salary_level", "prediction").count()
        accuracy_stats.write.mode("overwrite").saveAsTable("dbestate.job_prediction_accuracy")
        
        print("预测准确率统计已保存到 dbestate.job_prediction_accuracy")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def main():
    spark = create_spark_session()
    
    try:
        print("开始工作薪资预测模型训练")
        print("=" * 50)
        
        # 1. 加载数据
        df = load_job_data(spark)
        
        # 2. 训练模型
        model, predictions, train_data, test_data = train_salary_model(df)
        
        # 3. 评估模型
        evaluate_job_model(predictions)
        
        # 4. 工作推荐分析
        job_recommendation_analysis(model, spark)
        
        # 5. 保存结果
        save_job_model_results(model, predictions, spark)
        
        print("\n" + "=" * 50)
        print("工作薪资预测模型训练完成！")
        print("可以使用FineBI连接相关表进行可视化分析")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
