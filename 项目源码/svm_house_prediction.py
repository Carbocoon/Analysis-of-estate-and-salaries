from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when

def create_spark_session():
    spark = SparkSession.builder \
        .appName("多分类房价预测模型") \
        .master("local[*]") \
        .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
        .enableHiveSupport() \
        .getOrCreate()
    
    spark.sql("USE dbestate")
    return spark

def load_training_data(spark):
    """加载训练数据"""
    print("=== 加载训练数据 ===")
    
    df = spark.sql("""
        SELECT 
            city,
            area_avg,           
            room_count,         
            is_subway,          
            has_parking,        
            price_num,          
            price_level,        
            price_label         
        FROM house_ml_features
        WHERE area_avg IS NOT NULL 
          AND room_count IS NOT NULL
          AND price_level IS NOT NULL
    """)
    
    print(f"总数据量: {df.count()}")
    
    print("各城市数据分布:")
    df.groupBy("city").count().show()
    
    print("价格等级分布:")
    df.groupBy("price_label", "price_level").count().show()
    
    return df

def train_random_forest_model(df):
    """使用随机森林进行多分类"""
    print("\n=== 训练随机森林多分类模型 ===")
    
    # 1. 特征准备
    feature_cols = ["area_avg", "room_count", "is_subway", "has_parking"]
    
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    # 2. 随机森林分类器（天然支持多分类）
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="price_level",
        numTrees=100,           # 100棵决策树
        maxDepth=5,             # 每棵树最大深度
        seed=42
    )
    
    # 3. 创建管道
    pipeline = Pipeline(stages=[assembler, rf])
    
    # 4. 分割数据
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"训练数据量: {train_data.count()}")
    print(f"测试数据量: {test_data.count()}")
    
    # 5. 训练模型
    print("开始训练随机森林模型...")
    model = pipeline.fit(train_data)
    print("模型训练完成!")
    
    # 6. 预测
    predictions = model.transform(test_data)
    
    return model, predictions, train_data, test_data

def train_binary_svm_models(df):
    """方案2：训练多个二分类SVM模型"""
    print("\n=== 训练二分类SVM模型组合 ===")
    
    from pyspark.ml.classification import LinearSVC
    
    # 准备特征
    feature_cols = ["area_avg", "room_count", "is_subway", "has_parking"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    
    results = {}
    
    # 模型1：高价 vs 非高价
    print("训练模型1：高价房 vs 非高价房...")
    df_binary1 = df.withColumn("binary_label1", when(col("price_level") == 2, 1).otherwise(0))
    
    svm1 = LinearSVC(featuresCol="scaled_features", labelCol="binary_label1", maxIter=100)
    pipeline1 = Pipeline(stages=[assembler, scaler, svm1])
    
    train1, test1 = df_binary1.randomSplit([0.8, 0.2], seed=42)
    model1 = pipeline1.fit(train1)
    pred1 = model1.transform(test1)
    
    eval1 = MulticlassClassificationEvaluator(labelCol="binary_label1", predictionCol="prediction")
    acc1 = eval1.evaluate(pred1)
    print(f"高价房识别准确率: {acc1:.2%}")
    
    # 模型2：中价 vs 低价（在非高价房中）
    print("训练模型2：中价房 vs 低价房...")
    df_binary2 = df.filter(col("price_level") != 2).withColumn("binary_label2", 
                                                              when(col("price_level") == 1, 1).otherwise(0))
    
    svm2 = LinearSVC(featuresCol="scaled_features", labelCol="binary_label2", maxIter=100)
    pipeline2 = Pipeline(stages=[assembler, scaler, svm2])
    
    train2, test2 = df_binary2.randomSplit([0.8, 0.2], seed=42)
    model2 = pipeline2.fit(train2)
    pred2 = model2.transform(test2)
    
    eval2 = MulticlassClassificationEvaluator(labelCol="binary_label2", predictionCol="prediction")
    acc2 = eval2.evaluate(pred2)
    print(f"中价房vs低价房识别准确率: {acc2:.2%}")
    
    return {"model1": model1, "model2": model2, "acc1": acc1, "acc2": acc2}

def evaluate_model(predictions):
    """评估模型性能"""
    print("\n=== 模型性能评估 ===")
    
    # 计算准确率
    evaluator = MulticlassClassificationEvaluator(
        labelCol="price_level",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    accuracy = evaluator.evaluate(predictions)
    print(f"整体准确率: {accuracy:.2%}")
    
    # 显示预测结果示例
    print("\n预测结果示例:")
    result_sample = predictions.select(
        "city", "area_avg", "room_count", "is_subway", "has_parking",
        "price_num", "price_level", "prediction"
    ).limit(10)
    
    result_sample.show()
    
    # 混淆矩阵
    print("\n混淆矩阵（实际 vs 预测）:")
    confusion = predictions.groupBy("price_level", "prediction").count()
    confusion.orderBy("price_level", "prediction").show()
    
    # 各等级准确率
    print("\n各价格等级预测准确率:")
    for level in [0, 1, 2]:
        level_data = predictions.filter(col("price_level") == level)
        if level_data.count() > 0:
            correct = level_data.filter(col("prediction") == level).count()
            total = level_data.count()
            level_accuracy = correct / total
            level_name = ["低价", "中价", "高价"][level]
            print(f"{level_name}房准确率: {level_accuracy:.2%} ({correct}/{total})")
    
    return accuracy

def feature_importance_analysis(model, spark):
    """分析特征重要性（仅适用于随机森林）"""
    print("\n=== 特征重要性分析 ===")
    
    try:
        # 获取随机森林模型
        rf_model = model.stages[-1]  # 管道的最后一个阶段
        
        # 特征重要性
        importances = rf_model.featureImportances.toArray()
        feature_names = ["area_avg", "room_count", "is_subway", "has_parking"]
        
        print("特征重要性排序:")
        for i, (name, importance) in enumerate(zip(feature_names, importances)):
            print(f"{i+1}. {name}: {importance:.3f}")
            
        # 找出最重要的特征
        max_idx = importances.argmax()
        print(f"\n最重要的特征: {feature_names[max_idx]} (重要性: {importances[max_idx]:.3f})")
        
    except Exception as e:
        print(f"特征重要性分析失败: {e}")

def make_predictions_on_new_data(model, spark):
    """预测新房屋数据"""
    print("\n=== 预测新房屋价格等级 ===")
    
    # 示例新房屋
    new_houses = [
        (150.0, 4, 1, 1),  # 大面积，4室，地铁房，有停车位 → 应该是高价
        (80.0, 2, 0, 0),   # 中等面积，2室，非地铁房，无停车位 → 应该是中低价
        (200.0, 5, 1, 1),  # 超大面积，5室，地铁房，有停车位 → 应该是高价
        (50.0, 1, 0, 0),   # 小面积，1室，非地铁房，无停车位 → 应该是低价
    ]
    
    new_df = spark.createDataFrame(
        new_houses, 
        ["area_avg", "room_count", "is_subway", "has_parking"]
    )
    
    # 预测
    new_predictions = model.transform(new_df)
    
    # 添加标签
    result = new_predictions.withColumn(
        "predicted_label",
        when(col("prediction") == 0, "低价房")
        .when(col("prediction") == 1, "中价房")
        .otherwise("高价房")
    ).withColumn(
        "subway_text",
        when(col("is_subway") == 1, "地铁房").otherwise("非地铁房")
    ).withColumn(
        "parking_text", 
        when(col("has_parking") == 1, "有停车位").otherwise("无停车位")
    )
    
    print("新房屋价格预测:")
    result.select("area_avg", "room_count", "subway_text", "parking_text", "predicted_label").show(truncate=False)

def save_results(model, predictions, spark):
    """保存结果"""
    print("\n=== 保存预测结果 ===")
    
    try:
        # 保存预测结果
        predictions.select(
            "city", "area_avg", "room_count", "is_subway", "has_parking",
            "price_num", "price_level", "prediction"
        ).write.mode("overwrite").saveAsTable("dbestate.ml_predictions")
        
        print("预测结果已保存到 dbestate.ml_predictions")
        
        # 创建准确率汇总
        accuracy_by_city = predictions.groupBy("city", "price_level", "prediction").count()
        accuracy_by_city.write.mode("overwrite").saveAsTable("dbestate.ml_accuracy_by_city")
        
        print("城市准确率统计已保存到 dbestate.ml_accuracy_by_city")
        
    except Exception as e:
        print(f"保存失败: {e}")

def main():
    """主函数"""
    spark = create_spark_session()
    
    try:
        print("开始多分类房价预测模型训练")
        print("=" * 50)
        
        # 1. 加载数据
        df = load_training_data(spark)
        
        # 2. 训练随机森林模型（推荐）
        model, predictions, train_data, test_data = train_random_forest_model(df)
        
        # 3. 评估模型
        accuracy = evaluate_model(predictions)
        
        # 4. 特征重要性分析
        feature_importance_analysis(model, spark)
        
        # 5. 预测新数据
        make_predictions_on_new_data(model, spark)
        
        # 6. 保存结果
        save_results(model, predictions, spark)
        
        # 7. 额外：试试二分类SVM组合（可选）
        print("\n" + "="*30 + " 额外尝试 " + "="*30)
        binary_models = train_binary_svm_models(df)
        
        print("\n" + "=" * 50)
        print("模型训练完成！")
        print(f"最终准确率: {accuracy:.2%}")
        print("可以使用FineBI连接 dbestate.ml_predictions 表进行可视化")
        
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
