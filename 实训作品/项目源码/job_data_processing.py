from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when, split, trim, lower, substring, lit, regexp_replace
from pyspark.sql.types import DoubleType, IntegerType
import re

def create_spark_session():
    spark = SparkSession.builder \
        .appName("58同城工作数据预处理") \
        .enableHiveSupport() \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sql("USE dbestate")
    return spark

def clean_job_data(spark):
    """清理工作数据"""
    print("=== 开始清理工作数据 ===")
    
    cities = ['beijing', 'shanghai', 'guangzhou', 'shenzhen', 'chongqing', 'taiyuan']
    all_job_data = []
    
    for city in cities:
        print(f"\n处理 {city} 工作数据...")
        
        try:
            # 方法1：先查看表结构
            print(f"检查 job_{city} 表结构...")
            table_desc = spark.sql(f"DESCRIBE job_{city}")
            print("表结构:")
            table_desc.show()
            
            # 方法2：使用SELECT * 然后重命名，避免中文列名在WHERE中的问题
            df = spark.sql(f"SELECT * FROM job_{city}")
            
            # 获取列名
            columns = df.columns
            print(f"原始列名: {columns}")
            
            # 根据列位置重命名（假设第一列是职业，第二列是地址，第三列是薪资）
            if len(columns) >= 3:
                if len(columns) == 3:
                    df = df.toDF("job_name", "address", "salary")
                else:
                    # 如果有更多列，保留原有名称但重命名前三列
                    new_names = ["job_name", "address", "salary"] + columns[3:]
                    df = df.toDF(*new_names)
            
            # 添加城市列
            df = df.withColumn("city", lit(city))
            
            # 过滤掉表头行和空值
            df = df.filter(
                (col("job_name") != "职业") & 
                (col("job_name").isNotNull()) & 
                (col("salary").isNotNull()) &
                (col("job_name") != "") &
                (col("salary") != "")
            )
            
            print(f"{city} 原始数据量: {df.count()}")
            
            if df.count() == 0:
                print(f"{city} 无有效数据，跳过")
                continue
            
            # 显示样本数据
            print(f"{city} 原始数据示例:")
            df.show(3, truncate=False)
            
            # 清理薪资数据
            cleaned_df = df.withColumn(
                "salary_clean", 
                regexp_replace(col("salary"), r"[元,，]", "")
            )
            
            # 提取薪资范围
            cleaned_df = cleaned_df.withColumn(
                "salary_min",
                regexp_extract(col("salary_clean"), r"(\d+)", 1).cast(DoubleType())
            ).withColumn(
                "salary_max", 
                regexp_extract(col("salary_clean"), r"(\d+)-(\d+)", 2).cast(DoubleType())
            ).withColumn(
                "salary_avg",
                when(col("salary_max").isNull(), col("salary_min"))
                .otherwise((col("salary_min") + col("salary_max")) / 2)
            )
            
            # 处理"面议"等特殊情况
            cleaned_df = cleaned_df.withColumn(
                "salary_avg",
                when(col("salary").contains("面议"), lit(8000.0))  # 面议设为8000默认值
                .when(col("salary").contains("元/天"), col("salary_min") * 30)  # 日薪转月薪
                .when(col("salary").contains("元/小时"), col("salary_min") * 8 * 30)  # 时薪转月薪
                .otherwise(col("salary_avg"))
            )
            
            # 职业类型分类
            cleaned_df = cleaned_df.withColumn(
                "job_category",
                when(col("job_name").contains("司机") | col("job_name").contains("驾驶"), "司机")
                .when(col("job_name").contains("送餐") | col("job_name").contains("外卖") | col("job_name").contains("骑手"), "外卖配送")
                .when(col("job_name").contains("模特") | col("job_name").contains("试衣"), "模特展示")
                .when(col("job_name").contains("按摩") | col("job_name").contains("理疗") | col("job_name").contains("足疗"), "按摩理疗")
                .when(col("job_name").contains("服务员") | col("job_name").contains("接待"), "服务接待")
                .when(col("job_name").contains("保安") | col("job_name").contains("安检"), "安保安检")
                .when(col("job_name").contains("包装") | col("job_name").contains("普工") | col("job_name").contains("搬运"), "普工包装")
                .when(col("job_name").contains("销售") | col("job_name").contains("营销"), "销售营销")
                .when(col("job_name").contains("教练") | col("job_name").contains("健身"), "健身教练")
                .when(col("job_name").contains("月嫂") | col("job_name").contains("育婴") | col("job_name").contains("护工"), "护理育婴")
                .when(col("job_name").contains("快递") | col("job_name").contains("配送"), "快递配送")
                .when(col("job_name").contains("客服") | col("job_name").contains("电话"), "客服电话")
                .otherwise("其他职业")
            )
            
            # 工作特征提取
            cleaned_df = cleaned_df.withColumn(
                "has_accommodation", 
                when(col("job_name").contains("包住") | col("job_name").contains("食宿"), 1).otherwise(0)
            ).withColumn(
                "has_insurance",
                when(col("job_name").contains("五险") | col("job_name").contains("保险"), 1).otherwise(0)
            ).withColumn(
                "is_urgent",
                when(col("job_name").contains("急招") | col("job_name").contains("急聘"), 1).otherwise(0)
            ).withColumn(
                "is_daily_pay",
                when(col("job_name").contains("日结") | col("job_name").contains("天结"), 1).otherwise(0)
            ).withColumn(
                "no_experience_required",
                when(col("job_name").contains("无经验") | col("job_name").contains("不限工龄") | col("job_name").contains("新人"), 1).otherwise(0)
            )
            
            # 过滤有效数据
            final_df = cleaned_df.filter(
                (col("salary_avg").isNotNull()) & 
                (col("salary_avg") > 0) &
                (col("salary_avg") < 50000) &  # 过滤异常高薪
                (col("salary_avg") >= 2000)    # 过滤异常低薪
            ).select(
                "city", "job_name", "address", "job_category",
                "salary_avg", "salary_min", "salary_max",
                "has_accommodation", "has_insurance", "is_urgent", 
                "is_daily_pay", "no_experience_required"
            )
            
            count = final_df.count()
            print(f"{city} 清理后数据量: {count}")
            
            if count > 0:
                print(f"{city} 清理后数据示例:")
                final_df.show(5, truncate=False)
                all_job_data.append(final_df)
                
        except Exception as e:
            print(f"❌ {city} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 合并所有城市数据
    if all_job_data:
        combined_df = all_job_data[0]
        for df in all_job_data[1:]:
            combined_df = combined_df.union(df)
        
        total_count = combined_df.count()
        print(f"\n总计处理 {total_count} 条有效工作数据")
        
        # 显示总体统计
        print("\n总体数据概览:")
        combined_df.groupBy("city").count().show()
        combined_df.groupBy("job_category").count().show()
        
        return combined_df
    
    return None

def create_job_ml_features(df):
    """创建工作机器学习特征"""
    print("\n=== 创建工作机器学习特征 ===")
    
    # 创建薪资等级标签
    df_ml = df.withColumn(
        "salary_level",
        when(col("salary_avg") >= 12000, 2)  # 高薪：12000+
        .when(col("salary_avg") >= 8000, 1)   # 中薪：8000-12000
        .otherwise(0)                         # 低薪：8000以下
    ).withColumn(
        "salary_label",
        when(col("salary_avg") >= 12000, "high")
        .when(col("salary_avg") >= 8000, "medium")
        .otherwise("low")
    )
    
    # 创建城市经济等级
    df_ml = df_ml.withColumn(
        "city_level",
        when(col("city").isin(["beijing", "shanghai", "shenzhen"]), 2)  # 一线城市
        .when(col("city").isin(["guangzhou", "chongqing"]), 1)          # 新一线城市
        .otherwise(0)                                                    # 二线城市
    )
    
    print("各城市工作数据分布:")
    df_ml.groupBy("city").count().show()
    
    print("薪资等级分布:")
    df_ml.groupBy("salary_label").count().show()
    
    print("职业类型分布:")
    df_ml.groupBy("job_category").count().show()
    
    print("各城市薪资等级分布:")
    df_ml.groupBy("city", "salary_label").count().show()
    
    print("各城市平均薪资:")
    df_ml.groupBy("city").agg({"salary_avg": "avg"}).show()
    
    return df_ml

def analyze_job_market(df):
    """分析就业市场趋势"""
    print("\n=== 就业市场分析 ===")
    
    # 各城市平均薪资
    print("各城市平均薪资:")
    city_salary = df.groupBy("city").agg(
        {"salary_avg": "avg", "*": "count"}
    ).withColumnRenamed("avg(salary_avg)", "avg_salary") \
     .withColumnRenamed("count(1)", "job_count")
    city_salary.orderBy("avg_salary", ascending=False).show()
    
    # 各职业类型平均薪资
    print("各职业类型平均薪资:")
    job_salary = df.groupBy("job_category").agg(
        {"salary_avg": "avg", "*": "count"}
    ).withColumnRenamed("avg(salary_avg)", "avg_salary") \
     .withColumnRenamed("count(1)", "job_count")
    job_salary.orderBy("avg_salary", ascending=False).show()
    
    # 福利待遇分析
    print("福利待遇统计:")
    welfare_stats = df.select(
        "has_accommodation", "has_insurance", "is_urgent", 
        "is_daily_pay", "no_experience_required"
    ).describe()
    welfare_stats.show()
    
    return city_salary, job_salary

def save_job_results(spark, ml_df, city_salary, job_salary):
    """保存工作数据分析结果"""
    print("\n=== 保存工作分析结果 ===")
    
    try:
        # 保存机器学习特征数据
        ml_df.write.mode("overwrite").saveAsTable("dbestate.job_ml_features")
        print("工作机器学习特征已保存到 dbestate.job_ml_features")
        
        # 保存城市薪资汇总
        city_salary.write.mode("overwrite").saveAsTable("dbestate.city_job_summary")
        print("城市工作汇总已保存到 dbestate.city_job_summary")
        
        # 保存职业薪资汇总
        job_salary.write.mode("overwrite").saveAsTable("dbestate.job_category_summary")
        print("职业类型汇总已保存到 dbestate.job_category_summary")
        
        # 创建综合分析表
        comprehensive_analysis = ml_df.groupBy("city", "job_category", "salary_label").count()
        comprehensive_analysis.write.mode("overwrite").saveAsTable("dbestate.job_comprehensive_analysis")
        print("综合分析已保存到 dbestate.job_comprehensive_analysis")
        
        print("\n数据保存完成，可以进行下一步分析！")
        
    except Exception as e:
        print(f"保存失败: {e}")

def main():
    spark = create_spark_session()
    
    try:
        print("开始工作数据预处理分析")
        print("=" * 50)
        
        # 1. 清理工作数据
        job_df = clean_job_data(spark)
        
        if job_df is not None:
            # 2. 创建机器学习特征
            ml_df = create_job_ml_features(job_df)
            
            # 3. 分析就业市场
            city_salary, job_salary = analyze_job_market(job_df)
            
            # 4. 保存结果
            save_job_results(spark, ml_df, city_salary, job_salary)
            
            print("\n工作数据预处理完成！") 
        else:
            print("工作数据读取失败")
    
    except Exception as e:
        print(f"处理过程出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
