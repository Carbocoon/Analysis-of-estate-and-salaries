from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round, when, concat, lit
import os

def create_spark_session():
    spark = SparkSession.builder \
        .appName("导出中文列名CSV数据") \
        .enableHiveSupport() \
        .getOrCreate()
    
    spark.sql("USE dbestate")
    return spark

def check_table_structure(spark):
    """检查表结构"""
    print("=== 检查现有表结构 ===")
    
    tables = ["city_job_summary", "job_category_summary", "job_ml_features", "house_ml_features", "ml_predictions"]
    
    for table in tables:
        try:
            print(f"\n{table} 表结构:")
            desc = spark.sql(f"DESCRIBE dbestate.{table}")
            desc.show()
        except Exception as e:
            print(f"❌ {table} 表不存在: {e}")

def export_city_job_summary(spark):
    """导出城市工作汇总 - 中文列名"""
    print("=== 导出城市工作汇总数据 ===")
    
    city_job_df = spark.sql("""
        SELECT 
            city,
            job_count,
            ROUND(avg_salary, 0) as avg_salary
        FROM city_job_summary
        ORDER BY avg_salary DESC
    """)
    
    # 重命名为中文列名
    city_job_result = city_job_df.select(
        col("city").alias("城市"),
        col("job_count").alias("工作数量"),
        col("avg_salary").alias("平均薪资_元")
    )
    
    city_job_result.show()
    return city_job_result

def export_job_category_summary(spark):
    """导出职业类型汇总 - 中文列名（修复版）"""
    print("=== 导出职业类型汇总数据 ===")
    
    # 直接从原始表查询，计算住宿比例
    job_category_df = spark.sql("""
        SELECT 
            job_category,
            COUNT(*) as job_count,
            ROUND(AVG(salary_avg), 0) as avg_salary,
            ROUND(SUM(has_accommodation) * 100.0 / COUNT(*), 1) as accommodation_ratio
        FROM job_ml_features
        GROUP BY job_category
        ORDER BY avg_salary DESC
    """)
    
    # 重命名为中文列名
    job_category_result = job_category_df.select(
        col("job_category").alias("职业类型"),
        col("job_count").alias("岗位数量"),
        col("avg_salary").alias("平均薪资_元"),
        col("accommodation_ratio").alias("包住宿比例_百分比")
    )
    
    job_category_result.show()
    return job_category_result

def export_job_welfare_analysis(spark):
    """导出工作福利分析 - 中文列名（修复版）"""
    print("=== 导出工作福利分析数据 ===")
    
    # 直接从job_ml_features计算福利分析
    welfare_df = spark.sql("""
        SELECT 
            city,
            job_category,
            COUNT(*) as total_jobs,
            SUM(has_accommodation) as accommodation_jobs,
            SUM(has_insurance) as insurance_jobs,
            ROUND(SUM(has_accommodation) * 100.0 / COUNT(*), 1) as accommodation_rate,
            ROUND(SUM(has_insurance) * 100.0 / COUNT(*), 1) as insurance_rate,
            ROUND(AVG(salary_avg), 0) as avg_salary
        FROM job_ml_features
        GROUP BY city, job_category
        ORDER BY city, avg_salary DESC
    """)
    
    # 重命名为中文列名
    welfare_result = welfare_df.select(
        col("city").alias("城市"),
        col("job_category").alias("职业类型"),
        col("total_jobs").alias("总岗位数"),
        col("accommodation_jobs").alias("包住宿岗位数"),
        col("insurance_jobs").alias("有保险岗位数"),
        col("accommodation_rate").alias("住宿提供率_百分比"),
        col("insurance_rate").alias("保险提供率_百分比"),
        col("avg_salary").alias("平均薪资_元")
    )
    
    welfare_result.show()
    return welfare_result

def export_investment_analysis(spark):
    """导出投资分析 - 中文列名（修复版）"""
    print("=== 导出投资分析数据 ===")
    
    # 直接计算投资分析，不依赖预存表
    investment_df = spark.sql("""
        SELECT 
            j.city,
            ROUND(j.avg_salary, 0) as monthly_salary,
            ROUND(h.avg_price, 0) as price_per_sqm,
            ROUND(h.avg_area, 1) as avg_area,
            ROUND(h.avg_price * h.avg_area, 0) as total_house_price,
            ROUND((h.avg_price * h.avg_area) / (j.avg_salary * 12), 1) as years_to_buy,
            CASE 
                WHEN (h.avg_price * h.avg_area) / (j.avg_salary * 12) <= 15 THEN '容易购买'
                WHEN (h.avg_price * h.avg_area) / (j.avg_salary * 12) <= 25 THEN '中等难度'
                ELSE '困难购买'
            END as purchase_difficulty
        FROM city_job_summary j
        JOIN (
            SELECT 
                city,
                AVG(price_num) as avg_price,
                AVG(area_avg) as avg_area
            FROM house_ml_features
            GROUP BY city
        ) h ON j.city = h.city
        ORDER BY years_to_buy
    """)
    
    # 重命名为中文列名
    investment_result = investment_df.select(
        col("city").alias("城市"),
        col("monthly_salary").alias("月薪资_元"),
        col("price_per_sqm").alias("房价_元每平米"),
        col("avg_area").alias("平均面积_平米"),
        col("total_house_price").alias("房屋总价_元"),
        col("years_to_buy").alias("购房所需年数"),
        col("purchase_difficulty").alias("购房难度")
    )
    
    investment_result.show()
    return investment_result

def export_job_detail_data(spark):
    """导出工作详细数据 - 中文列名"""
    print("=== 导出工作详细数据 ===")
    
    job_detail_df = spark.sql("""
        SELECT 
            city,
            job_category,
            job_name,
            address,
            ROUND(salary_avg, 0) as salary_avg,
            salary_label,
            has_accommodation,
            has_insurance,
            is_urgent,
            is_daily_pay,
            no_experience_required
        FROM job_ml_features
        ORDER BY city, salary_avg DESC
    """)
    
    # 重命名为中文列名
    job_detail_result = job_detail_df.select(
        col("city").alias("城市"),
        col("job_category").alias("职业类型"),
        col("job_name").alias("职位名称"),
        col("address").alias("工作地址"),
        col("salary_avg").alias("平均薪资_元"),
        col("salary_label").alias("薪资等级"),
        when(col("has_accommodation") == 1, "包住宿").otherwise("不包住宿").alias("住宿情况"),
        when(col("has_insurance") == 1, "有保险").otherwise("无保险").alias("保险情况"),
        when(col("is_urgent") == 1, "急招").otherwise("普通").alias("招聘紧急度"),
        when(col("is_daily_pay") == 1, "日结").otherwise("月结").alias("薪资结算方式"),
        when(col("no_experience_required") == 1, "无经验要求").otherwise("有经验要求").alias("经验要求")
    )
    
    job_detail_result.show(10)
    return job_detail_result

def export_house_detail_data(spark):
    """导出房价详细数据 - 中文列名"""
    print("=== 导出房价详细数据 ===")
    
    house_detail_df = spark.sql("""
        SELECT 
            city,
            ROUND(area_avg, 1) as area_avg,
            room_count,
            ROUND(price_num, 0) as price_num,
            price_label,
            is_subway,
            has_parking
        FROM house_ml_features
        ORDER BY city, price_num DESC
    """)
    
    # 重命名为中文列名
    house_detail_result = house_detail_df.select(
        col("city").alias("城市"),
        col("area_avg").alias("面积_平米"),
        col("room_count").alias("房间数量"),
        col("price_num").alias("价格_元每平米"),
        col("price_label").alias("价格等级"),
        when(col("is_subway") == 1, "地铁房").otherwise("非地铁房").alias("地铁情况"),
        when(col("has_parking") == 1, "有停车位").otherwise("无停车位").alias("停车情况")
    )
    
    house_detail_result.show(10)
    return house_detail_result

def export_prediction_accuracy(spark):
    """导出预测准确率数据 - 中文列名"""
    print("=== 导出预测准确率数据 ===")
    
    # 房价预测准确率
    house_accuracy_df = spark.sql("""
        SELECT 
            city,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as accuracy_rate
        FROM ml_predictions
        GROUP BY city
        ORDER BY accuracy_rate DESC
    """)
    
    # 重命名为中文列名
    house_accuracy_result = house_accuracy_df.select(
        col("city").alias("城市"),
        col("total_predictions").alias("预测总数"),
        col("correct_predictions").alias("预测正确数量"),
        col("accuracy_rate").alias("预测准确率_百分比")
    )
    
    house_accuracy_result.show()
    return house_accuracy_result

def export_comprehensive_city_summary(spark):
    """导出城市综合汇总 - 按照你的格式"""
    print("=== 导出城市综合汇总数据 ===")
    
    comprehensive_df = spark.sql("""
        SELECT 
            h.city,
            h.house_count,
            ROUND(h.avg_price, 1) as avg_price,
            ROUND(h.avg_area, 1) as avg_area,
            h.subway_count,
            ROUND(h.subway_ratio, 1) as subway_ratio,
            COALESCE(a.correct_predictions, 0) as correct_predictions,
            ROUND(COALESCE(a.accuracy_rate, 0), 1) as accuracy_rate
        FROM (
            SELECT 
                city,
                COUNT(*) as house_count,
                AVG(price_num) as avg_price,
                AVG(area_avg) as avg_area,
                SUM(is_subway) as subway_count,
                SUM(is_subway) * 100.0 / COUNT(*) as subway_ratio
            FROM house_ml_features
            GROUP BY city
        ) h
        LEFT JOIN (
            SELECT 
                city,
                SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) as correct_predictions,
                SUM(CASE WHEN price_level = prediction THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy_rate
            FROM ml_predictions
            GROUP BY city
        ) a ON h.city = a.city
        ORDER BY h.avg_price DESC
    """)
    
    # 重命名为中文列名，格式与你的示例一致
    city_summary_result = comprehensive_df.select(
        col("city").alias("城市"),
        col("house_count").alias("房源数量"),
        col("avg_price").alias("平均价格_元每平米"),
        col("avg_area").alias("平均面积_平米"),
        col("subway_count").alias("地铁房数量"),
        col("subway_ratio").alias("地铁房比例_百分比"),
        col("correct_predictions").alias("预测正确数量"),
        col("accuracy_rate").alias("预测准确率_百分比")
    )
    
    city_summary_result.show()
    return city_summary_result

def save_to_csv(df, filename, output_path):
    """保存DataFrame为CSV文件"""
    try:
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 转换为Pandas DataFrame并保存
        pandas_df = df.toPandas()
        full_path = os.path.join(output_path, filename)
        pandas_df.to_csv(full_path, index=False, encoding='utf-8-sig')
        
        print(f"{filename} 已保存到 {full_path}")
        return True
    except Exception as e:
        print(f"保存 {filename} 失败: {e}")
        return False

def main():
    spark = create_spark_session()
    
    # 输出路径
    output_path = "/analysisProject/job_data"
    
    try:
        print("开始导出中文列名CSV数据")
        print("=" * 60)
        
        # 0. 检查表结构（可选）
        # check_table_structure(spark)
        
        # 1. 城市工作汇总
        city_job_df = export_city_job_summary(spark)
        save_to_csv(city_job_df, "city_job_summary.csv", output_path)
        
        # 2. 职业类型汇总（修复版）
        job_category_df = export_job_category_summary(spark)
        save_to_csv(job_category_df, "job_category_summary.csv", output_path)
        
        # 3. 工作福利分析（修复版）
        welfare_df = export_job_welfare_analysis(spark)
        save_to_csv(welfare_df, "job_welfare_analysis.csv", output_path)
        
        # 4. 投资分析（修复版）
        investment_df = export_investment_analysis(spark)
        save_to_csv(investment_df, "investment_analysis.csv", output_path)
        
        # 5. 工作详细数据
        job_detail_df = export_job_detail_data(spark)
        save_to_csv(job_detail_df, "job_detail_data.csv", output_path)
        
        # 6. 房价详细数据
        house_detail_df = export_house_detail_data(spark)
        save_to_csv(house_detail_df, "house_detail_data.csv", output_path)
        
        # 7. 预测准确率
        accuracy_df = export_prediction_accuracy(spark)
        save_to_csv(accuracy_df, "prediction_accuracy.csv", output_path)
        
        # 8. 城市综合汇总（按你的格式）
        city_summary_df = export_comprehensive_city_summary(spark)
        save_to_csv(city_summary_df, "city_summary.csv", output_path)
        
        print("\n" + "=" * 60)
        print("🎉 所有数据导出完成！")
        print(f"文件保存位置: {output_path}")
        print("\n导出的文件列表:")
        print("   - city_job_summary.csv (城市工作汇总)")
        print("   - job_category_summary.csv (职业类型汇总)")
        print("   - job_welfare_analysis.csv (工作福利分析)")
        print("   - investment_analysis.csv (投资分析)")
        print("   - job_detail_data.csv (工作详细数据)")
        print("   - house_detail_data.csv (房价详细数据)")
        print("   - prediction_accuracy.csv (预测准确率)")
        print("   - city_summary.csv (城市综合汇总)")
        print("\n所有列名均为中文，便于FineBI直接使用！")
        
        # 显示文件大小
        try:
            import subprocess
            result = subprocess.run(['ls', '-lh', output_path], capture_output=True, text=True)
            print(f"\n 文件详情:")
            print(result.stdout)
        except:
            pass
        
    except Exception as e:
        print(f"导出过程出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()

