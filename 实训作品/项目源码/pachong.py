import requests
import random
import time
from bs4 import BeautifulSoup
import pymysql

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',  # 请填入您的数据库密码
    'database': '58_data',
    'charset': 'utf8mb4'
}

# 检查代理可用性
def check_proxy(proxy):
    try:
        response = requests.get(url='https://www.example.com', 
                              proxies={'http': proxy, 'https': proxy}, 
                              timeout=5)
        return response.status_code == 200
    except:
        return False

# 构建代理池
def build_proxy_pool():
    proxies = [
        '127.0.0.1:8080',
    ]
    valid_proxies = [proxy for proxy in proxies if check_proxy(proxy)]
    return valid_proxies

# 数据库连接
def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

# 工作类
class Job:
    def __init__(self, name, address, salary):
        self.name = name
        self.address = address
        self.salary = salary
    
    def insert(self, city):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            sql = f"INSERT INTO job_{city} (name, address, salary) VALUES (%s, %s, %s)"
            cursor.execute(sql, (self.name, self.address, self.salary))
            conn.commit()
            conn.close()
            print(f"成功插入工作数据: {self.name}")
        except Exception as e:
            print(f"插入工作数据失败: {e}")

# 房源类
class House:
    def __init__(self, title, price, area, address):
        self.title = title
        self.price = price
        self.area = area
        self.address = address
    
    def insert(self, city):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            sql = f"INSERT INTO house_{city} (title, price, area, address) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (self.title, self.price, self.area, self.address))
            conn.commit()
            conn.close()
            print(f"成功插入房源数据: {self.title}")
        except Exception as e:
            print(f"插入房源数据失败: {e}")

# 招聘信息爬取
def zhaoping_information(resp1, city):
    page1 = BeautifulSoup(resp1.text, features="html.parser")
    ul = page1.find(name="ul", attrs={"id": "list_con"})
    
    if ul:
        for li in ul.find_all("li"):
            try:
                div = li.find("div", attrs={"class": "item_con job_title"})
                if not div:
                    continue
                    
                p = div.find("p", attrs={"class": "job_salary"})
                salary = ""
                if p:
                    salary = p.text.strip()
                else:
                    salary = "面议"
                
                div1 = div.find("div", attrs={"class": "job_name clearfix"})
                if not div1:
                    continue
                    
                name_span = div1.find("span", attrs={"class": "name"})
                address_span = div1.find("span", attrs={"class": "address"})
                
                if name_span and address_span:
                    name = name_span.text.strip()
                    address = address_span.text.strip()
                    
                    print(f"薪资: {salary}")
                    print(f"职位: {name}")
                    print(f"地址: {address}")
                    
                    job_item = Job(name, address, salary)
                    job_item.insert(city)
                    
            except Exception as e:
                print(f"解析招聘信息失败: {e}")
                continue

# 房源信息爬取
def house_information(resp1, city):
    page1 = BeautifulSoup(resp1.text, features="html.parser")
    
    # 寻找房源列表容器
    house_list = page1.find("div", attrs={"class": "list_con"}) or page1.find("ul", attrs={"id": "list_con"})
    
    if house_list:
        # 查找所有房源项
        house_items = house_list.find_all("div", attrs={"class": "item_mod"}) or house_list.find_all("li")
        
        for item in house_items:
            try:
                # 提取标题
                title_elem = item.find("a", attrs={"class": "title"}) or item.find("h3")
                title = title_elem.text.strip() if title_elem else "无标题"
                
                # 提取价格
                price_elem = item.find("span", attrs={"class": "price"}) or item.find("div", attrs={"class": "price"})
                price = price_elem.text.strip() if price_elem else "价格面议"
                
                # 提取面积
                area_elem = item.find("span", attrs={"class": "area"})
                area = area_elem.text.strip() if area_elem else "面积未知"
                
                # 提取地址
                address_elem = item.find("span", attrs={"class": "address"}) or item.find("div", attrs={"class": "address"})
                address = address_elem.text.strip() if address_elem else "地址未知"
                
                print(f"标题: {title}")
                print(f"价格: {price}")
                print(f"面积: {area}")
                print(f"地址: {address}")
                
                house_item = House(title, price, area, address)
                house_item.insert(city)
                
            except Exception as e:
                print(f"解析房源信息失败: {e}")
                continue

# 初始化数据库表
def init_database():
    cities = ["beijing", "shanghai", "guangzhou", "shenzhen", "chongqing", "taiyuan"]
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 创建数据库（如果不存在）
        cursor.execute("CREATE DATABASE IF NOT EXISTS 58_data")
        cursor.execute("USE 58_data")
        
        for city in cities:
            # 创建工作表
            job_table_sql = f"""
            CREATE TABLE IF NOT EXISTS job_{city} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                address VARCHAR(255),
                salary VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(job_table_sql)
            
            # 创建房源表
            house_table_sql = f"""
            CREATE TABLE IF NOT EXISTS house_{city} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255),
                price VARCHAR(100),
                area VARCHAR(100),
                address VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(house_table_sql)
        
        conn.commit()
        conn.close()
        print("数据库初始化完成")
        
    except Exception as e:
        print(f"数据库初始化失败: {e}")

# 主爬虫函数
def main():
    # 初始化数据库
    init_database()
    
    # 城市列表
    cities = ["beijing", "shanghai", "guangzhou", "shenzhen", "chongqing", "taiyuan"]
    
    # 请求头
    header = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.6.3 Safari/537.36 Edg/128.0.0.0"
    }
    
    # 构建代理池
    proxies_build = build_proxy_pool()
    
    for city in cities:
        print(f"开始爬取{city}的数据...")
        
        # 爬取招聘信息
        for page in range(1, 51):  # 爬取50页
            try:
                # 58同城招聘URL格式
                URL_zhaoping = f'https://{city}.58.com/job/'
                if page > 1:
                    URL_zhaoping = f'https://{city}.58.com/job/pn{page}/'
                
                if proxies_build:
                    proxy = random.choice(proxies_build)
                    proxies = {'http': proxy}
                else:
                    proxies = None
                
                resp1 = requests.get(URL_zhaoping, headers=header, proxies=proxies, timeout=10)
                print(f"正在爬取{city}招聘信息第{page}页")
                zhaoping_information(resp1, city)
                time.sleep(random.randint(2, 5))  # 随机延时避免被封
                
            except Exception as e:
                print(f"爬取{city}招聘信息第{page}页失败: {e}")
                continue
        
        # 爬取新房信息
        for page in range(1, 51):  # 爬取50页
            try:
                # 从截图看是新房楼盘信息
                URL_house = f'https://{city}.58.com/xinfang/loupan/all/p{page}/'
                
                if proxies_build:
                    proxy = random.choice(proxies_build)
                    proxies = {'http': proxy}
                else:
                    proxies = None
                
                resp1 = requests.get(URL_house, headers=header, proxies=proxies, timeout=10)
                print(f"正在爬取{city}房源信息第{page}页")
                house_information(resp1, city)
                time.sleep(random.randint(2, 5))  # 随机延时避免被封
                
            except Exception as e:
                print(f"爬取{city}房源信息第{page}页失败: {e}")
                continue
        
        print(f"{city}数据爬取完成")

if __name__ == "__main__":
    main()