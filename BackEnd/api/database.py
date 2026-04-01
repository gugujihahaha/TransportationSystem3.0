from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# 使用 SQLite 本地数据库文件
SQLALCHEMY_DATABASE_URL = "sqlite:///./transportation.db"

# connect_args={"check_same_thread": False} 是 SQLite 在 FastAPI 中必须的参数
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 依赖注入函数：每次请求获取一个数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()