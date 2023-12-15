from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
 
# DB_CONNECT_STRING = "postgresql://postgres:123456@localhost/shop"
# engine = create_engine(DB_CONNECT_STRING, echo=True)
# session = sessionmaker(engine)

DB_CONNECT_STRING = "postgresql+asyncpg://postgres:123456@localhost/shop"
engine = create_async_engine(DB_CONNECT_STRING, echo=True)
session = async_sessionmaker(engine, expire_on_commit= False)

def get_db_async_connection():
    return engine.connect() 

def get_db_async_session():
    return session()