from sqlalchemy import Column, String, DateTime, Text
from .database import Base
from datetime import datetime
import uuid

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    filename = Column(String, index=True)
    file_path = Column(String, nullable=True)
    status = Column(String, default="pending") # pending, processing, done, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Store JSON result as text
    result_json = Column(Text, nullable=True)
