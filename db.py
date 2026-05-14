
from sqlalchemy import create_engine
import pandas as pd

DB_URL = "mysql+pymysql://root:admin123@localhost/security_model"

engine = create_engine(DB_URL)


def load_data():
    query = """
    SELECT 
        t.id AS threat_id,
        t.name AS threat_name,
        t.category,
        t.severity,
        t.likelihood,
        t.impact,
        t.impact_rating,
        c.id AS control_id,
        c.name AS control_name,
        c.effectiveness
    FROM threats t
    JOIN control_threat_mappings m ON t.id = m.threat_id
    JOIN controls c ON c.id = m.control_id
    """

    df = pd.read_sql(query, engine)

    numeric_cols = ["severity", "likelihood", "impact_rating", "effectiveness"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["threat_name_norm"] = df["threat_name"].astype(str).str.strip().str.lower()
    df["control_name_norm"] = df["control_name"].astype(str).str.strip().str.lower()

    return df.fillna(0)








