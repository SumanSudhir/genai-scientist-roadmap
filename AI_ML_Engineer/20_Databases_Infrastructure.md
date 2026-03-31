# Topic 20: Databases & Data Infrastructure for ML

> **Track**: AI/ML Engineer — Practice-First, Code-Heavy
> **Prerequisites**: Topic 17 (ML System Design), Topic 18 (Model Deployment)
> **You will build**: SQL feature extraction queries, a Redis feature cache, a ChromaDB vector search system, and a complete data pipeline (API → PostgreSQL → features → Redis)

---

## Table of Contents

1. [Why Data Infrastructure Matters for ML Engineers](#1-why-data-infrastructure-matters-for-ml-engineers)
2. [SQL for ML Engineers — Core Patterns](#2-sql-for-ml-engineers--core-patterns)
3. [SQL Window Functions — The ML Engineer's Power Tool](#3-sql-window-functions--the-ml-engineers-power-tool)
4. [SQL Feature Extraction Queries](#4-sql-feature-extraction-queries)
5. [PostgreSQL — Setup and ML-Relevant Features](#5-postgresql--setup-and-ml-relevant-features)
6. [NoSQL — MongoDB for Document Storage](#6-nosql--mongodb-for-document-storage)
7. [Redis — Caching and Feature Serving](#7-redis--caching-and-feature-serving)
8. [Redis as a Feature Store for ML](#8-redis-as-a-feature-store-for-ml)
9. [Vector Databases — Concepts](#9-vector-databases--concepts)
10. [FAISS — Fast Similarity Search](#10-faiss--fast-similarity-search)
11. [ChromaDB — Embedded Vector Database](#11-chromadb--embedded-vector-database)
12. [Pinecone — Managed Vector Database](#12-pinecone--managed-vector-database)
13. [Qdrant — Production Vector Search](#13-qdrant--production-vector-search)
14. [Vector Indexing Algorithms — HNSW, IVF, PQ](#14-vector-indexing-algorithms--hnsw-ivf-pq)
15. [Data Warehousing — BigQuery, Snowflake, Redshift](#15-data-warehousing--bigquery-snowflake-redshift)
16. [Streaming — Kafka for Real-Time ML Pipelines](#16-streaming--kafka-for-real-time-ml-pipelines)
17. [Choosing the Right Database](#17-choosing-the-right-database)
18. [Practice Exercises](#18-practice-exercises)
19. [Mini-Project: Data Pipeline — API → PostgreSQL → Features → Redis](#19-mini-project-data-pipeline--api--postgresql--features--redis)
20. [Interview Questions & Answers](#20-interview-questions--answers)

---

## 1. Why Data Infrastructure Matters for ML Engineers

```
┌──────────────────────────────────────────────────────────────┐
│         Where ML Engineers Spend Their Time                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Data collection & cleaning    ████████████████████  40%     │
│  Feature engineering (SQL!)    ██████████            20%     │
│  Model training & tuning       ██████                15%     │
│  Serving & infrastructure      ██████                15%     │
│  Monitoring & debugging        ████                  10%     │
│                                                              │
│  → 60% of your work involves databases.                      │
│  → SQL is used in every ML interview.                        │
│  → Redis can 10x your serving latency.                       │
│  → Vector DBs are essential for RAG and search.              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### The ML Data Stack

```mermaid
flowchart LR
    A[Raw Events] --> B[Message Queue / Kafka]
    B --> C[Data Warehouse / BigQuery]
    C --> D[Feature Pipelines / SQL + Spark]
    D --> E[Feature Store / Redis]
    E --> F[ML Model Serving]
    D --> G[Training Data / Parquet + S3]
    G --> H[Model Training]
    F --> I[Vector DB / Qdrant]
```

---

## 2. SQL for ML Engineers — Core Patterns

Every ML engineer interview includes SQL. These are the patterns you **must** know cold.

### Joins

```sql
-- ── INNER JOIN: only matching rows ──────────────────────────
-- Get user features with their transactions
SELECT
    u.user_id,
    u.signup_date,
    u.country,
    t.transaction_id,
    t.amount,
    t.created_at
FROM users u
INNER JOIN transactions t ON u.user_id = t.user_id
WHERE t.created_at >= '2026-01-01';

-- ── LEFT JOIN: all users, even those with no transactions ───
-- Critical for ML: users with 0 transactions are important!
SELECT
    u.user_id,
    u.signup_date,
    COUNT(t.transaction_id) AS txn_count,
    COALESCE(SUM(t.amount), 0) AS total_amount
FROM users u
LEFT JOIN transactions t ON u.user_id = t.user_id
GROUP BY u.user_id, u.signup_date;

-- ── SELF JOIN: compare rows within the same table ───────────
-- Find users who had transactions within 1 minute of each other (velocity)
SELECT
    t1.user_id,
    t1.transaction_id AS txn1,
    t2.transaction_id AS txn2,
    t1.created_at AS time1,
    t2.created_at AS time2
FROM transactions t1
INNER JOIN transactions t2
    ON t1.user_id = t2.user_id
    AND t1.transaction_id < t2.transaction_id
    AND t2.created_at BETWEEN t1.created_at AND t1.created_at + INTERVAL '1 minute';
```

### Aggregations and GROUP BY

```sql
-- ── User-level features for ML ──────────────────────────────
SELECT
    user_id,
    COUNT(*) AS total_orders,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order_value,
    MIN(amount) AS min_order,
    MAX(amount) AS max_order,
    STDDEV(amount) AS std_order,
    COUNT(DISTINCT product_category) AS unique_categories,
    MIN(created_at) AS first_order_date,
    MAX(created_at) AS last_order_date,
    MAX(created_at) - MIN(created_at) AS customer_lifetime,
    COUNT(*) FILTER (WHERE status = 'returned') AS return_count
FROM orders
GROUP BY user_id
HAVING COUNT(*) >= 2;  -- only users with 2+ orders
```

### CTEs (Common Table Expressions)

```sql
-- ── CTEs make complex queries readable ──────────────────────
-- Compute RFM (Recency, Frequency, Monetary) features

WITH recency AS (
    SELECT
        user_id,
        CURRENT_DATE - MAX(created_at)::date AS days_since_last_order
    FROM orders
    GROUP BY user_id
),
frequency AS (
    SELECT
        user_id,
        COUNT(*) AS order_count,
        COUNT(*) / GREATEST(
            EXTRACT(MONTH FROM AGE(MAX(created_at), MIN(created_at))), 1
        ) AS orders_per_month
    FROM orders
    GROUP BY user_id
),
monetary AS (
    SELECT
        user_id,
        SUM(amount) AS total_spend,
        AVG(amount) AS avg_spend
    FROM orders
    GROUP BY user_id
)
SELECT
    r.user_id,
    r.days_since_last_order,
    f.order_count,
    f.orders_per_month,
    m.total_spend,
    m.avg_spend
FROM recency r
JOIN frequency f ON r.user_id = f.user_id
JOIN monetary m ON r.user_id = m.user_id;
```

---

## 3. SQL Window Functions — The ML Engineer's Power Tool

Window functions operate on a set of rows related to the current row **without collapsing them** (unlike GROUP BY). Essential for time-series features, ranking, and running aggregates.

$$
f(x_i) = \text{aggregate}\left(\{x_j : j \in \text{window}(i)\}\right)
$$

### Syntax

```sql
-- General form:
-- function() OVER (PARTITION BY col ORDER BY col ROWS/RANGE frame)

-- ── ROW_NUMBER, RANK, DENSE_RANK ────────────────────────────

-- Rank products by sales within each category
SELECT
    product_id,
    category,
    total_sales,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY total_sales DESC) AS rank_in_category,
    RANK() OVER (PARTITION BY category ORDER BY total_sales DESC) AS rank_with_gaps,
    DENSE_RANK() OVER (PARTITION BY category ORDER BY total_sales DESC) AS rank_no_gaps
FROM product_sales;

-- ── Get the most recent order per user (common interview Q) ─
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
    FROM orders
) ranked
WHERE rn = 1;
```

### Running Aggregates

```sql
-- ── Rolling averages (crucial for ML features) ──────────────

-- 7-day rolling average spend per user
SELECT
    user_id,
    order_date,
    amount,
    AVG(amount) OVER (
        PARTITION BY user_id
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg,
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_sum,
    COUNT(*) OVER (
        PARTITION BY user_id
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_count
FROM daily_user_spend;

-- ── Cumulative sum (running total) ──────────────────────────
SELECT
    user_id,
    order_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_spend
FROM orders;
```

### LAG and LEAD

```sql
-- ── Time between events (inter-event time) ──────────────────

SELECT
    user_id,
    created_at,
    amount,
    LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at) AS prev_order_time,
    created_at - LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at) AS time_since_prev,
    LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS prev_amount,
    amount - LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS amount_change,
    LEAD(created_at) OVER (PARTITION BY user_id ORDER BY created_at) AS next_order_time
FROM orders;

-- ── Percent change from previous period ─────────────────────
SELECT
    date,
    revenue,
    LAG(revenue) OVER (ORDER BY date) AS prev_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY date))
        / NULLIF(LAG(revenue) OVER (ORDER BY date), 0) * 100, 2
    ) AS pct_change
FROM daily_revenue;
```

### NTILE and Percentiles

```sql
-- ── Bucket users into deciles by spend ──────────────────────
SELECT
    user_id,
    total_spend,
    NTILE(10) OVER (ORDER BY total_spend) AS spend_decile,
    PERCENT_RANK() OVER (ORDER BY total_spend) AS spend_percentile
FROM user_stats;
```

---

## 4. SQL Feature Extraction Queries

Real-world feature queries for ML training data.

```sql
-- ══════════════════════════════════════════════════════════════
-- FEATURE EXTRACTION: Fraud Detection Training Data
-- ══════════════════════════════════════════════════════════════

WITH user_history AS (
    -- User-level aggregates (computed once, reused)
    SELECT
        user_id,
        COUNT(*) AS lifetime_txn_count,
        AVG(amount) AS lifetime_avg_amount,
        STDDEV(amount) AS lifetime_std_amount,
        MIN(created_at) AS first_txn_date,
        MAX(created_at) AS last_txn_date
    FROM transactions
    WHERE created_at < '2026-01-01'  -- point-in-time: no data leakage!
    GROUP BY user_id
),
velocity_features AS (
    -- Per-transaction velocity features using window functions
    SELECT
        t.transaction_id,
        t.user_id,
        t.amount,
        t.created_at,
        -- Transactions in last 1 hour
        COUNT(*) OVER (
            PARTITION BY t.user_id
            ORDER BY t.created_at
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) - 1 AS txn_count_1h,
        -- Total amount in last 1 hour
        SUM(t.amount) OVER (
            PARTITION BY t.user_id
            ORDER BY t.created_at
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) - t.amount AS amount_sum_1h,
        -- Transactions in last 24 hours
        COUNT(*) OVER (
            PARTITION BY t.user_id
            ORDER BY t.created_at
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) - 1 AS txn_count_24h,
        -- Time since last transaction
        t.created_at - LAG(t.created_at) OVER (
            PARTITION BY t.user_id ORDER BY t.created_at
        ) AS time_since_prev_txn,
        -- Amount difference from previous
        t.amount - LAG(t.amount) OVER (
            PARTITION BY t.user_id ORDER BY t.created_at
        ) AS amount_diff_prev
    FROM transactions t
)
SELECT
    v.transaction_id,
    v.user_id,
    v.amount,
    v.created_at,

    -- User features
    uh.lifetime_txn_count,
    uh.lifetime_avg_amount,
    uh.lifetime_std_amount,
    EXTRACT(DAY FROM v.created_at - uh.first_txn_date) AS account_age_days,

    -- Transaction features
    v.amount / NULLIF(uh.lifetime_avg_amount, 0) AS amount_to_avg_ratio,
    (v.amount - uh.lifetime_avg_amount) / NULLIF(uh.lifetime_std_amount, 0) AS amount_zscore,

    -- Velocity features
    v.txn_count_1h,
    v.amount_sum_1h,
    v.txn_count_24h,
    EXTRACT(EPOCH FROM v.time_since_prev_txn) / 60 AS minutes_since_prev_txn,
    v.amount_diff_prev,

    -- Time features
    EXTRACT(HOUR FROM v.created_at) AS hour_of_day,
    EXTRACT(DOW FROM v.created_at) AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM v.created_at) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,

    -- Label (from chargeback data)
    COALESCE(cb.is_fraud, 0) AS label

FROM velocity_features v
LEFT JOIN user_history uh ON v.user_id = uh.user_id
LEFT JOIN chargebacks cb ON v.transaction_id = cb.transaction_id
WHERE v.created_at BETWEEN '2026-01-01' AND '2026-01-31';
```

---

## 5. PostgreSQL — Setup and ML-Relevant Features

```bash
# Docker setup
docker run -d --name postgres \
    -e POSTGRES_PASSWORD=mlpassword \
    -e POSTGRES_DB=mldb \
    -p 5432:5432 \
    postgres:16
```

```python
"""
PostgreSQL with Python — patterns for ML data pipelines.
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from contextlib import contextmanager


# ── Connection manager ────────────────────────────────────────

@contextmanager
def get_connection(
    host="localhost", port=5432, dbname="mldb",
    user="postgres", password="mlpassword",
):
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password,
    )
    try:
        yield conn
    finally:
        conn.close()


# ── Create tables for ML data ────────────────────────────────

def create_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    signup_date TIMESTAMP NOT NULL,
                    country TEXT,
                    age INT
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    amount NUMERIC(12,2) NOT NULL,
                    category TEXT,
                    status TEXT DEFAULT 'completed',
                    created_at TIMESTAMP NOT NULL
                );

                -- Index for fast time-range queries (ML features)
                CREATE INDEX IF NOT EXISTS idx_txn_user_time
                ON transactions(user_id, created_at);

                -- Index for fast aggregation queries
                CREATE INDEX IF NOT EXISTS idx_txn_created
                ON transactions(created_at);

                -- Predictions table (store model outputs)
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    model_version TEXT NOT NULL,
                    score NUMERIC(6,4) NOT NULL,
                    features JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        conn.commit()


# ── Bulk insert (fast — using execute_values) ────────────────

def bulk_insert_transactions(transactions: list[dict]):
    """Insert many transactions efficiently."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            values = [
                (t["transaction_id"], t["user_id"], t["amount"],
                 t["category"], t["status"], t["created_at"])
                for t in transactions
            ]
            execute_values(
                cur,
                """INSERT INTO transactions
                   (transaction_id, user_id, amount, category, status, created_at)
                   VALUES %s
                   ON CONFLICT (transaction_id) DO NOTHING""",
                values,
                page_size=1000,
            )
        conn.commit()
        print(f"Inserted {len(transactions)} transactions")


# ── Read into Pandas (most common ML pattern) ────────────────

def query_to_dataframe(query: str, params: dict = None) -> pd.DataFrame:
    """Execute SQL and return a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params)


# Usage
# df = query_to_dataframe("""
#     SELECT user_id, COUNT(*) as txn_count, AVG(amount) as avg_amount
#     FROM transactions
#     WHERE created_at >= %(start_date)s
#     GROUP BY user_id
# """, params={"start_date": "2026-01-01"})


# ── Store predictions back to DB ──────────────────────────────

def store_predictions(predictions: list[dict]):
    """Store model predictions for monitoring and analysis."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            values = [
                (p["user_id"], p["model_version"], p["score"],
                 psycopg2.extras.Json(p.get("features", {})))
                for p in predictions
            ]
            execute_values(
                cur,
                """INSERT INTO predictions (user_id, model_version, score, features)
                   VALUES %s""",
                values,
            )
        conn.commit()


# ── PostgreSQL-specific ML features ──────────────────────────

PGVECTOR_SETUP = """
-- pgvector: native vector similarity search in PostgreSQL
-- Install: CREATE EXTENSION vector;

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)  -- OpenAI embedding dimension
);

-- Create HNSW index for fast similarity search
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Similarity search
SELECT id, content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM embeddings
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
"""
```

---

## 6. NoSQL — MongoDB for Document Storage

```python
"""
MongoDB — flexible document storage for ML metadata, configs, and logs.
"""

from pymongo import MongoClient
from datetime import datetime


# ── Connect ───────────────────────────────────────────────────

client = MongoClient("mongodb://localhost:27017/")
db = client["ml_platform"]


# ── Store experiment metadata ─────────────────────────────────

experiments = db["experiments"]

experiment = {
    "name": "fraud_detector_v3",
    "model_type": "XGBoost",
    "created_at": datetime.now(),
    "config": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 7,
        "features": ["amount", "velocity_1h", "account_age", "device_fingerprint"],
    },
    "metrics": {
        "auc": 0.9534,
        "precision_at_1pct_fpr": 0.72,
        "f1": 0.89,
    },
    "dataset": {
        "name": "fraud_jan_2026",
        "size": 1_500_000,
        "positive_rate": 0.008,
    },
    "tags": ["production-candidate", "fraud"],
}

result = experiments.insert_one(experiment)
print(f"Inserted: {result.inserted_id}")


# ── Query experiments ─────────────────────────────────────────

# Find best fraud detectors
best = experiments.find(
    {"tags": "fraud", "metrics.auc": {"$gt": 0.95}},
    sort=[("metrics.auc", -1)],
    limit=5,
)
for exp in best:
    print(f"{exp['name']}: AUC={exp['metrics']['auc']}")

# Find experiments with specific config
xgboost_exps = experiments.find({
    "model_type": "XGBoost",
    "config.n_estimators": {"$gte": 200},
})


# ── Store model predictions with flexible schema ─────────────

predictions_col = db["predictions"]

# Different models can have different output formats
predictions_col.insert_many([
    {
        "model": "fraud_v3",
        "user_id": "u123",
        "score": 0.87,
        "features_used": {"amount": 999.99, "velocity_1h": 5},
        "timestamp": datetime.now(),
    },
    {
        "model": "recommender_v2",
        "user_id": "u456",
        "recommendations": ["item_1", "item_2", "item_3"],
        "scores": [0.95, 0.87, 0.82],
        "timestamp": datetime.now(),
    },
])


# ── Aggregation pipeline (like SQL GROUP BY) ─────────────────

# Average AUC by model type
pipeline = [
    {"$group": {
        "_id": "$model_type",
        "avg_auc": {"$avg": "$metrics.auc"},
        "count": {"$sum": 1},
        "best_auc": {"$max": "$metrics.auc"},
    }},
    {"$sort": {"avg_auc": -1}},
]

for doc in experiments.aggregate(pipeline):
    print(f"{doc['_id']}: avg_auc={doc['avg_auc']:.4f} ({doc['count']} experiments)")
```

---

## 7. Redis — Caching and Feature Serving

Redis is the go-to for **low-latency** data access. Sub-millisecond reads make it perfect for ML serving.

```python
"""
Redis for ML: caching predictions, serving features, rate limiting.
"""

import redis
import json
import time
import hashlib
from typing import Any


# ── Connect ───────────────────────────────────────────────────

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Test connection
r.ping()  # True


# ── 1. Prediction caching ────────────────────────────────────

class PredictionCache:
    """
    Cache model predictions to avoid redundant inference.

    Cache hit → <1ms response
    Cache miss → compute prediction (~50ms) → cache it → return
    """

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 3600):
        self.r = redis_client
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(self, model_version: str, features: dict) -> str:
        """Deterministic cache key from model + features."""
        content = json.dumps({"model": model_version, "features": features}, sort_keys=True)
        return f"pred:{hashlib.md5(content.encode()).hexdigest()}"

    def get(self, model_version: str, features: dict) -> float | None:
        """Check cache for an existing prediction."""
        key = self._make_key(model_version, features)
        cached = self.r.get(key)
        if cached is not None:
            self.hits += 1
            return float(cached)
        self.misses += 1
        return None

    def set(self, model_version: str, features: dict, score: float):
        """Cache a prediction result."""
        key = self._make_key(model_version, features)
        self.r.setex(key, self.ttl, str(score))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Usage in serving API:
# cache = PredictionCache(r)
# cached_score = cache.get("v2.1", features)
# if cached_score is not None:
#     return cached_score  # <1ms
# score = model.predict(features)  # ~50ms
# cache.set("v2.1", features, score)
# return score


# ── 2. Session/user data for real-time features ──────────────

def update_user_session(user_id: str, event: dict):
    """Track user session data in Redis for real-time features."""
    key = f"session:{user_id}"
    pipe = r.pipeline()

    # Increment event count
    pipe.hincrby(key, "event_count", 1)
    pipe.hincrby(key, f"event_{event['type']}_count", 1)

    # Update last event time
    pipe.hset(key, "last_event_time", event["timestamp"])

    # Add to recent events list (keep last 50)
    pipe.lpush(f"events:{user_id}", json.dumps(event))
    pipe.ltrim(f"events:{user_id}", 0, 49)

    # Auto-expire session after 30 minutes
    pipe.expire(key, 1800)
    pipe.expire(f"events:{user_id}", 1800)

    pipe.execute()


def get_realtime_features(user_id: str) -> dict:
    """Get real-time session features for ML inference."""
    key = f"session:{user_id}"
    session = r.hgetall(key)

    if not session:
        return {"event_count": 0, "has_session": False}

    return {
        "has_session": True,
        "event_count": int(session.get("event_count", 0)),
        "click_count": int(session.get("event_click_count", 0)),
        "view_count": int(session.get("event_view_count", 0)),
        "last_event_time": session.get("last_event_time", ""),
    }


# ── 3. Rate limiting (for ML API protection) ─────────────────

def is_rate_limited(client_id: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
    """Sliding window rate limiter using Redis sorted sets."""
    key = f"ratelimit:{client_id}"
    now = time.time()
    window_start = now - window_seconds

    pipe = r.pipeline()
    # Remove old entries
    pipe.zremrangebyscore(key, 0, window_start)
    # Add current request
    pipe.zadd(key, {str(now): now})
    # Count requests in window
    pipe.zcard(key)
    # Set expiry on the key
    pipe.expire(key, window_seconds)
    results = pipe.execute()

    request_count = results[2]
    return request_count > max_requests


# ── 4. Pub/Sub for real-time ML events ───────────────────────

def publish_prediction_event(prediction: dict):
    """Publish prediction events for downstream consumers (monitoring, logging)."""
    r.publish("ml:predictions", json.dumps(prediction))


def subscribe_to_predictions():
    """Subscribe to prediction events (for monitoring service)."""
    pubsub = r.pubsub()
    pubsub.subscribe("ml:predictions")

    for message in pubsub.listen():
        if message["type"] == "message":
            prediction = json.loads(message["data"])
            print(f"Received: {prediction}")
```

---

## 8. Redis as a Feature Store for ML

```python
"""
Redis-based feature store: serve pre-computed features at low latency.
"""

import redis
import json
import time
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    name: str
    entity: str  # "user", "item"
    ttl_seconds: int = 86400  # 24h default


class RedisFeatureStore:
    """
    Feature store backed by Redis.

    Write path:  batch pipeline → compute features → write to Redis
    Read path:   serving API → get_features() → Redis → <1ms

    Key schema: feature:{entity}:{entity_id}
    Value: JSON dict of feature_name → value
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.configs: dict[str, FeatureConfig] = {}

    def register(self, config: FeatureConfig):
        self.configs[f"{config.entity}:{config.name}"] = config

    def write_features(self, entity: str, entity_id: str, features: dict[str, float], ttl: int = 86400):
        """Write features for a single entity (called by batch pipeline)."""
        key = f"feature:{entity}:{entity_id}"
        pipe = self.r.pipeline()
        pipe.hset(key, mapping={k: json.dumps(v) for k, v in features.items()})
        pipe.expire(key, ttl)
        pipe.execute()

    def write_features_bulk(self, entity: str, feature_data: dict[str, dict[str, float]], ttl: int = 86400):
        """Bulk write features for many entities (end of batch pipeline)."""
        pipe = self.r.pipeline()
        for entity_id, features in feature_data.items():
            key = f"feature:{entity}:{entity_id}"
            pipe.hset(key, mapping={k: json.dumps(v) for k, v in features.items()})
            pipe.expire(key, ttl)
        pipe.execute()
        print(f"Wrote features for {len(feature_data)} {entity}s")

    def get_features(self, entity: str, entity_id: str, feature_names: list[str]) -> dict[str, float | None]:
        """Get features for a single entity (called by serving API)."""
        key = f"feature:{entity}:{entity_id}"
        values = self.r.hmget(key, feature_names)
        return {
            name: json.loads(v) if v is not None else None
            for name, v in zip(feature_names, values)
        }

    def get_features_multi(
        self,
        entity: str,
        entity_ids: list[str],
        feature_names: list[str],
    ) -> dict[str, dict[str, float | None]]:
        """Get features for multiple entities in one round-trip."""
        pipe = self.r.pipeline()
        for eid in entity_ids:
            pipe.hmget(f"feature:{entity}:{eid}", feature_names)
        results = pipe.execute()

        return {
            eid: {
                name: json.loads(v) if v is not None else None
                for name, v in zip(feature_names, values)
            }
            for eid, values in zip(entity_ids, results)
        }

    def get_freshness(self, entity: str, entity_id: str) -> int:
        """Check TTL remaining (seconds). -1 = no expiry, -2 = key missing."""
        key = f"feature:{entity}:{entity_id}"
        return self.r.ttl(key)


# ── Usage ─────────────────────────────────────────────────────

store = RedisFeatureStore()

# Batch pipeline writes features (runs nightly)
store.write_features_bulk("user", {
    "u001": {"total_purchases": 42, "avg_order_value": 55.0, "days_since_signup": 365},
    "u002": {"total_purchases": 7, "avg_order_value": 120.0, "days_since_signup": 30},
    "u003": {"total_purchases": 150, "avg_order_value": 25.0, "days_since_signup": 1200},
})

# Serving API reads features (<1ms)
features = store.get_features("user", "u001", ["total_purchases", "avg_order_value"])
print(features)
# {'total_purchases': 42, 'avg_order_value': 55.0}

# Batch read for recommendations (multiple users at once)
batch = store.get_features_multi(
    "user", ["u001", "u002", "u003"],
    ["total_purchases", "avg_order_value"],
)
print(batch)
```

---

## 9. Vector Databases — Concepts

Vector databases store **embeddings** (high-dimensional floating-point vectors) and support **similarity search** — finding the nearest neighbors to a query vector.

$$
\text{similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{|\mathbf{q}| \cdot |\mathbf{d}|} \quad \text{(cosine similarity)}
$$

```
┌──────────────────────────────────────────────────────────────┐
│                   Vector DB Use Cases                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. RAG: Find relevant document chunks for LLM context       │
│  2. Semantic search: "Find similar products" by embedding    │
│  3. Recommendations: "Users like you also liked..."          │
│  4. Deduplication: Find near-duplicate content               │
│  5. Anomaly detection: Find points far from any cluster      │
│  6. Image search: Find visually similar images               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

| Vector DB | Type | Best For | Scale |
|-----------|------|----------|-------|
| **FAISS** | Library (in-process) | Research, prototyping, single-machine | Billions (single node) |
| **ChromaDB** | Embedded DB | Prototyping, small apps | Thousands-millions |
| **Pinecone** | Managed cloud | Production, no ops | Billions |
| **Qdrant** | Self-hosted / cloud | Production, filtering | Billions |
| **Weaviate** | Self-hosted / cloud | Multi-modal, GraphQL | Millions-billions |
| **pgvector** | PostgreSQL extension | Already using Postgres | Millions |

---

## 10. FAISS — Fast Similarity Search

```python
"""
FAISS (Facebook AI Similarity Search) — the fastest vector search library.
Best for: research, prototyping, single-machine workloads.
"""

import numpy as np
import faiss
import time


# ── Basic flat index (exact, brute-force) ─────────────────────

dimension = 384  # typical sentence-transformer embedding size
n_vectors = 100_000

# Generate random vectors (in practice: embeddings from a model)
np.random.seed(42)
vectors = np.random.randn(n_vectors, dimension).astype("float32")
faiss.normalize_L2(vectors)  # normalize for cosine similarity

# Build index
index = faiss.IndexFlatIP(dimension)  # IP = inner product (= cosine after normalization)
index.add(vectors)
print(f"Index size: {index.ntotal} vectors")

# Search
query = np.random.randn(1, dimension).astype("float32")
faiss.normalize_L2(query)

start = time.time()
scores, indices = index.search(query, k=10)
print(f"Flat search: {(time.time()-start)*1000:.2f}ms")
print(f"Top 10 indices: {indices[0]}")
print(f"Top 10 scores: {scores[0]}")


# ── IVF index (approximate, much faster for large datasets) ──

n_lists = 100  # number of Voronoi cells
quantizer = faiss.IndexFlatIP(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, n_lists, faiss.METRIC_INNER_PRODUCT)

# Must train on representative data
index_ivf.train(vectors)
index_ivf.add(vectors)

# nprobe: how many cells to search (trade-off: speed vs accuracy)
index_ivf.nprobe = 10

start = time.time()
scores, indices = index_ivf.search(query, k=10)
print(f"\nIVF search: {(time.time()-start)*1000:.2f}ms")


# ── HNSW index (best recall for approximate search) ──────────

index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M (neighbors per node)
index_hnsw.hnsw.efConstruction = 200  # construction quality
index_hnsw.hnsw.efSearch = 64  # search quality

index_hnsw.add(vectors)

start = time.time()
scores, indices = index_hnsw.search(query, k=10)
print(f"\nHNSW search: {(time.time()-start)*1000:.2f}ms")


# ── Save and load index ──────────────────────────────────────

faiss.write_index(index_hnsw, "vectors.index")
loaded_index = faiss.read_index("vectors.index")


# ── GPU acceleration ──────────────────────────────────────────

GPU_EXAMPLE = """
# Move index to GPU for massive speedup
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
scores, indices = gpu_index.search(query, k=10)
"""
```

---

## 11. ChromaDB — Embedded Vector Database

```python
"""
ChromaDB — the simplest vector database. Great for prototyping and small apps.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ── Setup ─────────────────────────────────────────────────────

# Persistent storage (survives restarts)
client = chromadb.PersistentClient(path="./chroma_data")

# Use a sentence transformer for embeddings
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get collection
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # cosine similarity
)


# ── Add documents ─────────────────────────────────────────────

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by biological neurons.",
    "Gradient descent is an optimization algorithm used in training.",
    "Transfer learning allows using pre-trained models on new tasks.",
    "BERT is a bidirectional encoder representation from transformers.",
    "Random forests are ensemble methods using multiple decision trees.",
    "Convolutional neural networks excel at image recognition tasks.",
    "Recurrent neural networks are designed for sequential data.",
    "Transformers use self-attention mechanisms for parallel processing.",
    "Reinforcement learning agents learn through trial and error.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[
        {"category": "ml_basics", "difficulty": "beginner"},
        {"category": "deep_learning", "difficulty": "intermediate"},
        {"category": "optimization", "difficulty": "intermediate"},
        {"category": "transfer_learning", "difficulty": "intermediate"},
        {"category": "nlp", "difficulty": "advanced"},
        {"category": "ml_basics", "difficulty": "beginner"},
        {"category": "computer_vision", "difficulty": "intermediate"},
        {"category": "deep_learning", "difficulty": "intermediate"},
        {"category": "nlp", "difficulty": "advanced"},
        {"category": "reinforcement_learning", "difficulty": "advanced"},
    ],
)

print(f"Collection has {collection.count()} documents")


# ── Query (semantic search) ───────────────────────────────────

results = collection.query(
    query_texts=["How do transformers work?"],
    n_results=3,
)
print("\nQuery: 'How do transformers work?'")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  [{dist:.4f}] {doc}")


# ── Query with metadata filter ────────────────────────────────

results = collection.query(
    query_texts=["What are the basics?"],
    n_results=3,
    where={"difficulty": "beginner"},  # filter by metadata
)
print("\nQuery: 'What are the basics?' (beginner only)")
for doc in results["documents"][0]:
    print(f"  {doc}")


# ── Query with combined filter ────────────────────────────────

results = collection.query(
    query_texts=["neural network architectures"],
    n_results=5,
    where={"$and": [
        {"category": {"$in": ["deep_learning", "nlp"]}},
        {"difficulty": {"$ne": "beginner"}},
    ]},
)


# ── Update and delete ────────────────────────────────────────

collection.update(
    ids=["doc_0"],
    documents=["Machine learning is a branch of AI that enables systems to learn from data."],
    metadatas=[{"category": "ml_basics", "difficulty": "beginner", "updated": True}],
)

collection.delete(ids=["doc_9"])
print(f"\nAfter delete: {collection.count()} documents")
```

---

## 12. Pinecone — Managed Vector Database

```python
"""
Pinecone — fully managed vector database. No infra to manage.
"""

PINECONE_CODE = '''
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="product-search",
    dimension=1536,  # OpenAI embedding size
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index("product-search")

# Upsert vectors with metadata
index.upsert(vectors=[
    {
        "id": "prod_001",
        "values": [0.1, 0.2, ...],  # 1536-dim embedding
        "metadata": {
            "category": "electronics",
            "price": 299.99,
            "in_stock": True,
            "title": "Wireless Headphones",
        },
    },
    # ... more vectors
])

# Query with metadata filter
results = index.query(
    vector=[0.15, 0.25, ...],  # query embedding
    top_k=10,
    include_metadata=True,
    filter={
        "category": {"$eq": "electronics"},
        "price": {"$lte": 500},
        "in_stock": {"$eq": True},
    },
)

for match in results["matches"]:
    print(f"Score: {match['score']:.4f} | {match['metadata']['title']}")

# Namespace support (multi-tenant)
index.upsert(vectors=[...], namespace="customer_123")
results = index.query(vector=[...], top_k=10, namespace="customer_123")

# Stats
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
'''
```

---

## 13. Qdrant — Production Vector Search

```python
"""
Qdrant — high-performance vector search with rich filtering.
Self-hosted (Docker) or cloud managed.
"""

# Docker: docker run -p 6333:6333 qdrant/qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
)
import numpy as np


# ── Connect ───────────────────────────────────────────────────

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.recreate_collection(
    collection_name="articles",
    vectors_config=VectorParams(
        size=384,  # embedding dimension
        distance=Distance.COSINE,
    ),
)


# ── Insert vectors ────────────────────────────────────────────

np.random.seed(42)
points = [
    PointStruct(
        id=i,
        vector=np.random.randn(384).tolist(),
        payload={
            "title": f"Article {i}",
            "category": np.random.choice(["ml", "nlp", "cv", "rl"]),
            "year": np.random.choice([2024, 2025, 2026]),
            "citations": int(np.random.exponential(50)),
        },
    )
    for i in range(1000)
]

client.upsert(collection_name="articles", points=points)
print(f"Inserted {len(points)} vectors")


# ── Search ────────────────────────────────────────────────────

query_vector = np.random.randn(384).tolist()

# Basic search
results = client.search(
    collection_name="articles",
    query_vector=query_vector,
    limit=5,
)
for r in results:
    print(f"Score: {r.score:.4f} | {r.payload['title']} ({r.payload['category']})")


# ── Filtered search ──────────────────────────────────────────

results = client.search(
    collection_name="articles",
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="nlp")),
            FieldCondition(key="year", range=Range(gte=2025)),
            FieldCondition(key="citations", range=Range(gte=20)),
        ]
    ),
    limit=5,
)
print("\nFiltered (NLP, 2025+, 20+ citations):")
for r in results:
    print(f"  {r.payload['title']} | citations={r.payload['citations']}")


# ── Batch search (multiple queries at once) ───────────────────

queries = [np.random.randn(384).tolist() for _ in range(3)]
batch_results = client.search_batch(
    collection_name="articles",
    requests=[
        {"vector": q, "limit": 3} for q in queries
    ],
)
```

---

## 14. Vector Indexing Algorithms — HNSW, IVF, PQ

```
┌──────────────────────────────────────────────────────────────┐
│           Vector Index Algorithms Compared                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  FLAT (brute force):                                         │
│  - Compare query to EVERY vector                             │
│  - 100% recall, O(n) time                                    │
│  - Fine for <100K vectors                                    │
│                                                              │
│  IVF (Inverted File Index):                                  │
│  - Cluster vectors into cells (Voronoi)                      │
│  - Search only nearby cells (nprobe)                         │
│  - Fast, but sensitive to nprobe setting                     │
│                                                              │
│  HNSW (Hierarchical Navigable Small World):                  │
│  - Build a multi-layer graph of neighbors                    │
│  - Navigate from coarse to fine layers                       │
│  - Best recall/speed trade-off, higher memory                │
│                                                              │
│  PQ (Product Quantization):                                  │
│  - Compress vectors by splitting into subvectors             │
│  - Each subvector quantized to a codebook entry              │
│  - 10-100x memory reduction, lossy                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### HNSW In-Depth

```
Layer 2 (sparse):    A -------- B
                     |
Layer 1 (medium):    A --- C --- B --- D
                     |    |     |    |
Layer 0 (dense):     A-E-C-F-G-B-H-D-I-J

Search: Start at top layer, greedily follow closest neighbors,
        drop to next layer, repeat until Layer 0.
```

Key parameters:
- $M$ = max connections per node (higher = better recall, more memory)
- $\text{efConstruction}$ = search width during build (higher = better index quality, slower build)
- $\text{efSearch}$ = search width during query (higher = better recall, slower query)

$$
\text{Memory} \approx n \times (d \times 4 + M \times 2 \times 8) \text{ bytes}
$$

Where $n$ = number of vectors, $d$ = dimension, $M$ = edges per node.

### Choosing an Index

| Scenario | Recommended Index |
|----------|------------------|
| < 100K vectors | Flat (exact) |
| 100K-10M, high recall needed | HNSW |
| 10M+, memory constrained | IVF-PQ |
| 10M+, high recall needed | IVF-HNSW |
| Billion scale | IVF-PQ with GPU |

---

## 15. Data Warehousing — BigQuery, Snowflake, Redshift

```
┌──────────────────────────────────────────────────────────────┐
│               Data Warehouses for ML                          │
├────────────────┬─────────────┬─────────────┬─────────────────┤
│ Feature        │ BigQuery    │ Snowflake   │ Redshift        │
├────────────────┼─────────────┼─────────────┼─────────────────┤
│ Cloud          │ GCP         │ Multi-cloud │ AWS             │
│ Pricing        │ Per-query   │ Per-compute │ Per-cluster     │
│ ML integration │ BigQuery ML │ Snowpark    │ SageMaker       │
│ Scaling        │ Serverless  │ Auto-scale  │ Manual resize   │
│ Best for       │ GCP shops   │ Multi-cloud │ AWS-heavy shops │
│ SQL dialect    │ Standard    │ ANSI + ext  │ PostgreSQL-like │
└────────────────┴─────────────┴─────────────┴─────────────────┘
```

```python
"""
BigQuery for ML feature extraction (pattern — requires GCP credentials).
"""

BIGQUERY_PATTERN = '''
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="my-project")

# Extract ML training features
query = """
SELECT
    user_id,
    COUNT(*) AS order_count,
    AVG(amount) AS avg_amount,
    STDDEV(amount) AS std_amount,
    DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) AS recency_days,
    COUNTIF(status = 'returned') / COUNT(*) AS return_rate
FROM `my-project.analytics.orders`
WHERE order_date BETWEEN '2025-01-01' AND '2026-01-01'
GROUP BY user_id
HAVING COUNT(*) >= 5
"""

# Run query → DataFrame (BigQuery handles TB-scale data)
df = client.query(query).to_dataframe()
print(f"Extracted features for {len(df)} users")

# Save to GCS for model training
df.to_parquet("gs://my-bucket/features/user_features.parquet")
'''

# BigQuery ML — train models directly in SQL
BQML_EXAMPLE = """
-- Train a logistic regression in BigQuery (no Python needed!)
CREATE OR REPLACE MODEL `my_project.ml.churn_predictor`
OPTIONS(
    model_type='LOGISTIC_REG',
    input_label_cols=['churned'],
    auto_class_weights=TRUE
) AS
SELECT
    user_id,
    order_count,
    avg_amount,
    recency_days,
    return_rate,
    churned
FROM `my_project.features.user_features`;

-- Predict
SELECT *
FROM ML.PREDICT(MODEL `my_project.ml.churn_predictor`,
    (SELECT * FROM `my_project.features.new_users`)
);
"""
```

---

## 16. Streaming — Kafka for Real-Time ML Pipelines

```
┌──────────────────────────────────────────────────────────────┐
│              Kafka for Real-Time ML                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Producer         Kafka Broker        Consumer               │
│  (app events) ──→ [topic: events] ──→ (feature pipeline)    │
│                   [topic: events] ──→ (monitoring pipeline)  │
│                   [topic: events] ──→ (analytics pipeline)   │
│                                                              │
│  Key concepts:                                               │
│  - Topics: named channels (e.g., "user_events")             │
│  - Partitions: parallelism within a topic                    │
│  - Consumer groups: each group gets every message once       │
│  - Offset: position in the partition (enables replay)        │
│                                                              │
│  ML use cases:                                               │
│  1. Real-time feature computation (event → feature store)    │
│  2. Model prediction logging (serve → Kafka → warehouse)     │
│  3. Event-driven retraining triggers                         │
│  4. Real-time anomaly detection                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

```python
"""
Kafka producer and consumer for ML event pipelines.
"""

# ── Producer: emit ML events ──────────────────────────────────

KAFKA_PRODUCER = '''
from kafka import KafkaProducer
import json
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: k.encode("utf-8") if k else None,
)

# Emit a user event
def emit_event(user_id: str, event_type: str, data: dict):
    event = {
        "user_id": user_id,
        "event_type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    producer.send(
        topic="user_events",
        key=user_id,  # ensures all events for a user go to same partition
        value=event,
    )

# Example: user makes a purchase
emit_event("u123", "purchase", {"amount": 49.99, "product": "headphones"})
emit_event("u123", "page_view", {"page": "/checkout"})

producer.flush()
'''


# ── Consumer: real-time feature computation ───────────────────

KAFKA_CONSUMER = '''
from kafka import KafkaConsumer
import json
import redis

consumer = KafkaConsumer(
    "user_events",
    bootstrap_servers=["localhost:9092"],
    group_id="feature_pipeline",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest",
)

r = redis.Redis()

def update_features(event):
    """Update real-time features in Redis based on Kafka event."""
    user_id = event["user_id"]
    key = f"realtime:{user_id}"

    if event["event_type"] == "purchase":
        r.hincrby(key, "purchase_count_1h", 1)
        r.hincrbyfloat(key, "spend_1h", event["data"]["amount"])

    elif event["event_type"] == "page_view":
        r.hincrby(key, "pageview_count_1h", 1)

    r.expire(key, 3600)  # features expire after 1 hour

# Process events
for message in consumer:
    update_features(message.value)
'''
```

---

## 17. Choosing the Right Database

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Database Decision Tree for ML                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  What are you storing?                                              │
│  │                                                                  │
│  ├── Structured data (tables, columns)                              │
│  │   ├── < 10GB, simple queries → SQLite                           │
│  │   ├── OLTP (app backend) → PostgreSQL / MySQL                   │
│  │   └── OLAP (analytics, ML features) → BigQuery / Snowflake     │
│  │                                                                  │
│  ├── Embeddings / vectors                                           │
│  │   ├── Prototyping → ChromaDB (embedded)                         │
│  │   ├── Already using PostgreSQL → pgvector                       │
│  │   ├── Managed, no ops → Pinecone                                │
│  │   └── Self-hosted, production → Qdrant or Weaviate              │
│  │                                                                  │
│  ├── Key-value (features, cache)                                    │
│  │   └── Always → Redis (nothing else comes close on latency)      │
│  │                                                                  │
│  ├── Documents (flexible schema)                                    │
│  │   └── MongoDB (experiment metadata, configs, logs)              │
│  │                                                                  │
│  └── Event streams                                                  │
│      └── Kafka (real-time features, event-driven ML)               │
│                                                                     │
│  In practice, most ML systems use 2-3 databases:                    │
│  PostgreSQL (app data) + Redis (features) + Qdrant (vectors)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 18. Practice Exercises

- [ ] **Exercise 1**: Write SQL queries using window functions to compute: (a) rolling 7-day average, (b) rank within partition, (c) time since previous event using LAG. Use any dataset.

- [ ] **Exercise 2**: Write the fraud detection feature extraction query (Section 4) and run it against a PostgreSQL database with synthetic data.

- [ ] **Exercise 3**: Set up Redis locally. Implement the `PredictionCache` class. Benchmark cache hit latency vs cache miss latency (should be <1ms vs ~50ms).

- [ ] **Exercise 4**: Build the `RedisFeatureStore`. Write features from a batch pipeline, then read them in a FastAPI serving endpoint. Measure read latency.

- [ ] **Exercise 5**: Set up ChromaDB. Index 100 documents with metadata. Write 5 queries: 2 basic semantic search, 3 with metadata filters. Verify relevance.

- [ ] **Exercise 6**: Set up FAISS. Index 1M random vectors. Compare search latency for Flat vs IVF vs HNSW indexes. Plot recall vs latency trade-off.

- [ ] **Exercise 7**: Solve 10 LeetCode SQL problems (Medium difficulty). Focus on: window functions, CTEs, self-joins, GROUP BY with HAVING.

- [ ] **Exercise 8**: Set up Qdrant with Docker. Index 10K vectors with rich metadata. Implement filtered search with at least 3 filter conditions.

---

## 19. Mini-Project: Data Pipeline — API → PostgreSQL → Features → Redis

Build a complete data pipeline that ingests data, computes features, and serves them at low latency.

```
┌──────────────────────────────────────────────────────────────┐
│        Mini-Project: ML Data Pipeline                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Architecture:                                               │
│  ┌────────────┐   ┌──────────┐   ┌────────────┐            │
│  │ Data Source │──→│PostgreSQL│──→│  Feature    │            │
│  │ (API/CSV)  │   │ (raw)    │   │  Pipeline   │            │
│  └────────────┘   └──────────┘   └─────┬──────┘            │
│                                        │                     │
│                                  ┌─────┴──────┐             │
│                                  │   Redis     │             │
│                                  │ (features)  │             │
│                                  └─────┬──────┘             │
│                                        │                     │
│                                  ┌─────┴──────┐             │
│                                  │  FastAPI    │             │
│                                  │ (serving)   │             │
│                                  └────────────┘             │
│                                                              │
│  Steps:                                                      │
│  1. Ingest: Load CSV or API data into PostgreSQL             │
│  2. Features: SQL query extracts user features               │
│  3. Materialize: Write features to Redis                     │
│  4. Serve: FastAPI reads features from Redis for prediction  │
│                                                              │
│  Deliverables:                                               │
│  ├── ingest.py (load data → PostgreSQL)                      │
│  ├── features.py (SQL feature extraction)                    │
│  ├── materialize.py (PostgreSQL → Redis)                     │
│  ├── serve.py (FastAPI + Redis feature reads)                │
│  ├── docker-compose.yml (PostgreSQL + Redis + API)           │
│  └── README.md                                               │
│                                                              │
│  Test:                                                       │
│  1. curl localhost:8000/features/user_123                     │
│     → Returns features in <5ms                               │
│  2. curl localhost:8000/predict -d '{"user_id": "user_123"}' │
│     → Returns prediction using cached features               │
│                                                              │
│  Stretch goals:                                              │
│  - Add Kafka for real-time event ingestion                   │
│  - Add ChromaDB for product semantic search                  │
│  - Add monitoring (Prometheus metrics on Redis hit rate)      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: mldb
      POSTGRES_PASSWORD: mlpassword
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 5

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://postgres:mlpassword@postgres:5432/mldb
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  pg_data:
```

---

## 20. Interview Questions & Answers

### Q1: Write a SQL query using window functions to compute rolling 7-day averages.

**Answer**:
```sql
SELECT
    user_id,
    order_date,
    amount,
    AVG(amount) OVER (
        PARTITION BY user_id
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg,
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_sum
FROM orders
ORDER BY user_id, order_date;
```

Use `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` for a 7-row window. Use `RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW` for a true calendar-day window (when rows may be missing for some days).

---

### Q2: Compare FAISS vs Pinecone vs ChromaDB. When would you use each?

**Answer**:

| Factor | FAISS | Pinecone | ChromaDB |
|--------|-------|----------|----------|
| Type | Library | Managed cloud | Embedded DB |
| Setup | `pip install faiss-cpu` | Sign up, get API key | `pip install chromadb` |
| Scale | Billions (single machine) | Billions (managed) | Thousands-millions |
| Filtering | No metadata filtering | Rich metadata filters | Basic metadata filters |
| Persistence | Manual (save/load) | Built-in | Built-in |
| Cost | Free | $0.096/1M reads + storage | Free |
| Operations | You manage everything | Zero ops | Zero ops (embedded) |

**FAISS**: Research, benchmarks, when you need raw speed and control, no metadata filtering needed.
**ChromaDB**: Prototyping, hackathons, small RAG apps, embedded use cases.
**Pinecone**: Production RAG at scale, when you want zero ops, rich filtering, and managed infrastructure.
**Qdrant** (not listed but important): Self-hosted production, when you need Pinecone-like features without vendor lock-in.

---

### Q3: How does HNSW indexing work at a high level? What are the trade-offs vs brute force?

**Answer**: HNSW builds a multi-layer graph:

1. **Bottom layer**: All vectors connected to their nearest neighbors (dense graph)
2. **Upper layers**: Progressively sparser — only a subset of vectors, connected across longer distances
3. **Search**: Start at the top layer, greedily navigate to the nearest node, drop to the next layer, repeat until the bottom

**Trade-offs vs brute force**:
- Speed: HNSW = $O(\log n)$ vs brute force = $O(n)$. 1000x faster at 1M vectors
- Recall: HNSW ≈ 95-99% recall (tunable with `efSearch`). Brute force = 100%
- Memory: HNSW needs ~1.5-2x more memory (stores the graph structure)
- Build time: HNSW takes minutes to build. Brute force = zero build time
- Update: HNSW supports incremental inserts. Some other indexes require full rebuild

HNSW is the best general-purpose index for most production use cases.

---

### Q4: Design a real-time feature pipeline using Kafka + Redis.

**Answer**:

```
Events → Kafka → Feature Pipeline → Redis → ML Serving

1. PRODUCER: App emits events to Kafka topic "user_events"
   {user_id: "u123", event: "purchase", amount: 49.99, timestamp: "..."}

2. KAFKA: Partitioned by user_id (all events for a user go to same partition,
   ensuring ordering)

3. CONSUMER (feature pipeline):
   - Reads events from Kafka
   - Computes real-time features:
     * purchase_count_1h: INCR in Redis (with 1h TTL)
     * total_spend_1h: INCRBYFLOAT in Redis
     * last_event_time: SET in Redis
   - Uses Redis atomic operations (pipeline/multi) for consistency

4. REDIS: Stores features as hash maps
   Key: "realtime:u123"
   Fields: {purchase_count_1h: 3, total_spend_1h: 149.97, ...}

5. SERVING: FastAPI reads features from Redis (<1ms)
   Combines with batch features for model inference
```

Key design decisions: partition by user_id for ordering, use Redis TTLs for automatic expiry, use pipelining for bulk updates, use consumer groups for horizontal scaling of the feature pipeline.

---

### Q5: How do you handle a DataFrame with 50% missing values in a column?

**Answer**: Depends on why the values are missing and the column's importance:

1. **Investigate first**: Is the missingness random (MCAR), correlated with other features (MAR), or correlated with the target (MNAR)?

2. **Options**:
   - **Drop the column**: If 50% missing AND low feature importance → remove it
   - **Impute with indicator**: Fill with median/mode AND add a binary `is_missing` column. The missingness itself can be informative
   - **Model-based imputation**: Use other features to predict the missing values (KNN imputer, iterative imputer)
   - **Category encoding**: For categorical columns, treat "missing" as its own category
   - **Leave as-is**: XGBoost and LightGBM handle missing values natively (they learn optimal split directions for missing values)

3. **Never**: Fill with 0 without thinking (0 might be a valid value). Forward-fill time series without considering gaps.

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Option 1: Impute + indicator
df["col_imputed"] = df["col"].fillna(df["col"].median())
df["col_missing"] = df["col"].isna().astype(int)

# Option 2: Let tree models handle it
# XGBoost/LightGBM: just pass the data as-is
```

---

### Q6: Write a SQL query to find the top 3 products per category by revenue.

**Answer**:
```sql
WITH ranked AS (
    SELECT
        category,
        product_id,
        SUM(amount) AS total_revenue,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY SUM(amount) DESC
        ) AS rank
    FROM orders
    GROUP BY category, product_id
)
SELECT category, product_id, total_revenue
FROM ranked
WHERE rank <= 3
ORDER BY category, rank;
```

Use `ROW_NUMBER()` if you want exactly 3 (no ties). Use `DENSE_RANK()` if you want to include ties (may return more than 3).

---

### Q7: Your serving latency jumped from 50ms to 500ms. Redis is involved. What do you check?

**Answer**: Investigate in order of likelihood:

1. **Redis connectivity**: Is Redis reachable? Check `redis-cli ping`. Network issue between app and Redis?
2. **Redis memory**: Is Redis out of memory? (`INFO memory`). Eviction happening? OOM = severe slowdown
3. **Slow commands**: Run `SLOWLOG GET 10`. Any commands taking >10ms? Large keys? KEYS * in production?
4. **Connection pool**: Are you creating new connections per request? (Should use a connection pool). Pool exhausted?
5. **Key size**: Did a feature value become unexpectedly large? (Serialized object instead of a number)
6. **Network**: Is there a firewall, proxy, or DNS change between your app and Redis?
7. **Not Redis**: Is the slowdown actually in the model inference, not Redis? Profile end-to-end to isolate

**Quick diagnostic**:
```bash
redis-cli INFO stats     # check operations/sec, keyspace hits/misses
redis-cli INFO memory    # check used_memory, evicted_keys
redis-cli SLOWLOG GET 10 # check slow commands
redis-cli --latency      # check baseline latency
```

---

### Q8: Explain the CAP theorem. How does it apply to ML infrastructure?

**Answer**: CAP theorem: a distributed system can provide at most 2 of 3:

- **C**onsistency: Every read returns the latest write
- **A**vailability: Every request gets a response
- **P**artition tolerance: System works despite network failures

In practice, P is non-negotiable (networks fail), so the choice is C vs A:

| ML Component | Choice | Why |
|-------------|--------|-----|
| Feature store (Redis) | AP | Serve stale features rather than fail. A missing feature is worse than a slightly outdated one |
| Model predictions DB | AP | Better to serve old predictions than 503 |
| Training data (PostgreSQL) | CP | Consistency matters — incorrect labels corrupt training |
| Experiment tracking (MLflow) | CP | Metrics must be accurate |
| Vector DB (Qdrant) | Tunable | Can configure per-collection |

For ML serving: **availability > consistency**. A stale feature is fine. A timeout is not.

---

### Q9: How would you migrate an ML system from using a SQL database for feature lookups to using Redis?

**Answer**: Phased migration:

**Phase 1 — Dual write**:
- Keep writing features to PostgreSQL (existing)
- Also write to Redis (new)
- Serving still reads from PostgreSQL

**Phase 2 — Shadow read**:
- Read from both PostgreSQL and Redis
- Compare results (log discrepancies, don't affect serving)
- Verify Redis has correct data

**Phase 3 — Cutover with fallback**:
- Serve from Redis primary
- Fall back to PostgreSQL if Redis is unavailable or returns null
- Monitor: Redis hit rate, latency improvement, fallback rate

**Phase 4 — Remove fallback**:
- Once Redis is stable (>99.9% hit rate, <1ms latency), remove PostgreSQL reads
- Keep PostgreSQL as the source of truth for batch feature computation
- Redis is populated by the batch pipeline (nightly) + stream pipeline (real-time)

Key risk: Redis is volatile (data loss on restart unless using AOF persistence). Always keep PostgreSQL as the durable source and treat Redis as a cache/serving layer.

---

### Q10: Design a data infrastructure for a recommendation system serving 10M users.

**Answer**:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Event Collection:                                      │
│  App → Kafka → S3 (raw events, partitioned by date)    │
│                                                         │
│  Batch Pipeline (nightly):                              │
│  S3 → Spark → BigQuery/Snowflake (feature extraction)  │
│  BigQuery → Parquet → Model Training                    │
│  BigQuery → Redis (materialized user features)          │
│  BigQuery → FAISS index (item embeddings, rebuilt daily)│
│                                                         │
│  Real-Time Pipeline:                                    │
│  Kafka → Flink/consumer → Redis (real-time features)    │
│  (session clicks, real-time velocity)                   │
│                                                         │
│  Serving:                                               │
│  Request → Redis (user features, <1ms)                  │
│         → FAISS (candidate retrieval, ~5ms)             │
│         → Model (ranking, ~10ms)                        │
│         → Response (total ~20ms)                        │
│                                                         │
│  Storage summary:                                       │
│  - Kafka: event streaming (hours retention)             │
│  - S3: raw events (permanent, cheap)                    │
│  - BigQuery: analytics + feature extraction (TB-scale)  │
│  - Redis: feature serving (10M users × 50 features)     │
│  - FAISS: item embeddings (1M items × 256 dims)         │
│  - PostgreSQL: metadata, configs, model registry         │
│                                                         │
│  Cost estimate (monthly):                               │
│  Redis (r6g.xlarge): ~$400                              │
│  BigQuery: ~$200 (on-demand)                            │
│  S3: ~$50                                               │
│  Kafka (MSK): ~$300                                     │
│  Compute (training): ~$500                              │
│  Total: ~$1,500/month                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

*Next: Topic 21 — Computer Vision in Production, or any other topic.*
