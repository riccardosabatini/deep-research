import aiosqlite
import asyncpg
import json
from typing import Optional, Dict, Any
from .configuration import Config

async def init_db():
    config = Config.from_env()
    
    if config.db_provider == "sqlite":
        async with aiosqlite.connect(config.db_uri) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    query TEXT PRIMARY KEY,
                    raw_result TEXT,
                    learnings TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()
            
    elif config.db_provider == "postgres":
        conn = await asyncpg.connect(config.db_uri)
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    query TEXT PRIMARY KEY,
                    raw_result TEXT,
                    learnings TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        finally:
            await conn.close()

async def get_search_result(query: str) -> Optional[Dict[str, Any]]:
    config = Config.from_env()
    
    if config.db_provider == "sqlite":
        async with aiosqlite.connect(config.db_uri) as db:
            async with db.execute("SELECT raw_result, learnings FROM search_results WHERE query = ?", (query,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "raw_result": json.loads(row[0]),
                        "learnings": row[1]
                    }
                    
    elif config.db_provider == "postgres":
        conn = await asyncpg.connect(config.db_uri)
        try:
            row = await conn.fetchrow("SELECT raw_result, learnings FROM search_results WHERE query = $1", query)
            if row:
                return {
                    "raw_result": json.loads(row["raw_result"]),
                    "learnings": row["learnings"]
                }
        finally:
            await conn.close()
            
    return None

async def save_search_result(query: str, raw_result: Dict[str, Any], learnings: str):
    config = Config.from_env()
    
    if config.db_provider == "sqlite":
        async with aiosqlite.connect(config.db_uri) as db:
            await db.execute(
                "INSERT OR REPLACE INTO search_results (query, raw_result, learnings) VALUES (?, ?, ?)",
                (query, json.dumps(raw_result), learnings)
            )
            await db.commit()
            
    elif config.db_provider == "postgres":
        conn = await asyncpg.connect(config.db_uri)
        try:
            await conn.execute(
                """
                INSERT INTO search_results (query, raw_result, learnings) 
                VALUES ($1, $2, $3) 
                ON CONFLICT (query) DO UPDATE 
                SET raw_result = $2, learnings = $3
                """,
                query, json.dumps(raw_result), learnings
            )
        finally:
            await conn.close()
