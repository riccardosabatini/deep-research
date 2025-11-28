import aiosqlite
import asyncpg
import json
from typing import Optional, Dict, Any
from .configuration import Config

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
                    research_id TEXT,
                    query TEXT,
                    raw_result TEXT,
                    learnings TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (research_id, query)
                )
            """)
            await db.commit()
            
    elif config.db_provider == "postgres":
        conn = await asyncpg.connect(config.db_uri)
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS search_results (
                    research_id TEXT,
                    query TEXT,
                    raw_result TEXT,
                    learnings TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (research_id, query)
                )
            """)
        finally:
            await conn.close()

async def get_search_result(research_id: str, query: str) -> Optional[Dict[str, Any]]:
    config = Config.from_env()
    
    if config.db_provider == "sqlite":
        async with aiosqlite.connect(config.db_uri) as db:
            async with db.execute(
                "SELECT raw_result, learnings FROM search_results WHERE research_id = ? AND query = ?", 
                (research_id, query)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "raw_result": json.loads(row[0]),
                        "learnings": row[1]
                    }
                    
    elif config.db_provider == "postgres":
        conn = await asyncpg.connect(config.db_uri)
        try:
            row = await conn.fetchrow(
                "SELECT raw_result, learnings FROM search_results WHERE research_id = $1 AND query = $2", 
                research_id, query
            )
            if row:
                return {
                    "raw_result": json.loads(row["raw_result"]),
                    "learnings": row["learnings"]
                }
        finally:
            await conn.close()
            
    return None

async def save_search_result(research_id: str, query: str, raw_result: Dict[str, Any], learnings: str):
    config = Config.from_env()
    
    if config.db_provider == "sqlite":
        async with aiosqlite.connect(config.db_uri) as db:
            await db.execute(
                "INSERT OR REPLACE INTO search_results (research_id, query, raw_result, learnings) VALUES (?, ?, ?, ?)",
                (research_id, query, json.dumps(raw_result), learnings)
            )
            await db.commit()
            
    elif config.db_provider == "postgres":
        conn = await asyncpg.connect(config.db_uri)
        try:
            await conn.execute(
                """
                INSERT INTO search_results (research_id, query, raw_result, learnings) 
                VALUES ($1, $2, $3, $4) 
                ON CONFLICT (research_id, query) DO UPDATE 
                SET raw_result = $3, learnings = $4
                """,
                research_id, query, json.dumps(raw_result), learnings
            )
        finally:
            await conn.close()
