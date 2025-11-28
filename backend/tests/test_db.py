import pytest
import aiosqlite
import json
from unittest.mock import MagicMock
from deepresearch.results_db import init_db, get_search_result, save_search_result, save_report, get_report
from deepresearch.configuration import Config

@pytest.fixture
def mock_db_config(mocker):
    mock_config = MagicMock(spec=Config)
    mock_config.db_provider = "sqlite"
    mock_config.db_uri = ":memory:"
    mocker.patch("deepresearch.results_db.Config.from_env", return_value=mock_config)
    return mock_config

@pytest.mark.asyncio
async def test_init_db(mock_db_config):
    # init_db creates tables. For :memory:, we need to keep the connection open?
    # src.results_db.init_db opens a connection, creates tables, and closes it.
    # For :memory:, closing the connection loses the DB.
    # So testing init_db with :memory: via the function is tricky if it closes connection.
    
    # However, src.results_db functions open a NEW connection each time.
    # If db_uri is ":memory:", each call creates a FRESH in-memory DB.
    # So we can't share state between init_db and get/save if we use ":memory:" string.
    
    # We need to use a shared connection or a file-based DB for testing persistence across calls.
    # Or patch aiosqlite.connect to return a shared mock connection?
    
    # Let's use a temporary file for SQLite testing to ensure persistence.
    pass

@pytest.mark.asyncio
async def test_db_operations(tmp_path, mocker):
    # Use a temp file for DB
    db_path = tmp_path / "test.db"
    db_uri = str(db_path)
    
    mock_config = MagicMock(spec=Config)
    mock_config.db_provider = "sqlite"
    mock_config.db_uri = db_uri
    mocker.patch("deepresearch.results_db.Config.from_env", return_value=mock_config)
    
    # 1. Init DB
    await init_db()
    
    # Verify table exists
    async with aiosqlite.connect(db_uri) as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_results';") as cursor:
            result = await cursor.fetchone()
            assert result is not None
            
    # 2. Save Result
    research_id = "test_thread"
    query = "test query"
    raw_result = {"foo": "bar"}
    learnings = ["l1", "l2"]
    
    await save_search_result(research_id, query, raw_result, learnings)
    
    # 3. Get Result
    result = await get_search_result(research_id, query)
    assert result is not None
    assert result["raw_result"] == raw_result
    assert result["learnings"] == learnings
    
    # 4. Get Non-existent Result
    result = await get_search_result(research_id, "other query")
    assert result is None
    
    # 5. Get Result with different research_id
    result = await get_search_result("other_thread", query)
    assert result is None

@pytest.mark.asyncio
async def test_report_operations(tmp_path, mocker):
    # Use a temp file for DB
    db_path = tmp_path / "test_report.db"
    db_uri = str(db_path)
    
    mock_config = MagicMock(spec=Config)
    mock_config.db_provider = "sqlite"
    mock_config.db_uri = db_uri
    mocker.patch("deepresearch.results_db.Config.from_env", return_value=mock_config)
    
    # 1. Init DB
    await init_db()
    
    # Verify table exists
    async with aiosqlite.connect(db_uri) as db:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reports';") as cursor:
            result = await cursor.fetchone()
            assert result is not None
            
    # 2. Save Report
    research_id = "test_thread_report"
    report_content = "This is a test report."
    
    await save_report(research_id, report_content)
    
    # 3. Get Report
    saved_report = await get_report(research_id)
    assert saved_report == report_content
    
    # 4. Update Report
    updated_report = "This is an updated report."
    await save_report(research_id, updated_report)
    
    saved_report = await get_report(research_id)
    assert saved_report == updated_report
    
    # 5. Get Non-existent Report
    saved_report = await get_report("non_existent_thread")
    assert saved_report is None
