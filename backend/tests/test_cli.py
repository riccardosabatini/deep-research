import pytest
import sys
from main import parse_arguments

def test_cli_defaults():
    """Test default arguments"""
    args, _ = parse_arguments(["--query", "test query"])
    assert args.query == "test query"
    assert args.feedback_mode == "human"
    assert args.thread_id is None

def test_cli_feedback_mode_human():
    """Test explicit human feedback mode"""
    args, _ = parse_arguments(["--query", "test query", "--feedback-mode", "human"])
    assert args.feedback_mode == "human"

def test_cli_feedback_mode_auto():
    """Test explicit auto feedback mode"""
    args, _ = parse_arguments(["--query", "test query", "--feedback-mode", "auto"])
    assert args.feedback_mode == "auto"

def test_cli_invalid_feedback_mode():
    """Test invalid feedback mode raises error"""
    with pytest.raises(SystemExit):
        parse_arguments(["--query", "test query", "--feedback-mode", "invalid"])

def test_cli_thread_id():
    """Test thread ID argument"""
    args, _ = parse_arguments(["--thread-id", "123"])
    assert args.thread_id == "123"
    assert args.query is None

def test_cli_report_pages():
    """Test report pages argument"""
    args, _ = parse_arguments(["--query", "test query", "--report-pages", "10"])
    assert args.report_pages == 10

def test_cli_report_pages_default():
    """Test default report pages"""
    args, _ = parse_arguments(["--query", "test query"])
    assert args.report_pages == 5

def test_cli_max_search_results():
    """Test max search results argument"""
    args, _ = parse_arguments(["--query", "test query", "--max-search-results", "10"])
    assert args.max_search_results == 10

def test_cli_max_search_results_default():
    """Test default max search results"""
    args, _ = parse_arguments(["--query", "test query"])
    assert args.max_search_results == 5

def test_cli_prompt_set():
    """Test prompt set argument"""
    args, _ = parse_arguments(["--query", "test query", "--prompt-set", "example"])
    assert args.prompt_set == "example"

def test_cli_prompt_set_default():
    """Test default prompt set"""
    args, _ = parse_arguments(["--query", "test query"])
    assert args.prompt_set == "default"
