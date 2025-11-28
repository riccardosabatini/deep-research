import pytest
import sys
from main import parse_arguments

def test_cli_defaults():
    """Test default arguments"""
    args = parse_arguments(["test query"])
    assert args.query == "test query"
    assert args.feedback_mode == "human"
    assert args.thread_id is None

def test_cli_feedback_mode_human():
    """Test explicit human feedback mode"""
    args = parse_arguments(["test query", "--feedback-mode", "human"])
    assert args.feedback_mode == "human"

def test_cli_feedback_mode_auto():
    """Test explicit auto feedback mode"""
    args = parse_arguments(["test query", "--feedback-mode", "auto"])
    assert args.feedback_mode == "auto"

def test_cli_invalid_feedback_mode():
    """Test invalid feedback mode raises error"""
    with pytest.raises(SystemExit):
        parse_arguments(["test query", "--feedback-mode", "invalid"])

def test_cli_thread_id():
    """Test thread ID argument"""
    args = parse_arguments(["--thread-id", "123"])
    assert args.thread_id == "123"
    assert args.query is None
