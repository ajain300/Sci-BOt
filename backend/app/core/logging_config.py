import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(log_level: str = "DEBUG") -> None:
    """
    Set up logging configuration for the backend.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory in the project root
    project_root = Path(__file__).parent.parent.parent  # backend/app/core -> backend/
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = log_dir / "backend.log"
    print(f"Creating log file at: {log_file.absolute()}")  # Debug print
    
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Create loggers for different components
    loggers = {
        "active_learning": logging.getLogger("app.core.active_learning"),
        "api": logging.getLogger("app.api"),
        "optimization": logging.getLogger("app.core.optimization"),
    }
    
    # Set level for all component loggers
    for logger in loggers.values():
        logger.setLevel(log_level) 