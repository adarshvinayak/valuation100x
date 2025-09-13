"""
Secure Configuration Management for DeepResearch Backend
Validates environment variables and provides secure configuration access
"""

import os
import re
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when configuration validation fails"""
    pass

class SecureConfig:
    """Secure configuration manager with validation"""

    # Required environment variables
    REQUIRED_VARS = [
        'OPENAI_API_KEY',
        'TAVILY_API_KEY',
        'FMP_API_KEY',
        'SEC_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY'
    ]

    # Optional environment variables with defaults
    OPTIONAL_VARS = {
        'REDIS_URL': 'redis://localhost:6379',
        'APP_ENV': 'development',
        'LOG_LEVEL': 'INFO',
        'REQUEST_COOLDOWN_SECONDS': '30',
        'MAX_CONCURRENT_ANALYSES': '5',
        'ENABLE_ANALYSIS_CACHE': 'true',
        'ENABLE_VECTOR_SEARCH': 'true',
        'EMBEDDING_PROVIDER': 'local',
        'BLOOMBERG_API_KEY': None
    }

    def __init__(self):
        self._config = {}
        self._validated = False
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables"""
        # Load required variables
        for var in self.REQUIRED_VARS:
            value = os.getenv(var)
            if value is None:
                raise ConfigurationError(f"Required environment variable {var} is not set")
            self._config[var] = value

        # Load optional variables with defaults
        for var, default in self.OPTIONAL_VARS.items():
            value = os.getenv(var, default)
            self._config[var] = value

    def _validate_api_keys(self):
        """Validate API key formats"""
        validations = {
            'OPENAI_API_KEY': lambda x: x.startswith('sk-') and len(x) > 20,
            'TAVILY_API_KEY': lambda x: x.startswith('tvly-') and len(x) > 10,
            'FMP_API_KEY': lambda x: len(x) >= 20 and x.replace('-', '').replace('_', '').isalnum(),
            'SEC_API_KEY': lambda x: len(x) >= 20 and x.isalnum(),
            'SUPABASE_URL': lambda x: x.startswith('https://') and 'supabase.co' in x,
            'SUPABASE_ANON_KEY': lambda x: len(x) >= 100  # Supabase keys are long
        }

        for var, validator in validations.items():
            if var in self._config and not validator(self._config[var]):
                logger.warning(f"⚠️ {var} format validation failed - please verify the key")

    def _validate_numeric_config(self):
        """Validate numeric configuration values"""
        numeric_vars = ['REQUEST_COOLDOWN_SECONDS', 'MAX_CONCURRENT_ANALYSES']

        for var in numeric_vars:
            try:
                self._config[var] = int(self._config[var])
            except (ValueError, TypeError):
                logger.warning(f"⚠️ {var} must be numeric, using default")
                self._config[var] = int(self.OPTIONAL_VARS.get(var, '30'))

    def _validate_boolean_config(self):
        """Validate boolean configuration values"""
        boolean_vars = ['ENABLE_ANALYSIS_CACHE', 'ENABLE_VECTOR_SEARCH']

        for var in boolean_vars:
            value = self._config[var].lower()
            if value in ['true', '1', 'yes', 'on']:
                self._config[var] = True
            elif value in ['false', '0', 'no', 'off']:
                self._config[var] = False
            else:
                logger.warning(f"⚠️ {var} must be boolean, using default")
                self._config[var] = self.OPTIONAL_VARS.get(var, 'true').lower() == 'true'

    def validate(self) -> bool:
        """Validate all configuration"""
        try:
            self._validate_api_keys()
            self._validate_numeric_config()
            self._validate_boolean_config()
            self._validated = True
            logger.info("✅ Configuration validation successful")
            return True
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value with dict-like access"""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._config

    @property
    def is_validated(self) -> bool:
        """Check if configuration has been validated"""
        return self._validated

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration (for debugging - masks sensitive values)"""
        masked_config = {}
        sensitive_keys = ['API_KEY', 'SECRET', 'TOKEN', 'PASSWORD']

        for key, value in self._config.items():
            if any(sensitive in key.upper() for sensitive in sensitive_keys):
                masked_config[key] = f"***{key}***"  # Mask sensitive values
            else:
                masked_config[key] = value

        return masked_config

# Global configuration instance
config = SecureConfig()

def validate_ticker_symbol(ticker: str) -> tuple[bool, str]:
    """
    Validate stock ticker symbol format and security

    Args:
        ticker: Stock ticker symbol to validate

    Returns:
        tuple: (is_valid, error_message)
    """
    if not ticker:
        return False, "Ticker symbol cannot be empty"

    # Remove whitespace and convert to uppercase
    ticker = ticker.strip().upper()

    # Basic length validation
    if len(ticker) < 1 or len(ticker) > 10:
        return False, "Ticker symbol must be 1-10 characters long"

    # Character validation - only letters, numbers, and dots allowed
    if not re.match(r'^[A-Z0-9.]+$', ticker):
        return False, "Ticker symbol can only contain letters, numbers, and dots"

    # Security checks - prevent path traversal and injection
    if any(char in ticker for char in ['/', '\\', '..', '<', '>', ':', '|', '?', '*']):
        return False, "Ticker symbol contains invalid characters"

    # Prevent SQL injection patterns
    sql_patterns = ['--', '/*', '*/', 'xp_', 'sp_', 'exec', 'union', 'select', 'drop', 'delete']
    if any(pattern in ticker.lower() for pattern in sql_patterns):
        return False, "Ticker symbol contains potentially malicious patterns"

    return True, ""

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        str: Sanitized text
    """
    if not text:
        return ""

    # Remove null bytes and other control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Limit length
    if len(text) > max_length:
        text = text[:max_length]

    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", ';', '\\', '\n', '\r', '\t']
    for char in dangerous_chars:
        text = text.replace(char, '')

    return text.strip()

def validate_request_data(data: Dict[str, Any], required_fields: List[str]) -> tuple[bool, str]:
    """
    Validate request data structure

    Args:
        data: Request data dictionary
        required_fields: List of required field names

    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Request data must be a JSON object"

    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            missing_fields.append(field)

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    return True, ""

# Initialize configuration on import
try:
    if not config.validate():
        logger.error("❌ Configuration validation failed on startup")
        raise ConfigurationError("Invalid configuration")
    else:
        logger.info("✅ Secure configuration loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load secure configuration: {e}")
    raise
