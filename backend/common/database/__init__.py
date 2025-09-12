"""
Database Module for DeepResearch
Handles Supabase integration and data persistence
"""

from .supabase_client import (
    SupabaseManager,
    supabase_manager,
    initialize_supabase,
    store_analysis,
    get_analysis,
    upload_report
)

__all__ = [
    'SupabaseManager',
    'supabase_manager', 
    'initialize_supabase',
    'store_analysis',
    'get_analysis',
    'upload_report'
]
