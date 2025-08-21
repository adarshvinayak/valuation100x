"""
Supabase Integration for DeepResearch
Handles database operations, file storage, and real-time subscriptions
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

logger = logging.getLogger(__name__)


class SupabaseManager:
    """Manages Supabase database and storage operations"""
    
    def __init__(self, 
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None):
        
        if not SUPABASE_AVAILABLE:
            raise RuntimeError("Supabase client not available. Install with: pip install supabase")
        
        # Get credentials from environment or parameters
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY environment variables are required")
        
        self.client: Optional[Client] = None
        self.initialized = False
        
        logger.info("SupabaseManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize Supabase client connection"""
        try:
            # Create Supabase client
            self.client = create_client(self.supabase_url, self.supabase_key)
            
            # Test connection with a simple query
            result = self.client.table('analysis_results').select('id').limit(1).execute()
            
            self.initialized = True
            logger.info("✅ Supabase connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase: {e}")
            self.initialized = False
            return False
    
    def _ensure_initialized(self):
        """Ensure client is initialized before operations"""
        if not self.initialized or not self.client:
            raise RuntimeError("SupabaseManager not initialized. Call initialize() first.")
    
    # ========================================
    # ANALYSIS RESULTS OPERATIONS
    # ========================================
    
    async def store_analysis_result(self, analysis_data: Dict[str, Any]) -> str:
        """Store analysis result in database"""
        self._ensure_initialized()
        
        try:
            # Prepare data for insertion
            insert_data = {
                'id': analysis_data.get('id') or str(uuid.uuid4()),
                'ticker': analysis_data.get('ticker', '').upper(),
                'analysis_type': analysis_data.get('analysis_type', 'comprehensive'),
                'status': analysis_data.get('status', 'running'),
                'score': analysis_data.get('investment_score'),
                'fair_value': analysis_data.get('fair_value'),
                'current_price': analysis_data.get('current_price'),
                'upside_percent': analysis_data.get('upside_percent'),
                'recommendation': analysis_data.get('recommendation'),
                'confidence_score': analysis_data.get('confidence_score'),
                'results_json': analysis_data.get('results_json') or analysis_data,
                'summary': analysis_data.get('summary'),
                'error_message': analysis_data.get('error_message'),
                'user_id': analysis_data.get('user_id'),
                'session_id': analysis_data.get('session_id'),
                'processing_time_seconds': analysis_data.get('processing_time_seconds'),
                'created_at': analysis_data.get('created_at') or datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Remove None values
            insert_data = {k: v for k, v in insert_data.items() if v is not None}
            
            # Insert into database
            result = self.client.table('analysis_results').insert(insert_data).execute()
            
            if result.data:
                analysis_id = result.data[0]['id']
                logger.info(f"✅ Stored analysis result for {insert_data['ticker']} (ID: {analysis_id})")
                return analysis_id
            else:
                raise Exception("No data returned from insert operation")
                
        except Exception as e:
            logger.error(f"❌ Failed to store analysis result: {e}")
            raise
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result by ID"""
        self._ensure_initialized()
        
        try:
            result = self.client.table('analysis_results').select('*').eq('id', analysis_id).execute()
            
            if result.data:
                logger.info(f"✅ Retrieved analysis result: {analysis_id}")
                return result.data[0]
            else:
                logger.warning(f"⚠️ Analysis result not found: {analysis_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve analysis result {analysis_id}: {e}")
            return None
    
    async def get_analysis_history(self, 
                                 ticker: Optional[str] = None,
                                 user_id: Optional[str] = None,
                                 session_id: Optional[str] = None,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis history with optional filters"""
        self._ensure_initialized()
        
        try:
            query = self.client.table('analysis_results').select('*')
            
            # Apply filters
            if ticker:
                query = query.eq('ticker', ticker.upper())
            if user_id:
                query = query.eq('user_id', user_id)
            if session_id:
                query = query.eq('session_id', session_id)
            
            # Order by creation date and limit
            result = query.order('created_at', desc=True).limit(limit).execute()
            
            logger.info(f"✅ Retrieved {len(result.data)} analysis records")
            return result.data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve analysis history: {e}")
            return []
    
    async def update_analysis_status(self, 
                                   analysis_id: str, 
                                   status: str,
                                   additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update analysis status and optional additional data"""
        self._ensure_initialized()
        
        try:
            update_data = {
                'status': status,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Add completion timestamp if completed
            if status == 'completed':
                update_data['completed_at'] = datetime.now(timezone.utc).isoformat()
            
            # Add any additional data
            if additional_data:
                update_data.update(additional_data)
            
            result = self.client.table('analysis_results').update(update_data).eq('id', analysis_id).execute()
            
            if result.data:
                logger.info(f"✅ Updated analysis {analysis_id} status to {status}")
                return True
            else:
                logger.warning(f"⚠️ No rows updated for analysis {analysis_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update analysis status {analysis_id}: {e}")
            return False
    
    # ========================================
    # FILE STORAGE OPERATIONS
    # ========================================
    
    async def upload_file(self, 
                         bucket: str, 
                         file_path: str, 
                         file_data: bytes,
                         content_type: Optional[str] = None) -> Optional[str]:
        """Upload file to Supabase Storage"""
        self._ensure_initialized()
        
        try:
            # Upload file to storage
            result = self.client.storage.from_(bucket).upload(
                file_path, 
                file_data,
                file_options={
                    'content-type': content_type or 'application/octet-stream'
                }
            )
            
            if result:
                # Get public URL for analysis-reports bucket
                if bucket == 'analysis-reports':
                    public_url = self.client.storage.from_(bucket).get_public_url(file_path)
                    logger.info(f"✅ Uploaded file to {bucket}/{file_path}")
                    return public_url
                else:
                    logger.info(f"✅ Uploaded private file to {bucket}/{file_path}")
                    return f"{bucket}/{file_path}"
            else:
                raise Exception("Upload failed - no result returned")
                
        except Exception as e:
            logger.error(f"❌ Failed to upload file to {bucket}/{file_path}: {e}")
            return None
    
    async def download_file(self, bucket: str, file_path: str) -> Optional[bytes]:
        """Download file from Supabase Storage"""
        self._ensure_initialized()
        
        try:
            result = self.client.storage.from_(bucket).download(file_path)
            
            if result:
                logger.info(f"✅ Downloaded file from {bucket}/{file_path}")
                return result
            else:
                logger.warning(f"⚠️ File not found: {bucket}/{file_path}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to download file from {bucket}/{file_path}: {e}")
            return None
    
    async def upload_analysis_report(self, 
                                   ticker: str, 
                                   analysis_id: str,
                                   report_content: str,
                                   report_type: str = 'markdown') -> Optional[str]:
        """Upload analysis report to storage"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = 'md' if report_type == 'markdown' else 'json'
            file_path = f"{ticker}/{analysis_id}_{timestamp}.{file_extension}"
            
            content_type = 'text/markdown' if report_type == 'markdown' else 'application/json'
            
            return await self.upload_file(
                bucket='analysis-reports',
                file_path=file_path,
                file_data=report_content.encode('utf-8'),
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to upload analysis report: {e}")
            return None
    
    async def upload_vector_index(self, 
                                ticker: str, 
                                index_data: bytes) -> Optional[str]:
        """Upload FAISS vector index to storage"""
        try:
            file_path = f"{ticker}/faiss_index"
            
            return await self.upload_file(
                bucket='vector-indexes',
                file_path=file_path,
                file_data=index_data,
                content_type='application/octet-stream'
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to upload vector index: {e}")
            return None
    
    # ========================================
    # SESSION MANAGEMENT
    # ========================================
    
    async def create_user_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new user session"""
        self._ensure_initialized()
        
        try:
            session_id = str(uuid.uuid4())
            insert_data = {
                'session_id': session_id,
                'user_data': session_data,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table('user_sessions').insert(insert_data).execute()
            
            if result.data:
                logger.info(f"✅ Created user session: {session_id}")
                return session_id
            else:
                raise Exception("Failed to create session")
                
        except Exception as e:
            logger.error(f"❌ Failed to create user session: {e}")
            raise
    
    async def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user session data"""
        self._ensure_initialized()
        
        try:
            result = self.client.table('user_sessions').select('*').eq('session_id', session_id).execute()
            
            if result.data:
                return result.data[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve user session {session_id}: {e}")
            return None
    
    # ========================================
    # CACHING OPERATIONS
    # ========================================
    
    async def cache_set(self, key: str, value: Dict[str, Any], ttl_hours: int = 24) -> bool:
        """Store data in analysis cache"""
        self._ensure_initialized()
        
        try:
            insert_data = {
                'cache_key': key,
                'cache_value': value,
                'ttl_hours': ttl_hours,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'expires_at': datetime.now(timezone.utc).isoformat()  # Will be calculated by DB
            }
            
            # Use upsert to handle key conflicts
            result = self.client.table('analysis_cache').upsert(insert_data, on_conflict='cache_key').execute()
            
            if result.data:
                logger.info(f"✅ Cached data with key: {key}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to cache data for key {key}: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from analysis cache"""
        self._ensure_initialized()
        
        try:
            # Get non-expired cache entries
            result = self.client.table('analysis_cache').select('cache_value').eq('cache_key', key).gt('expires_at', datetime.now(timezone.utc).isoformat()).execute()
            
            if result.data:
                logger.info(f"✅ Cache hit for key: {key}")
                return result.data[0]['cache_value']
            else:
                logger.info(f"🔍 Cache miss for key: {key}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve cache for key {key}: {e}")
            return None
    
    # ========================================
    # ANALYTICS & MONITORING
    # ========================================
    
    async def log_api_usage(self, 
                           endpoint: str,
                           method: str,
                           status_code: int,
                           response_time_ms: int,
                           session_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           ticker: Optional[str] = None,
                           error_message: Optional[str] = None) -> bool:
        """Log API usage for analytics"""
        self._ensure_initialized()
        
        try:
            insert_data = {
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time_ms': response_time_ms,
                'session_id': session_id,
                'user_id': user_id,
                'ticker': ticker,
                'error_message': error_message,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Remove None values
            insert_data = {k: v for k, v in insert_data.items() if v is not None}
            
            result = self.client.table('api_usage').insert(insert_data).execute()
            
            return bool(result.data)
            
        except Exception as e:
            logger.error(f"❌ Failed to log API usage: {e}")
            return False
    
    async def close(self):
        """Close Supabase connection (cleanup)"""
        if self.client:
            # Supabase client doesn't need explicit closing
            self.initialized = False
            logger.info("🔌 Supabase client disconnected")


# Global instance
supabase_manager = SupabaseManager()


async def initialize_supabase() -> bool:
    """Initialize the global Supabase manager"""
    return await supabase_manager.initialize()


# Convenience functions
async def store_analysis(analysis_data: Dict[str, Any]) -> str:
    """Store analysis result"""
    return await supabase_manager.store_analysis_result(analysis_data)


async def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get analysis result by ID"""
    return await supabase_manager.get_analysis_result(analysis_id)


async def upload_report(ticker: str, analysis_id: str, content: str, report_type: str = 'markdown') -> Optional[str]:
    """Upload analysis report"""
    return await supabase_manager.upload_analysis_report(ticker, analysis_id, content, report_type)
