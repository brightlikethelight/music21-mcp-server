"""
Security Middleware Integration for Music21 MCP Server

Integrates all security components:
- OAuth 2.1 with PKCE authentication
- Comprehensive input validation
- Real-time security monitoring
- SOC2/GDPR compliant audit logging
- Multi-tier rate limiting
- Intrusion detection

Provides seamless security layer for FastMCP server.
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Tuple
import logging
from datetime import datetime

from .oauth_provider import OAuth2Provider, verify_oauth_token
from .input_validation import SecurityAwareValidator
from .monitoring import SecurityMonitor, EventType, ThreatLevel, get_security_monitor
from .audit_logging import (
    AuditLogger, AuditEventType, DataCategory, ComplianceLevel,
    get_audit_logger, log_authentication_event, log_data_operation, log_security_event
)

# Security middleware logger
security_logger = logging.getLogger('music21_mcp.security.middleware')


class SecurityContext:
    """Security context for requests"""
    
    def __init__(self):
        self.user_id: Optional[str] = None
        self.client_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.source_ip: Optional[str] = None
        self.user_agent: Optional[str] = None
        self.request_id: Optional[str] = None
        self.authenticated: bool = False
        self.permissions: set = set()
        self.start_time: float = time.time()


class SecurityConfig:
    """Security configuration for MCP server"""
    
    def __init__(self):
        # Authentication settings
        self.require_authentication: bool = False  # Start with False for compatibility
        self.oauth_provider: Optional[OAuth2Provider] = None
        
        # Validation settings
        self.validate_inputs: bool = True
        self.strict_validation: bool = True
        
        # Monitoring settings
        self.enable_monitoring: bool = True
        self.enable_audit_logging: bool = True
        
        # Rate limiting (integrated with monitoring)
        self.rate_limiting_enabled: bool = True
        
        # Development mode (less strict security)
        self.development_mode: bool = True  # Default for backward compatibility


class SecurityMiddleware:
    """
    Comprehensive security middleware for Music21 MCP Server
    
    Provides:
    - Authentication and authorization
    - Input validation and sanitization
    - Real-time security monitoring
    - Comprehensive audit logging
    - Rate limiting and DDoS protection
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        
        # Initialize security components
        self.security_monitor = get_security_monitor()
        self.audit_logger = get_audit_logger()
        self.validator = SecurityAwareValidator()
        
        security_logger.info("Security middleware initialized")
    
    async def authenticate_request(self, context: SecurityContext, token: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Authenticate incoming request"""
        
        if not self.config.require_authentication:
            # Development mode - skip authentication but log access
            await self.audit_logger.log_event(
                AuditEventType.DATA_ACCESS,
                user_id="anonymous",
                source_ip=context.source_ip,
                action="development_access",
                outcome="success",
                business_justification="Development mode access"
            )
            context.authenticated = True
            return True, None
        
        if not token:
            await log_authentication_event(
                AuditEventType.AUTH_FAILURE,
                "anonymous",
                context.source_ip or "unknown",
                "failure"
            )
            return False, "Missing authentication token"
        
        try:
            # Verify OAuth token
            if self.config.oauth_provider:
                token_data = await self.config.oauth_provider.validate_token(token)
                context.user_id = token_data.get("client_id")
                context.client_id = token_data.get("client_id")
                context.authenticated = True
                
                await log_authentication_event(
                    AuditEventType.TOKEN_ISSUED,
                    context.user_id,
                    context.source_ip or "unknown",
                    "success"
                )
                
                return True, None
            else:
                return False, "OAuth provider not configured"
                
        except Exception as e:
            await log_authentication_event(
                AuditEventType.AUTH_FAILURE,
                context.user_id or "unknown",
                context.source_ip or "unknown",
                "failure"
            )
            return False, f"Authentication failed: {str(e)}"
    
    async def validate_request(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate request parameters"""
        
        if not self.config.validate_inputs:
            return True, None, parameters
        
        try:
            # Get validator for tool
            validator_class = self.validator.get_validator_for_tool(tool_name)
            
            if validator_class:
                # Validate parameters
                validated_data = self.validator.safe_validate(validator_class, parameters)
                return True, None, validated_data.dict() if hasattr(validated_data, 'dict') else parameters
            else:
                # No specific validator - basic validation
                if self.config.strict_validation:
                    security_logger.warning(f"No validator found for tool: {tool_name}")
                
                return True, None, parameters
                
        except ValueError as e:
            # Log validation failure
            await self.audit_logger.log_event(
                AuditEventType.SECURITY_VIOLATION,
                action="input_validation_failure",
                resource=tool_name,
                outcome="blocked",
                metadata={"error": str(e), "parameters": parameters}
            )
            
            return False, f"Input validation failed: {str(e)}", None
        
        except Exception as e:
            security_logger.error(f"Validation error for {tool_name}: {e}")
            return False, "Validation error", None
    
    async def monitor_request(self, context: SecurityContext, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Monitor request for security threats"""
        
        if not self.config.enable_monitoring:
            return True, None
        
        try:
            # Monitor with security system
            is_allowed, block_reason = await self.security_monitor.monitor_request(
                client_ip=context.source_ip or "unknown",
                user_agent=context.user_agent,
                client_id=context.client_id,
                endpoint=tool_name,
                payload=parameters
            )
            
            if not is_allowed:
                # Log security block
                await log_security_event(
                    EventType.ACCESS_DENIED,
                    context.source_ip or "unknown",
                    {
                        "tool": tool_name,
                        "reason": block_reason,
                        "user_id": context.user_id,
                        "parameters": parameters
                    }
                )
                
                return False, block_reason
            
            return True, None
            
        except Exception as e:
            security_logger.error(f"Security monitoring error: {e}")
            # Fail secure - block on monitoring errors in production
            if not self.config.development_mode:
                return False, "Security monitoring error"
            return True, None
    
    async def log_request(self, context: SecurityContext, tool_name: str, parameters: Dict[str, Any], outcome: str, response: Optional[Dict[str, Any]] = None):
        """Log request for audit trail"""
        
        if not self.config.enable_audit_logging:
            return
        
        try:
            # Determine data category based on tool
            data_category = self._get_data_category(tool_name)
            
            # Calculate duration
            duration_ms = int((time.time() - context.start_time) * 1000)
            
            # Log the operation
            await self.audit_logger.log_event(
                event_type=self._get_audit_event_type(tool_name),
                user_id=context.user_id or "anonymous",
                session_id=context.session_id,
                client_id=context.client_id,
                source_ip=context.source_ip,
                user_agent=context.user_agent,
                resource=tool_name,
                action=self._get_action_name(tool_name),
                outcome=outcome,
                data_category=data_category,
                duration_ms=duration_ms,
                business_justification="Music21 MCP service provision",
                legal_basis="Legitimate interest - service provision",
                metadata={
                    "parameters": parameters,
                    "response_summary": self._summarize_response(response) if response else None
                },
                compliance_level=ComplianceLevel.SOC2_TYPE_II
            )
            
        except Exception as e:
            security_logger.error(f"Audit logging error: {e}")
    
    def _get_data_category(self, tool_name: str) -> DataCategory:
        """Determine data category based on tool"""
        if tool_name in ['import_score', 'export_score', 'score_info']:
            return DataCategory.MUSICAL_DATA
        elif tool_name in ['key_analysis', 'harmony_analysis', 'pattern_recognition']:
            return DataCategory.TECHNICAL_DATA
        elif tool_name in ['list_scores']:
            return DataCategory.USAGE_DATA
        else:
            return DataCategory.SYSTEM_DATA
    
    def _get_audit_event_type(self, tool_name: str) -> AuditEventType:
        """Determine audit event type based on tool"""
        if tool_name == 'import_score':
            return AuditEventType.SCORE_IMPORTED
        elif tool_name == 'export_score':
            return AuditEventType.SCORE_EXPORTED
        elif tool_name == 'delete_score':
            return AuditEventType.SCORE_DELETED
        elif tool_name in ['key_analysis', 'chord_analysis']:
            return AuditEventType.SCORE_ANALYZED
        elif tool_name == 'harmony_analysis':
            return AuditEventType.HARMONY_ANALYSIS
        elif tool_name == 'pattern_recognition':
            return AuditEventType.PATTERN_ANALYSIS
        else:
            return AuditEventType.DATA_ACCESS
    
    def _get_action_name(self, tool_name: str) -> str:
        """Get human-readable action name"""
        action_map = {
            'import_score': 'import',
            'export_score': 'export',
            'delete_score': 'delete',
            'list_scores': 'list',
            'score_info': 'view',
            'key_analysis': 'analyze_key',
            'chord_analysis': 'analyze_chords',
            'harmony_analysis': 'analyze_harmony',
            'pattern_recognition': 'analyze_patterns'
        }
        return action_map.get(tool_name, 'execute')
    
    def _summarize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of response for logging (without sensitive data)"""
        if not response:
            return {}
        
        summary = {
            "status": response.get("status", "unknown"),
            "has_data": bool(response.get("data")),
            "response_size": len(str(response))
        }
        
        # Add specific summaries based on response type
        if "total_count" in response:
            summary["total_count"] = response["total_count"]
        
        if "message" in response:
            summary["has_message"] = True
        
        return summary
    
    async def handle_security_error(self, context: SecurityContext, error_type: str, error_message: str) -> Dict[str, Any]:
        """Handle security errors consistently"""
        
        # Log security error
        await self.audit_logger.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id=context.user_id,
            source_ip=context.source_ip,
            action="security_error",
            outcome="blocked",
            metadata={
                "error_type": error_type,
                "error_message": error_message
            }
        )
        
        # Return appropriate error response
        if self.config.development_mode:
            return {
                "status": "error",
                "error_type": error_type,
                "message": error_message
            }
        else:
            # Generic error message for production
            return {
                "status": "error",
                "message": "Access denied"
            }


def create_security_middleware(config: Optional[SecurityConfig] = None) -> SecurityMiddleware:
    """Create configured security middleware"""
    return SecurityMiddleware(config)


def create_development_security_config() -> SecurityConfig:
    """Create security config suitable for development"""
    config = SecurityConfig()
    config.require_authentication = False
    config.development_mode = True
    config.strict_validation = False
    config.enable_monitoring = True
    config.enable_audit_logging = True
    return config


def create_production_security_config(oauth_provider: OAuth2Provider) -> SecurityConfig:
    """Create security config for production deployment"""
    config = SecurityConfig()
    config.require_authentication = True
    config.oauth_provider = oauth_provider
    config.development_mode = False
    config.strict_validation = True
    config.enable_monitoring = True
    config.enable_audit_logging = True
    return config


# Export main classes
__all__ = [
    'SecurityMiddleware',
    'SecurityConfig',
    'SecurityContext',
    'create_security_middleware',
    'create_development_security_config',
    'create_production_security_config'
]