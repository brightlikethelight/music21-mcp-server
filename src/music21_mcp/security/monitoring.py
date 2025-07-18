"""
Security Monitoring System for Music21 MCP Server

Enterprise-grade security monitoring with:
- Real-time intrusion detection
- Anomaly detection and behavioral analysis
- Multi-tier rate limiting
- Threat intelligence and correlation
- Automated incident response
- SOC2/GDPR compliant logging

Complies with 2024 security standards for production deployment.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import hashlib
import ipaddress
from pathlib import Path

from pydantic import BaseModel, Field

# Security monitoring logger
security_logger = logging.getLogger('music21_mcp.security.monitoring')
audit_logger = logging.getLogger('music21_mcp.audit')


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Security event types"""
    AUTH_FAILURE = "auth_failure"
    VALIDATION_FAILURE = "validation_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_ERROR = "system_error"
    ACCESS_DENIED = "access_denied"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    threat_level: ThreatLevel
    source_ip: Optional[str]
    user_agent: Optional[str]
    client_id: Optional[str]
    endpoint: Optional[str]
    payload_hash: Optional[str]
    details: Dict[str, Any]
    mitigated: bool = False
    mitigation_action: Optional[str] = None


class SecurityMetrics(BaseModel):
    """Real-time security metrics"""
    total_requests: int = 0
    blocked_requests: int = 0
    suspicious_events: int = 0
    active_threats: int = 0
    rate_limited_ips: int = 0
    anomaly_score: float = 0.0
    threat_score: float = 0.0
    uptime_hours: float = 0.0


class RateLimitConfig(BaseModel):
    """Multi-tier rate limiting configuration"""
    requests_per_minute: int = Field(default=60, ge=1, le=1000)
    requests_per_hour: int = Field(default=1000, ge=1, le=10000)
    requests_per_day: int = Field(default=10000, ge=1, le=100000)
    burst_threshold: int = Field(default=10, ge=1, le=100)
    lockout_duration: int = Field(default=300, ge=60, le=3600)  # seconds


class AnomalyThresholds(BaseModel):
    """Anomaly detection thresholds"""
    max_requests_per_second: float = Field(default=10.0, ge=0.1, le=100.0)
    max_payload_size: int = Field(default=1048576, ge=1024, le=10485760)  # 1MB default
    max_response_time: float = Field(default=30.0, ge=1.0, le=300.0)
    unusual_pattern_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    geographical_anomaly_threshold: float = Field(default=0.9, ge=0.5, le=1.0)


class SecurityMonitor:
    """
    Comprehensive security monitoring system
    
    Features:
    - Real-time threat detection
    - Multi-tier rate limiting  
    - Behavioral anomaly detection
    - Automated incident response
    - Security metrics and alerting
    """
    
    def __init__(self, 
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 anomaly_thresholds: Optional[AnomalyThresholds] = None):
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.anomaly_thresholds = anomaly_thresholds or AnomalyThresholds()
        
        # Event storage
        self.security_events: deque = deque(maxlen=10000)  # Last 10k events
        self.threat_intelligence: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting tracking
        self.request_counts: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                'minute': deque(maxlen=self.rate_limit_config.requests_per_minute),
                'hour': deque(maxlen=self.rate_limit_config.requests_per_hour),
                'day': deque(maxlen=self.rate_limit_config.requests_per_day)
            }
        )
        self.blocked_ips: Dict[str, float] = {}  # IP -> blocked_until_timestamp
        
        # Anomaly detection
        self.baseline_metrics: Dict[str, List[float]] = defaultdict(list)
        self.user_behavior_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Real-time metrics
        self.metrics = SecurityMetrics()
        self.start_time = time.time()
        
        # Known threat patterns
        self.threat_patterns = self._load_threat_patterns()
        
        # Geolocation tracking (simple country-based)
        self.user_locations: Dict[str, Set[str]] = defaultdict(set)
        
        security_logger.info("Security monitoring system initialized")
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load known threat patterns for detection"""
        return {
            'sql_injection': [
                r"'?\s*(or|and)\s+\d+\s*=\s*\d+",
                r"union\s+select",
                r"drop\s+table",
                r"insert\s+into",
                r"delete\s+from",
                r"update\s+.*\s+set",
                r"exec\s*\(",
                r"xp_cmdshell"
            ],
            'xss_injection': [
                r"<script[^>]*>",
                r"javascript:",
                r"vbscript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"onclick\s*=",
                r"eval\s*\(",
                r"document\.cookie"
            ],
            'path_traversal': [
                r"\.\.[\\/]",
                r"[\\/]etc[\\/]passwd",
                r"[\\/]windows[\\/]system32",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"..%2f",
                r"..%5c"
            ],
            'command_injection': [
                r";\s*(cat|ls|dir|type)\s+",
                r";\s*(rm|del)\s+",
                r";\s*wget\s+",
                r";\s*curl\s+",
                r"\|\s*(nc|netcat)\s+",
                r"&\s*(ping|nslookup)\s+",
                r"`.*`",
                r"\$\(.*\)"
            ],
            'music21_specific': [
                r"corpus\.parse\s*\(\s*['\"]\.\.[\\/]",
                r"converter\.parse\s*\(\s*['\"]\/etc",
                r"stream\.write\s*\(\s*['\"]\/",
                r"music21.*exec",
                r"music21.*eval"
            ]
        }
    
    async def monitor_request(self, 
                            client_ip: str,
                            user_agent: Optional[str] = None,
                            client_id: Optional[str] = None,
                            endpoint: str = "",
                            payload: Optional[Dict[str, Any]] = None,
                            response_time: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Monitor incoming request for security threats
        
        Returns:
            Tuple of (is_allowed, block_reason)
        """
        self.metrics.total_requests += 1
        current_time = time.time()
        
        # Update uptime
        self.metrics.uptime_hours = (current_time - self.start_time) / 3600
        
        # Check if IP is currently blocked
        if client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                await self._log_security_event(
                    EventType.ACCESS_DENIED,
                    ThreatLevel.MEDIUM,
                    client_ip=client_ip,
                    endpoint=endpoint,
                    details={"reason": "IP blocked due to previous violations"}
                )
                return False, "IP temporarily blocked"
            else:
                # Unblock expired IPs
                del self.blocked_ips[client_ip]
        
        # Rate limiting check
        rate_limit_result = await self._check_rate_limits(client_ip, current_time)
        if not rate_limit_result[0]:
            return rate_limit_result
        
        # Threat pattern detection
        threat_result = await self._detect_threat_patterns(
            client_ip, user_agent, endpoint, payload
        )
        if not threat_result[0]:
            return threat_result
        
        # Anomaly detection
        anomaly_result = await self._detect_anomalies(
            client_ip, client_id, endpoint, payload, response_time
        )
        if not anomaly_result[0]:
            return anomaly_result
        
        # Update user behavior profile
        await self._update_behavior_profile(client_ip, client_id, endpoint, current_time)
        
        return True, None
    
    async def _check_rate_limits(self, client_ip: str, current_time: float) -> Tuple[bool, Optional[str]]:
        """Check multi-tier rate limits"""
        now = datetime.fromtimestamp(current_time)
        
        # Add current request to tracking
        for timeframe in ['minute', 'hour', 'day']:
            self.request_counts[client_ip][timeframe].append(current_time)
        
        # Clean old entries
        cutoff_times = {
            'minute': current_time - 60,
            'hour': current_time - 3600,
            'day': current_time - 86400
        }
        
        for timeframe, cutoff in cutoff_times.items():
            requests = self.request_counts[client_ip][timeframe]
            while requests and requests[0] < cutoff:
                requests.popleft()
        
        # Check limits
        limits = {
            'minute': self.rate_limit_config.requests_per_minute,
            'hour': self.rate_limit_config.requests_per_hour,
            'day': self.rate_limit_config.requests_per_day
        }
        
        for timeframe, limit in limits.items():
            request_count = len(self.request_counts[client_ip][timeframe])
            if request_count > limit:
                # Block IP
                self.blocked_ips[client_ip] = current_time + self.rate_limit_config.lockout_duration
                self.metrics.blocked_requests += 1
                self.metrics.rate_limited_ips += 1
                
                await self._log_security_event(
                    EventType.RATE_LIMIT_EXCEEDED,
                    ThreatLevel.MEDIUM,
                    client_ip=client_ip,
                    details={
                        "timeframe": timeframe,
                        "request_count": request_count,
                        "limit": limit,
                        "lockout_duration": self.rate_limit_config.lockout_duration
                    }
                )
                
                return False, f"Rate limit exceeded for {timeframe}"
        
        # Check burst threshold
        recent_requests = [
            t for t in self.request_counts[client_ip]['minute']
            if current_time - t < 10  # Last 10 seconds
        ]
        
        if len(recent_requests) > self.rate_limit_config.burst_threshold:
            self.blocked_ips[client_ip] = current_time + 60  # Short burst block
            self.metrics.blocked_requests += 1
            
            await self._log_security_event(
                EventType.RATE_LIMIT_EXCEEDED,
                ThreatLevel.HIGH,
                client_ip=client_ip,
                details={
                    "burst_requests": len(recent_requests),
                    "burst_threshold": self.rate_limit_config.burst_threshold
                }
            )
            
            return False, "Burst rate limit exceeded"
        
        return True, None
    
    async def _detect_threat_patterns(self,
                                    client_ip: str,
                                    user_agent: Optional[str],
                                    endpoint: str,
                                    payload: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Detect known threat patterns in request data"""
        import re
        
        # Combine all text data for pattern matching
        text_data = [endpoint]
        if user_agent:
            text_data.append(user_agent)
        if payload:
            text_data.append(json.dumps(payload, default=str))
        
        combined_text = ' '.join(text_data).lower()
        
        # Check each threat pattern category
        for category, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    # Threat detected
                    threat_level = self._calculate_threat_level(category, pattern)
                    
                    # Block high/critical threats immediately
                    if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        self.blocked_ips[client_ip] = time.time() + self.rate_limit_config.lockout_duration
                        self.metrics.blocked_requests += 1
                        self.metrics.active_threats += 1
                    
                    await self._log_security_event(
                        EventType.INTRUSION_ATTEMPT,
                        threat_level,
                        client_ip=client_ip,
                        user_agent=user_agent,
                        endpoint=endpoint,
                        payload_hash=self._hash_payload(payload) if payload else None,
                        details={
                            "threat_category": category,
                            "matched_pattern": pattern,
                            "matched_text": combined_text[:200]  # First 200 chars
                        }
                    )
                    
                    if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        return False, f"Threat pattern detected: {category}"
        
        return True, None
    
    async def _detect_anomalies(self,
                              client_ip: str,
                              client_id: Optional[str],
                              endpoint: str,
                              payload: Optional[Dict[str, Any]],
                              response_time: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Detect behavioral and statistical anomalies"""
        anomalies = []
        current_time = time.time()
        
        # Request rate anomaly
        recent_requests = [
            t for t in self.request_counts[client_ip]['minute']
            if current_time - t < 1  # Last second
        ]
        requests_per_second = len(recent_requests)
        
        if requests_per_second > self.anomaly_thresholds.max_requests_per_second:
            anomalies.append({
                "type": "high_request_rate",
                "value": requests_per_second,
                "threshold": self.anomaly_thresholds.max_requests_per_second
            })
        
        # Payload size anomaly
        if payload:
            payload_size = len(json.dumps(payload, default=str))
            if payload_size > self.anomaly_thresholds.max_payload_size:
                anomalies.append({
                    "type": "large_payload",
                    "value": payload_size,
                    "threshold": self.anomaly_thresholds.max_payload_size
                })
        
        # Response time anomaly
        if response_time and response_time > self.anomaly_thresholds.max_response_time:
            anomalies.append({
                "type": "slow_response",
                "value": response_time,
                "threshold": self.anomaly_thresholds.max_response_time
            })
        
        # User behavior anomaly
        behavior_anomaly = await self._check_behavioral_anomaly(client_ip, client_id, endpoint)
        if behavior_anomaly:
            anomalies.append(behavior_anomaly)
        
        # Process anomalies
        if anomalies:
            anomaly_score = len(anomalies) / 4.0  # Normalize to 0-1
            self.metrics.anomaly_score = max(self.metrics.anomaly_score, anomaly_score)
            
            threat_level = ThreatLevel.LOW
            if anomaly_score > 0.75:
                threat_level = ThreatLevel.CRITICAL
            elif anomaly_score > 0.5:
                threat_level = ThreatLevel.HIGH
            elif anomaly_score > 0.25:
                threat_level = ThreatLevel.MEDIUM
            
            await self._log_security_event(
                EventType.ANOMALY_DETECTED,
                threat_level,
                client_ip=client_ip,
                client_id=client_id,
                endpoint=endpoint,
                details={
                    "anomalies": anomalies,
                    "anomaly_score": anomaly_score
                }
            )
            
            # Block critical anomalies
            if threat_level == ThreatLevel.CRITICAL:
                self.blocked_ips[client_ip] = time.time() + self.rate_limit_config.lockout_duration
                self.metrics.blocked_requests += 1
                return False, "Critical anomaly detected"
        
        return True, None
    
    async def _check_behavioral_anomaly(self,
                                      client_ip: str,
                                      client_id: Optional[str],
                                      endpoint: str) -> Optional[Dict[str, Any]]:
        """Check for unusual user behavior patterns"""
        user_key = client_id or client_ip
        
        if user_key not in self.user_behavior_profiles:
            return None
        
        profile = self.user_behavior_profiles[user_key]
        
        # Check endpoint usage patterns
        endpoint_history = profile.get('endpoints', {})
        total_requests = sum(endpoint_history.values())
        
        if total_requests > 10:  # Need some history
            endpoint_frequency = endpoint_history.get(endpoint, 0) / total_requests
            
            # If this endpoint is used very rarely by this user, it's suspicious
            if endpoint_frequency < 0.05 and total_requests > 50:
                return {
                    "type": "unusual_endpoint",
                    "endpoint": endpoint,
                    "frequency": endpoint_frequency,
                    "total_requests": total_requests
                }
        
        # Check request timing patterns
        request_times = profile.get('request_times', [])
        if len(request_times) > 20:
            # Check for unusual request patterns (too regular or too random)
            intervals = [request_times[i] - request_times[i-1] for i in range(1, len(request_times))]
            avg_interval = sum(intervals) / len(intervals)
            
            # Detect bot-like behavior (too regular)
            if avg_interval > 0 and all(abs(interval - avg_interval) < 0.1 for interval in intervals[-10:]):
                return {
                    "type": "bot_like_timing",
                    "avg_interval": avg_interval,
                    "pattern_detected": True
                }
        
        return None
    
    async def _update_behavior_profile(self,
                                     client_ip: str,
                                     client_id: Optional[str],
                                     endpoint: str,
                                     current_time: float):
        """Update user behavior profile for anomaly detection"""
        user_key = client_id or client_ip
        
        if user_key not in self.user_behavior_profiles:
            self.user_behavior_profiles[user_key] = {
                'endpoints': {},
                'request_times': deque(maxlen=100),
                'first_seen': current_time,
                'last_seen': current_time
            }
        
        profile = self.user_behavior_profiles[user_key]
        profile['endpoints'][endpoint] = profile['endpoints'].get(endpoint, 0) + 1
        profile['request_times'].append(current_time)
        profile['last_seen'] = current_time
    
    def _calculate_threat_level(self, category: str, pattern: str) -> ThreatLevel:
        """Calculate threat level based on pattern category and specificity"""
        high_risk_categories = {'command_injection', 'path_traversal', 'music21_specific'}
        medium_risk_categories = {'sql_injection', 'xss_injection'}
        
        if category in high_risk_categories:
            return ThreatLevel.HIGH
        elif category in medium_risk_categories:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """Create hash of payload for tracking"""
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(payload_str.encode()).hexdigest()[:16]
    
    async def _log_security_event(self,
                                event_type: EventType,
                                threat_level: ThreatLevel,
                                client_ip: Optional[str] = None,
                                user_agent: Optional[str] = None,
                                client_id: Optional[str] = None,
                                endpoint: Optional[str] = None,
                                payload_hash: Optional[str] = None,
                                details: Optional[Dict[str, Any]] = None):
        """Log security event with comprehensive details"""
        event_id = hashlib.md5(f"{time.time()}{client_ip}{event_type.value}".encode()).hexdigest()
        
        event = SecurityEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=client_ip,
            user_agent=user_agent,
            client_id=client_id,
            endpoint=endpoint,
            payload_hash=payload_hash,
            details=details or {}
        )
        
        # Store event
        self.security_events.append(event)
        self.metrics.suspicious_events += 1
        
        # Log to security logger
        log_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "threat_level": event.threat_level.value,
            "source_ip": event.source_ip,
            "client_id": event.client_id,
            "endpoint": event.endpoint,
            "details": event.details
        }
        
        if threat_level == ThreatLevel.CRITICAL:
            security_logger.critical(f"CRITICAL SECURITY EVENT: {json.dumps(log_data)}")
        elif threat_level == ThreatLevel.HIGH:
            security_logger.error(f"HIGH THREAT DETECTED: {json.dumps(log_data)}")
        elif threat_level == ThreatLevel.MEDIUM:
            security_logger.warning(f"MEDIUM THREAT: {json.dumps(log_data)}")
        else:
            security_logger.info(f"Security event: {json.dumps(log_data)}")
        
        # Audit log for compliance
        audit_logger.info(json.dumps(log_data))
        
        # Update threat intelligence
        await self._update_threat_intelligence(event)
    
    async def _update_threat_intelligence(self, event: SecurityEvent):
        """Update threat intelligence database"""
        if event.source_ip:
            ip_key = event.source_ip
            if ip_key not in self.threat_intelligence:
                self.threat_intelligence[ip_key] = {
                    'first_seen': event.timestamp,
                    'last_seen': event.timestamp,
                    'threat_score': 0.0,
                    'event_count': 0,
                    'event_types': set(),
                    'blocked': False
                }
            
            intel = self.threat_intelligence[ip_key]
            intel['last_seen'] = event.timestamp
            intel['event_count'] += 1
            intel['event_types'].add(event.event_type.value)
            
            # Calculate threat score
            base_score = {
                ThreatLevel.LOW: 0.1,
                ThreatLevel.MEDIUM: 0.3,
                ThreatLevel.HIGH: 0.7,
                ThreatLevel.CRITICAL: 1.0
            }[event.threat_level]
            
            # Escalate score based on frequency
            frequency_multiplier = min(intel['event_count'] / 10.0, 2.0)
            intel['threat_score'] = min(base_score * frequency_multiplier, 1.0)
            
            # Auto-block high-threat IPs
            if intel['threat_score'] > 0.8 and not intel['blocked']:
                self.blocked_ips[event.source_ip] = time.time() + (self.rate_limit_config.lockout_duration * 2)
                intel['blocked'] = True
                security_logger.error(f"Auto-blocked high-threat IP: {event.source_ip}")
    
    def get_security_metrics(self) -> SecurityMetrics:
        """Get current security metrics"""
        # Update dynamic metrics
        self.metrics.uptime_hours = (time.time() - self.start_time) / 3600
        self.metrics.rate_limited_ips = len(self.blocked_ips)
        self.metrics.active_threats = len([
            ip for ip, data in self.threat_intelligence.items()
            if data['threat_score'] > 0.5
        ])
        
        # Calculate overall threat score
        if self.threat_intelligence:
            avg_threat_score = sum(
                data['threat_score'] for data in self.threat_intelligence.values()
            ) / len(self.threat_intelligence)
            self.metrics.threat_score = avg_threat_score
        
        return self.metrics
    
    def get_recent_events(self, limit: int = 100, threat_level: Optional[ThreatLevel] = None) -> List[SecurityEvent]:
        """Get recent security events"""
        events = list(self.security_events)
        
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_threat_intelligence_summary(self) -> Dict[str, Any]:
        """Get threat intelligence summary"""
        if not self.threat_intelligence:
            return {"total_threats": 0, "top_threats": []}
        
        # Sort by threat score
        sorted_threats = sorted(
            self.threat_intelligence.items(),
            key=lambda x: x[1]['threat_score'],
            reverse=True
        )
        
        return {
            "total_threats": len(self.threat_intelligence),
            "high_threats": len([t for t in self.threat_intelligence.values() if t['threat_score'] > 0.7]),
            "blocked_ips": len(self.blocked_ips),
            "top_threats": [
                {
                    "ip": ip,
                    "threat_score": data['threat_score'],
                    "event_count": data['event_count'],
                    "event_types": list(data['event_types']),
                    "first_seen": data['first_seen'].isoformat(),
                    "last_seen": data['last_seen'].isoformat()
                }
                for ip, data in sorted_threats[:10]
            ]
        }
    
    async def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old security data"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Clean old events
        original_count = len(self.security_events)
        self.security_events = deque(
            [e for e in self.security_events if e.timestamp > cutoff_date],
            maxlen=10000
        )
        
        # Clean old threat intelligence
        old_threats = [
            ip for ip, data in self.threat_intelligence.items()
            if data['last_seen'] < cutoff_date
        ]
        
        for ip in old_threats:
            del self.threat_intelligence[ip]
        
        # Clean expired IP blocks
        current_time = time.time()
        expired_blocks = [
            ip for ip, expire_time in self.blocked_ips.items()
            if current_time > expire_time
        ]
        
        for ip in expired_blocks:
            del self.blocked_ips[ip]
        
        security_logger.info(
            f"Cleanup completed: removed {original_count - len(self.security_events)} old events, "
            f"{len(old_threats)} old threats, {len(expired_blocks)} expired blocks"
        )


# Global security monitor instance
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


def initialize_security_monitoring(rate_limit_config: Optional[RateLimitConfig] = None,
                                 anomaly_thresholds: Optional[AnomalyThresholds] = None) -> SecurityMonitor:
    """Initialize global security monitoring"""
    global _security_monitor
    _security_monitor = SecurityMonitor(rate_limit_config, anomaly_thresholds)
    return _security_monitor


# Export main classes
__all__ = [
    'SecurityMonitor',
    'SecurityEvent',
    'SecurityMetrics',
    'ThreatLevel',
    'EventType',
    'RateLimitConfig',
    'AnomalyThresholds',
    'get_security_monitor',
    'initialize_security_monitoring'
]