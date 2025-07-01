#!/usr/bin/env python3
"""
Real-time monitoring dashboard for production stress test
Shows metrics, alerts, and system health in a terminal UI
"""
import asyncio
import curses
import json
import logging
import os
import platform
import psutil
import socket
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Configure logging to file only (not console)
log_dir = Path("stress_test_logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=log_dir / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics from various sources"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.metrics_history = {
            'requests_per_second': deque(maxlen=300),  # 5 minutes
            'error_rate': deque(maxlen=300),
            'memory_usage': deque(maxlen=300),
            'response_time': deque(maxlen=300),
            'cpu_usage': deque(maxlen=300),
        }
        self.alerts = deque(maxlen=100)
        self.last_metrics = {}
        
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'server': await self._get_server_metrics(),
            'system': self._get_system_metrics(),
            'network': await self._get_network_metrics(),
        }
        
        # Update history
        if metrics['server']:
            rps = metrics['server'].get('requests_per_second', 0)
            self.metrics_history['requests_per_second'].append(rps)
            
            error_rate = metrics['server'].get('error_rate', 0)
            self.metrics_history['error_rate'].append(error_rate)
            
            response_time = metrics['server'].get('avg_response_time', 0)
            self.metrics_history['response_time'].append(response_time)
        
        self.metrics_history['memory_usage'].append(metrics['system']['memory_percent'])
        self.metrics_history['cpu_usage'].append(metrics['system']['cpu_percent'])
        
        # Check for alerts
        self._check_alerts(metrics)
        
        self.last_metrics = metrics
        return metrics
    
    async def _get_server_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from server"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get health check
                health_response = await client.get(f"{self.server_url}/health/detailed")
                health_data = health_response.json()
                
                # Get metrics
                metrics_response = await client.get(f"{self.server_url}/metrics")
                metrics_text = metrics_response.text
                
                # Parse prometheus metrics
                parsed_metrics = self._parse_prometheus_metrics(metrics_text)
                
                # Calculate derived metrics
                total_requests = parsed_metrics.get('music21_requests_total', 0)
                failed_requests = parsed_metrics.get('music21_requests_failed', 0)
                
                # Estimate RPS (requests per second)
                if hasattr(self, '_last_total_requests'):
                    time_diff = 1.0  # Collection interval
                    rps = (total_requests - self._last_total_requests) / time_diff
                else:
                    rps = 0
                self._last_total_requests = total_requests
                
                return {
                    'status': health_data.get('status', 'unknown'),
                    'uptime': health_data.get('uptime_seconds', 0),
                    'total_requests': total_requests,
                    'failed_requests': failed_requests,
                    'success_rate': ((total_requests - failed_requests) / total_requests * 100) if total_requests > 0 else 0,
                    'requests_per_second': rps,
                    'error_rate': failed_requests / total_requests * 100 if total_requests > 0 else 0,
                    'memory_mb': parsed_metrics.get('music21_memory_usage_mb', 0),
                    'active_requests': health_data.get('metrics', {}).get('active_requests', 0),
                    'circuit_breakers': health_data.get('circuit_breakers', {}),
                    'health_checks': health_data.get('checks', {}),
                }
                
        except Exception as e:
            logger.error(f"Failed to get server metrics: {e}")
            return None
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus format metrics"""
        metrics = {}
        
        for line in metrics_text.split('\n'):
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0].split('{')[0]  # Remove labels
                    try:
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                    except:
                        pass
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # Process specific metrics
        try:
            # Find music21 server process
            server_process = None
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'music21' in str(proc.info.get('cmdline', [])):
                    server_process = proc
                    break
            
            if server_process:
                proc_info = {
                    'pid': server_process.pid,
                    'cpu_percent': server_process.cpu_percent(),
                    'memory_mb': server_process.memory_info().rss / 1024 / 1024,
                    'num_threads': server_process.num_threads(),
                    'num_fds': len(server_process.open_files()),
                }
            else:
                proc_info = None
                
        except:
            proc_info = None
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'memory_total_gb': memory.total / 1024 / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / 1024 / 1024 / 1024,
            'network_sent_mb': net_io.bytes_sent / 1024 / 1024,
            'network_recv_mb': net_io.bytes_recv / 1024 / 1024,
            'server_process': proc_info,
        }
    
    async def _get_network_metrics(self) -> Dict[str, Any]:
        """Test network connectivity"""
        metrics = {}
        
        # Test server port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            metrics['server_port_open'] = result == 0
        except:
            metrics['server_port_open'] = False
        
        # Test server response time
        try:
            start = time.time()
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.get(f"{self.server_url}/health")
            metrics['health_check_ms'] = (time.time() - start) * 1000
        except:
            metrics['health_check_ms'] = -1
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        
        # Server down
        if not metrics['server']:
            self.alerts.append({
                'time': datetime.now(),
                'level': 'CRITICAL',
                'message': 'Server is not responding'
            })
        
        elif metrics['server']:
            # High error rate
            if metrics['server']['error_rate'] > 5:
                self.alerts.append({
                    'time': datetime.now(),
                    'level': 'WARNING',
                    'message': f"High error rate: {metrics['server']['error_rate']:.1f}%"
                })
            
            # Circuit breakers open
            for name, breaker in metrics['server'].get('circuit_breakers', {}).items():
                if breaker.get('state') == 'open':
                    self.alerts.append({
                        'time': datetime.now(),
                        'level': 'WARNING',
                        'message': f"Circuit breaker open: {name}"
                    })
            
            # Memory pressure
            if metrics['server']['memory_mb'] > 3000:
                self.alerts.append({
                    'time': datetime.now(),
                    'level': 'WARNING',
                    'message': f"High memory usage: {metrics['server']['memory_mb']:.0f}MB"
                })
        
        # System alerts
        if metrics['system']['cpu_percent'] > 90:
            self.alerts.append({
                'time': datetime.now(),
                'level': 'WARNING',
                'message': f"High CPU usage: {metrics['system']['cpu_percent']:.1f}%"
            })
        
        if metrics['system']['memory_percent'] > 90:
            self.alerts.append({
                'time': datetime.now(),
                'level': 'CRITICAL',
                'message': f"Critical memory usage: {metrics['system']['memory_percent']:.1f}%"
            })


class TerminalDashboard:
    """Terminal UI for monitoring"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.stdscr = None
        self.running = True
        
    def start(self):
        """Start the dashboard"""
        curses.wrapper(self.run)
    
    def run(self, stdscr):
        """Main dashboard loop"""
        self.stdscr = stdscr
        
        # Configure curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100) # Refresh rate
        
        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Good
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # Error
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Info
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        
        # Main loop
        while self.running:
            try:
                # Get latest metrics
                metrics = asyncio.run(self.collector.collect_metrics())
                
                # Clear screen
                stdscr.clear()
                
                # Draw dashboard
                self.draw_header(metrics)
                self.draw_server_metrics(metrics, start_row=3)
                self.draw_system_metrics(metrics, start_row=12)
                self.draw_alerts(start_row=20)
                self.draw_graphs(start_row=30)
                
                # Refresh
                stdscr.refresh()
                
                # Check for quit
                key = stdscr.getch()
                if key == ord('q'):
                    self.running = False
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                time.sleep(1)
    
    def draw_header(self, metrics: Dict[str, Any]):
        """Draw header"""
        height, width = self.stdscr.getmaxyx()
        
        # Title
        title = "Music21 MCP Server - Production Stress Test Monitor"
        self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.addstr(0, (width - len(title)) // 2, title.center(width))
        self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stdscr.addstr(1, width - len(timestamp) - 1, timestamp)
        
        # Instructions
        self.stdscr.addstr(height - 1, 0, "Press 'q' to quit")
    
    def draw_server_metrics(self, metrics: Dict[str, Any], start_row: int):
        """Draw server metrics"""
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_row, 0, "SERVER METRICS")
        self.stdscr.attroff(curses.A_BOLD)
        
        row = start_row + 1
        
        if not metrics['server']:
            self.stdscr.attron(curses.color_pair(3))
            self.stdscr.addstr(row, 2, "SERVER OFFLINE")
            self.stdscr.attroff(curses.color_pair(3))
            return
        
        server = metrics['server']
        
        # Status
        status_color = {
            'healthy': 1,
            'degraded': 2,
            'unhealthy': 3
        }.get(server['status'], 3)
        
        self.stdscr.addstr(row, 2, f"Status: ")
        self.stdscr.attron(curses.color_pair(status_color))
        self.stdscr.addstr(server['status'].upper())
        self.stdscr.attroff(curses.color_pair(status_color))
        
        # Uptime
        uptime = timedelta(seconds=int(server['uptime']))
        self.stdscr.addstr(row, 30, f"Uptime: {uptime}")
        row += 1
        
        # Request metrics
        self.stdscr.addstr(row, 2, f"Total Requests: {server['total_requests']:,}")
        self.stdscr.addstr(row, 30, f"RPS: {server['requests_per_second']:.1f}")
        self.stdscr.addstr(row, 50, f"Active: {server['active_requests']}")
        row += 1
        
        # Success rate
        success_color = 1 if server['success_rate'] >= 95 else 2 if server['success_rate'] >= 90 else 3
        self.stdscr.addstr(row, 2, "Success Rate: ")
        self.stdscr.attron(curses.color_pair(success_color))
        self.stdscr.addstr(f"{server['success_rate']:.1f}%")
        self.stdscr.attroff(curses.color_pair(success_color))
        
        # Error rate
        self.stdscr.addstr(row, 30, f"Errors: {server['failed_requests']:,}")
        row += 1
        
        # Memory
        mem_color = 1 if server['memory_mb'] < 2000 else 2 if server['memory_mb'] < 3000 else 3
        self.stdscr.addstr(row, 2, "Memory: ")
        self.stdscr.attron(curses.color_pair(mem_color))
        self.stdscr.addstr(f"{server['memory_mb']:.0f}MB")
        self.stdscr.attroff(curses.color_pair(mem_color))
        row += 1
        
        # Circuit breakers
        open_breakers = [name for name, b in server['circuit_breakers'].items() if b['state'] == 'open']
        if open_breakers:
            self.stdscr.attron(curses.color_pair(3))
            self.stdscr.addstr(row, 2, f"Open Circuit Breakers: {', '.join(open_breakers)}")
            self.stdscr.attroff(curses.color_pair(3))
            row += 1
    
    def draw_system_metrics(self, metrics: Dict[str, Any], start_row: int):
        """Draw system metrics"""
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_row, 0, "SYSTEM METRICS")
        self.stdscr.attroff(curses.A_BOLD)
        
        row = start_row + 1
        system = metrics['system']
        
        # CPU
        cpu_color = 1 if system['cpu_percent'] < 70 else 2 if system['cpu_percent'] < 90 else 3
        self.stdscr.addstr(row, 2, "CPU: ")
        self.stdscr.attron(curses.color_pair(cpu_color))
        self.stdscr.addstr(f"{system['cpu_percent']:.1f}%")
        self.stdscr.attroff(curses.color_pair(cpu_color))
        self.stdscr.addstr(f" ({system['cpu_count']} cores)")
        row += 1
        
        # Memory
        mem_color = 1 if system['memory_percent'] < 70 else 2 if system['memory_percent'] < 90 else 3
        self.stdscr.addstr(row, 2, "Memory: ")
        self.stdscr.attron(curses.color_pair(mem_color))
        self.stdscr.addstr(f"{system['memory_percent']:.1f}%")
        self.stdscr.attroff(curses.color_pair(mem_color))
        self.stdscr.addstr(f" ({system['memory_available_gb']:.1f}GB free)")
        row += 1
        
        # Disk
        self.stdscr.addstr(row, 2, f"Disk: {system['disk_percent']:.1f}% ({system['disk_free_gb']:.1f}GB free)")
        row += 1
        
        # Network
        self.stdscr.addstr(row, 2, f"Network - Sent: {system['network_sent_mb']:.1f}MB, Recv: {system['network_recv_mb']:.1f}MB")
        row += 1
        
        # Server process
        if system['server_process']:
            proc = system['server_process']
            self.stdscr.addstr(row, 2, f"Server Process - PID: {proc['pid']}, "
                              f"CPU: {proc['cpu_percent']:.1f}%, "
                              f"Memory: {proc['memory_mb']:.0f}MB, "
                              f"Threads: {proc['num_threads']}, "
                              f"FDs: {proc['num_fds']}")
    
    def draw_alerts(self, start_row: int):
        """Draw recent alerts"""
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_row, 0, "RECENT ALERTS")
        self.stdscr.attroff(curses.A_BOLD)
        
        row = start_row + 1
        height, width = self.stdscr.getmaxyx()
        max_alerts = min(5, height - row - 10)  # Leave space for graphs
        
        if not self.collector.alerts:
            self.stdscr.addstr(row, 2, "No alerts")
            return
        
        # Show most recent alerts
        recent_alerts = list(self.collector.alerts)[-max_alerts:]
        
        for alert in recent_alerts:
            time_str = alert['time'].strftime("%H:%M:%S")
            level = alert['level']
            message = alert['message']
            
            color = 2 if level == 'WARNING' else 3
            
            self.stdscr.addstr(row, 2, f"{time_str} ")
            self.stdscr.attron(curses.color_pair(color))
            self.stdscr.addstr(f"[{level}]")
            self.stdscr.attroff(curses.color_pair(color))
            self.stdscr.addstr(f" {message[:width-20]}")
            row += 1
    
    def draw_graphs(self, start_row: int):
        """Draw mini graphs"""
        height, width = self.stdscr.getmaxyx()
        
        if start_row + 10 > height:
            return  # Not enough space
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_row, 0, "METRICS HISTORY (5 min)")
        self.stdscr.attroff(curses.A_BOLD)
        
        # Draw RPS graph
        self._draw_mini_graph(
            "RPS",
            self.collector.metrics_history['requests_per_second'],
            start_row + 2,
            0,
            width // 2 - 2,
            5
        )
        
        # Draw Memory graph
        self._draw_mini_graph(
            "Memory %",
            self.collector.metrics_history['memory_usage'],
            start_row + 2,
            width // 2,
            width // 2 - 2,
            5,
            max_val=100
        )
    
    def _draw_mini_graph(self, label: str, data: deque, row: int, col: int, 
                        width: int, height: int, max_val: Optional[float] = None):
        """Draw a mini ASCII graph"""
        if not data:
            return
        
        # Calculate scale
        if max_val is None:
            max_val = max(data) if data else 1
        
        if max_val == 0:
            max_val = 1
        
        # Draw label
        self.stdscr.addstr(row, col, f"{label}: ")
        
        # Draw graph
        graph_width = min(len(data), width - len(label) - 3)
        graph_data = list(data)[-graph_width:]
        
        for h in range(height):
            graph_row = row + height - h - 1
            self.stdscr.addstr(graph_row, col + len(label) + 2, "")
            
            for i, value in enumerate(graph_data):
                graph_col = col + len(label) + 3 + i
                
                # Calculate if this height should be filled
                normalized = value / max_val * height
                
                if normalized >= height - h:
                    self.stdscr.addstr(graph_row, graph_col, "█")
                else:
                    self.stdscr.addstr(graph_row, graph_col, " ")
        
        # Show current value
        if data:
            current = data[-1]
            self.stdscr.addstr(row + height, col + len(label) + 2, f"{current:.1f}")


async def run_monitor():
    """Run the monitoring dashboard"""
    collector = MetricsCollector()
    dashboard = TerminalDashboard(collector)
    
    print("Starting Music21 MCP Server Monitor...")
    print("Connecting to server...")
    
    # Wait for initial connection
    for i in range(10):
        metrics = await collector.collect_metrics()
        if metrics['server']:
            break
        print(f"Waiting for server... ({i+1}/10)")
        await asyncio.sleep(1)
    
    # Start dashboard
    dashboard.start()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Music21 MCP Server stress test')
    parser.add_argument('--server', default='http://localhost:8000', help='Server URL')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Run monitor
    try:
        asyncio.run(run_monitor())
    except KeyboardInterrupt:
        print("\nMonitor stopped")
    except Exception as e:
        print(f"Monitor error: {e}")
        raise


if __name__ == "__main__":
    main()