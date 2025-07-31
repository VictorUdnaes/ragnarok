import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

# FastAPI imports with error handling
try:
    import uvicorn
    from fastapi import FastAPI, Request, HTTPException
    from starlette.middleware.base import BaseHTTPMiddleware  # Direct import from Starlette
    from pydantic import BaseModel
except ImportError as e:
    print(f"❌ Missing FastAPI dependencies: {e}")
    print("📦 Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Rich imports with error handling
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.traceback import install
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
except ImportError as e:
    print(f"❌ Missing Rich dependency: {e}")
    print("📦 Install with: pip install rich")
    sys.exit(1)

# Install rich traceback handler for beautiful error traces
install(show_locals=True)

class IndentedFormatter(logging.Formatter):
    """Custom formatter that handles indentation properly for messages between request start and end"""
    
    # Class variable to track if we're currently inside a request
    _inside_request = False
    
    def __init__(self, fmt):
        super().__init__(fmt)
        self.base_fmt = fmt
        # Calculate the prefix length (logger name field width + spacing)
        self.prefix_length = 26  # 25 chars for logger name + 1 space
    
    def format(self, record):
        original_message = record.getMessage()
        message = original_message.strip()
        
        # Pre-process multi-line messages to add proper indentation
        processed_message = self._preprocess_multiline_message(original_message)
        
        # Temporarily replace the record's message with our processed version
        original_msg = record.msg
        original_args = record.args
        record.msg = processed_message
        record.args = ()
        
        try:
            # Check if this is a request start (LoggingAspect with -->)
            if record.name == "LoggingAspect" and message.startswith('-->'):
                IndentedFormatter._inside_request = True
                return super().format(record)
            
            # Check if this is a request end (LoggingAspect with <--)
            elif record.name == "LoggingAspect" and message.startswith('<--'):
                IndentedFormatter._inside_request = False
                return super().format(record)
            
            # If we're inside a request, indent all messages (except empty ones)
            elif IndentedFormatter._inside_request and message:
                # Add request indentation to the processed message
                indented_message = self._add_request_indentation(processed_message)
                record.msg = indented_message
                return super().format(record)
            
            # Default formatting for everything else
            else:
                return super().format(record)
        finally:
            # Restore original record state
            record.msg = original_msg
            record.args = original_args
    
    def _preprocess_multiline_message(self, message):
        """Pre-process multi-line messages to add proper continuation indentation"""
        lines = message.split('\n')
        if len(lines) <= 1:
            return message
        
        # Keep the first line as is
        result = [lines[0]]
        
        # For continuation lines, add proper spacing to align with the message content
        continuation_indent = ' ' * self.prefix_length
        
        for line in lines[1:]:
            if line.strip():  # Only add indentation to non-empty lines
                result.append(continuation_indent + line.strip())
            else:
                result.append('')  # Keep empty lines empty
        
        return '\n'.join(result)
    
    def _add_request_indentation(self, message):
        """Add request-level indentation (4 spaces) to all lines of the message"""
        lines = message.split('\n')
        result = []
        
        for line in lines:
            if line.strip():  # Only add indentation to non-empty lines
                result.append('    ' + line)
            else:
                result.append(line)  # Keep empty lines as they are
        
        return '\n'.join(result)

class RichLoggingSetup:
    """Rich logging configuration for FastAPI"""
    
    def __init__(self):
        self.console = Console()
        self.setup_logging()
    
    def setup_logging(self):
        """Configure Rich logging to match your preferred format"""
        
        # Create Rich handler with minimal time display and custom format
        rich_handler = RichHandler(
            console=self.console,
            show_time=False,  # No timestamps as requested
            show_path=False,  # Clean format without paths
            enable_link_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            tracebacks_extra_lines=2,
            tracebacks_theme="monokai"
        )
        
        # Custom format that matches your style: LEVEL LoggerName message
        rich_handler.setFormatter(
            IndentedFormatter(
                fmt="%(name)-25s %(message)s"
            )
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
        root_logger.addHandler(rich_handler)
        
        # Configure uvicorn loggers to use Rich
        for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.addHandler(rich_handler)
            logger.propagate = False
        
        # Silence noisy loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Silence access logs
    
    def log_startup_banner(self):
        """Display a beautiful startup banner"""
        startup_panel = Panel.fit(
            "[bold green]🚀 RAG Application[/bold green]\n"
            "[blue]Starting up with Rich logging...[/blue]",
            title="[bold magenta]STARTUP[/bold magenta]",
            border_style="green"
        )
        self.console.print(startup_panel)

# Custom middleware for Rich request logging
class RichLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests with your preferred format"""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger("LoggingAspect")
        self.console = Console()
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log incoming request - clear request start
        self.logger.info("")
        self.logger.info(f"[bold blue]-->[/bold blue] [green][{request.method}][/green] [cyan]{request.url.path}[/cyan]")
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log response with your format
            if response.status_code < 300:
                status_color = "green"
            elif response.status_code < 400:
                status_color = "yellow"  
            else:
                status_color = "red"
            
            self.logger.info("")
            # Response log
            self.logger.info(f"[yellow]Response:[/yellow] [bold]<{response.status_code} OK>[/bold]")
            
            # Optional: Log response body for certain endpoints (be careful with large responses)
            # self.logger.info(f"[yellow]Response body:[/yellow] {{status: 'success'}}")
            
            # Request completion with timing
            self.logger.info(f"[bold blue]<--[/bold blue] [green][{request.method}][/green] [cyan]{request.url.path}[/cyan] [magenta]({duration_ms}ms)[/magenta] [bold][{status_color}][{response.status_code}][/{status_color}][/bold]")
            
            return response
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log error response
            self.logger.error(f"[yellow]Response:[/yellow] [bold red]<500 ERROR>[/bold red]")
            self.logger.error(f"[yellow]Response body:[/yellow] {{error: '{str(e)}'")
            self.logger.error(f"[bold blue]<--[/bold blue] [green][{request.method}][/green] [cyan]{request.url.path}[/cyan] [magenta]({duration_ms}ms)[/magenta] [bold red][500 ERROR][/bold red]")
            
            raise
