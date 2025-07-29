#!/usr/bin/env python3
"""Production deployment script."""

import sys
import subprocess
import shutil
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Handles production deployment."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
    
    def check_requirements(self) -> bool:
        """Check if all requirements are met."""
        logger.info("Checking deployment requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
        
        # Check Ollama
        try:
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
            logger.info("✅ Ollama installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Ollama not installed or not in PATH")
            return False
        
        # Check required directories
        required_dirs = [
            settings.DATA_DIR,
            settings.RAW_DATA_DIR,
            settings.PROCESSED_DATA_DIR,
            settings.MODELS_DIR
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                logger.info(f"Creating directory: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def setup_environment(self):
        """Set up production environment."""
        logger.info("Setting up production environment...")
        
        # Install dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", 
            str(self.project_root / "requirements.txt")
        ], check=True)
        
        # Setup Ollama models
        subprocess.run([
            "bash", str(self.project_root / "scripts" / "setup_ollama.sh")
        ], check=True)
        
        logger.info("✅ Environment setup complete")
    
    def create_systemd_services(self):
        """Create systemd service files for production."""
        
        mcp_service = f"""[Unit]
Description=Personal LLM Chatbot MCP Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/venv/bin
ExecStart={self.project_root}/venv/bin/python scripts/run_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        streamlit_service = f"""[Unit]
Description=Personal LLM Chatbot Streamlit UI
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/venv/bin
ExecStart={self.project_root}/venv/bin/streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        # Write service files
        service_dir = Path("/etc/systemd/system")
        if service_dir.exists():
            with open(service_dir / "llm-chatbot-mcp.service", "w") as f:
                f.write(mcp_service)
            
            with open(service_dir / "llm-chatbot-ui.service", "w") as f:
                f.write(streamlit_service)
            
            logger.info("✅ Systemd service files created")
        else:
            logger.warning("⚠️  Systemd not available, skipping service creation")
    
    def create_nginx_config(self):
        """Create nginx reverse proxy configuration."""
        
        nginx_config = """server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain
    
    # MCP API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Streamlit UI
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
"""
        
        config_file = self.project_root / "nginx.conf"
        with open(config_file, "w") as f:
            f.write(nginx_config)
        
        logger.info(f"✅ Nginx config created at {config_file}")
        logger.info("   Copy to /etc/nginx/sites-available/ and enable")
    
    def deploy(self):
        """Run complete deployment."""
        logger.info("🚀 Starting production deployment...")
        
        if not self.check_requirements():
            logger.error("❌ Requirements check failed")
            return False
        
        try:
            self.setup_environment()
            self.create_systemd_services()
            self.create_nginx_config()
            
            logger.info("✅ Deployment complete!")
            logger.info("Next steps:")
            logger.info("1. Copy nginx config to /etc/nginx/sites-available/")
            logger.info("2. Enable and start systemd services")
            logger.info("3. Configure SSL certificate")
            logger.info("4. Add your documents to data/raw/")
            logger.info("5. Run python scripts/ingest_data.py")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False

if __name__ == "__main__":
    deployer = ProductionDeployer()
    success = deployer.deploy()
    sys.exit(0 if success else 1)
