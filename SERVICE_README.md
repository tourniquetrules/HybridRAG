# DoclingAgent System Service

This directory contains configuration files and scripts to run the DoclingAgent Emergency Medicine RAG System as a Linux systemd service.

## Files

- `doclingagent.service` - Systemd service configuration file
- `service-manager.sh` - Management script for installing and controlling the service

## Quick Setup

1. **Install the service:**
   ```bash
   sudo ./service-manager.sh install
   ```

2. **Start the service:**
   ```bash
   sudo ./service-manager.sh start
   ```

3. **Check status:**
   ```bash
   ./service-manager.sh status
   ```

## Service Management Commands

### Installation
```bash
# Install and enable the service (requires root)
sudo ./service-manager.sh install

# Uninstall the service (requires root)
sudo ./service-manager.sh uninstall
```

### Control
```bash
# Start the service
sudo ./service-manager.sh start

# Stop the service
sudo ./service-manager.sh stop

# Restart the service
sudo ./service-manager.sh restart
```

### Monitoring
```bash
# Check service status
./service-manager.sh status

# View recent logs
./service-manager.sh logs

# Follow logs in real-time
./service-manager.sh logs -f
```

### Direct systemctl commands
You can also use standard systemctl commands:
```bash
# Start/stop/restart
sudo systemctl start doclingagent
sudo systemctl stop doclingagent
sudo systemctl restart doclingagent

# Enable/disable auto-start on boot
sudo systemctl enable doclingagent
sudo systemctl disable doclingagent

# Check status
systemctl status doclingagent

# View logs
journalctl -u doclingagent -f
```

## Service Configuration

The service is configured to:

- **Auto-start on boot** after network is available
- **Auto-restart** if the application crashes (10-second delay)
- **Run as user** `tourniquetrules` (not root for security)
- **Log output** to `/home/tourniquetrules/doclingagent/service.log`
- **Resource limits**: 4GB memory, 200% CPU quota
- **Security hardening**: No new privileges, private tmp, protected system

## Application Access

Once the service is running, the FastAPI application will be available at:
- **HTTP**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/health

## Log Files

The service creates several log files:

1. **Service logs**: `/home/tourniquetrules/doclingagent/service.log`
   - Contains stdout/stderr from the service
2. **Application logs**: `/home/tourniquetrules/doclingagent/server.log`
   - Application-specific logging
3. **System journal**: `journalctl -u doclingagent`
   - Systemd service management logs

## Troubleshooting

### Service won't start
```bash
# Check service status for error messages
./service-manager.sh status

# Check recent logs
./service-manager.sh logs

# Check if port is already in use
sudo netstat -tlnp | grep :5000
```

### Permission issues
```bash
# Ensure correct ownership of application directory
sudo chown -R tourniquetrules:tourniquetrules /home/tourniquetrules/doclingagent

# Check service file permissions
ls -la /etc/systemd/system/doclingagent.service
```

### Virtual environment issues
```bash
# Verify virtual environment exists and has dependencies
ls -la /home/tourniquetrules/doclingagent/venv_py310/bin/python

# Test manual activation
cd /home/tourniquetrules/doclingagent
source venv_py310/bin/activate
python -c "import fastapi; print('FastAPI available')"
```

### Resource monitoring
```bash
# Monitor service resource usage
systemctl status doclingagent

# Check memory usage
ps aux | grep fastapi_app.py

# Check system resources
htop
```

## Security Considerations

The service is configured with security hardening:

- Runs as non-root user
- No new privileges allowed
- Private temporary directory
- Protected system files (read-only)
- Resource limits to prevent resource exhaustion

## Updating the Service

When you update the application code:

1. **For code changes (no service file changes):**
   ```bash
   sudo ./service-manager.sh restart
   ```

2. **For service configuration changes:**
   ```bash
   sudo ./service-manager.sh uninstall
   sudo ./service-manager.sh install
   sudo ./service-manager.sh start
   ```

## Performance Tuning

The service is configured with resource limits that can be adjusted in `doclingagent.service`:

- **MemoryLimit**: Currently set to 4G
- **CPUQuota**: Currently set to 200% (2 CPU cores max)

To modify these limits:
1. Edit `doclingagent.service`
2. Run `sudo ./service-manager.sh uninstall && sudo ./service-manager.sh install`
3. Run `sudo ./service-manager.sh start`