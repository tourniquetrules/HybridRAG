#!/bin/bash

# DoclingAgent Service Management Script
# This script helps install, configure, and manage the DoclingAgent systemd service

set -e

SERVICE_NAME="doclingagent"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOCAL_SERVICE_FILE="$(dirname "$0")/doclingagent.service"
APP_DIR="$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root for system operations
check_root() {
    if [[ $EUID -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

# Install the service
install_service() {
    print_status "Installing DoclingAgent service..."
    
    if ! check_root; then
        print_error "Installing system service requires root privileges"
        echo "Please run: sudo $0 install"
        exit 1
    fi
    
    # Copy service file to systemd directory
    cp "$LOCAL_SERVICE_FILE" "$SERVICE_FILE"
    print_status "Service file copied to $SERVICE_FILE"
    
    # Reload systemd daemon
    systemctl daemon-reload
    print_status "Systemd daemon reloaded"
    
    # Enable the service to start on boot
    systemctl enable "$SERVICE_NAME"
    print_status "Service enabled to start on boot"
    
    print_status "Service installation completed successfully!"
    echo "Use 'sudo systemctl start doclingagent' to start the service"
}

# Remove the service
uninstall_service() {
    print_status "Uninstalling DoclingAgent service..."
    
    if ! check_root; then
        print_error "Uninstalling system service requires root privileges"
        echo "Please run: sudo $0 uninstall"
        exit 1
    fi
    
    # Stop the service if running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
        print_status "Service stopped"
    fi
    
    # Disable the service
    if systemctl is-enabled --quiet "$SERVICE_NAME"; then
        systemctl disable "$SERVICE_NAME"
        print_status "Service disabled"
    fi
    
    # Remove service file
    if [[ -f "$SERVICE_FILE" ]]; then
        rm "$SERVICE_FILE"
        print_status "Service file removed"
    fi
    
    # Reload systemd daemon
    systemctl daemon-reload
    print_status "Systemd daemon reloaded"
    
    print_status "Service uninstallation completed successfully!"
}

# Start the service
start_service() {
    if ! check_root; then
        print_error "Managing system service requires root privileges"
        echo "Please run: sudo $0 start"
        exit 1
    fi
    
    systemctl start "$SERVICE_NAME"
    print_status "Service started"
}

# Stop the service
stop_service() {
    if ! check_root; then
        print_error "Managing system service requires root privileges"
        echo "Please run: sudo $0 stop"
        exit 1
    fi
    
    systemctl stop "$SERVICE_NAME"
    print_status "Service stopped"
}

# Restart the service
restart_service() {
    if ! check_root; then
        print_error "Managing system service requires root privileges"
        echo "Please run: sudo $0 restart"
        exit 1
    fi
    
    systemctl restart "$SERVICE_NAME"
    print_status "Service restarted"
}

# Show service status
status_service() {
    print_status "DoclingAgent service status:"
    systemctl status "$SERVICE_NAME" --no-pager
}

# Show service logs
logs_service() {
    print_status "DoclingAgent service logs:"
    if [[ "$2" == "-f" ]] || [[ "$2" == "--follow" ]]; then
        journalctl -u "$SERVICE_NAME" -f
    else
        journalctl -u "$SERVICE_NAME" -n 50 --no-pager
    fi
}

# Show help
show_help() {
    echo "DoclingAgent Service Management Script"
    echo "Usage: $0 {install|uninstall|start|stop|restart|status|logs|help}"
    echo ""
    echo "Commands:"
    echo "  install     Install the systemd service (requires root)"
    echo "  uninstall   Remove the systemd service (requires root)"
    echo "  start       Start the service (requires root)"
    echo "  stop        Stop the service (requires root)"
    echo "  restart     Restart the service (requires root)"
    echo "  status      Show service status"
    echo "  logs        Show recent service logs"
    echo "  logs -f     Follow service logs in real-time"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  sudo $0 install    # Install and enable the service"
    echo "  sudo $0 start      # Start the service"
    echo "  $0 status          # Check service status"
    echo "  $0 logs -f         # Follow logs in real-time"
}

# Main script logic
case "${1:-}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        logs_service "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: ${1:-}"
        echo ""
        show_help
        exit 1
        ;;
esac