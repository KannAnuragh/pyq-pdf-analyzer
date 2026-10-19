#!/usr/bin/env bash
# ==============================================================================
# PDF QA Agent - Bare-Metal / Native Production Deployment Script
# ==============================================================================
# Usage:
#   ./deploy.sh build          - Install dependencies & build production assets
#   ./deploy.sh start          - Start backend and frontend services in background
#   ./deploy.sh stop           - Stop all running services
#   ./deploy.sh restart        - Restart all services
#   ./deploy.sh status         - Check status of running services
#   ./deploy.sh logs [service] - Tail logs (e.g. ./deploy.sh logs backend)
#   ./deploy.sh install-systemd- Generate & install systemd services for Linux VM
# ==============================================================================

set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"
FRONTEND_DIR="$APP_DIR/frontend"
DATA_DIR="$APP_DIR/data"
LOGS_DIR="$DATA_DIR/logs"
RUN_DIR="$DATA_DIR/run"

BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
BACKEND_LOG="$LOGS_DIR/backend.log"
FRONTEND_LOG="$LOGS_DIR/frontend.log"

BACKEND_PORT="${PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

mkdir -p "$LOGS_DIR" "$RUN_DIR" "$DATA_DIR/uploads"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Verifying host system prerequisites..."
    
    # 1. Python
    if ! command -v python3 &>/dev/null; then
        log_error "python3 is not installed. Please install Python 3.10+."
        exit 1
    fi

    # 2. Node & npm
    if ! command -v node &>/dev/null; then
        log_error "Node.js is not installed. Please install Node.js 18+ (e.g., via nvm or package manager)."
        exit 1
    fi
    if ! command -v npm &>/dev/null; then
        log_error "npm is not installed."
        exit 1
    fi

    # 3. Tesseract OCR
    if ! command -v tesseract &>/dev/null; then
        log_warn "Tesseract OCR binary not found in PATH."
        log_warn "Install via: sudo apt install tesseract-ocr tesseract-ocr-eng (Debian/Ubuntu) or pacman -S tesseract (Arch)"
    fi

    # 4. Poppler (pdftoppm)
    if ! command -v pdftoppm &>/dev/null; then
        log_warn "pdftoppm (poppler-utils) not found. PDF to image conversion may fail."
        log_warn "Install via: sudo apt install poppler-utils (Debian/Ubuntu) or pacman -S poppler (Arch)"
    fi

    log_success "Prerequisites check completed."
}

is_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# ------------------------------------------------------------------------------
# Build / Setup
# ------------------------------------------------------------------------------
build() {
    check_dependencies

    # 1. Python virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating Python virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi

    log_info "Installing/updating Python dependencies..."
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    if [ -f "$APP_DIR/requirements.txt" ]; then
        "$VENV_DIR/bin/python" -m pip install -r "$APP_DIR/requirements.txt"
    fi

    # 2. Database initialization
    log_info "Initializing database schema..."
    PYTHONPATH="$APP_DIR/.." "$VENV_DIR/bin/python" -c "
from pdf_qa_agent.storage.database import init_db
init_db()
print('Database initialized successfully.')
"

    # 3. Frontend Next.js build
    log_info "Building Next.js frontend production bundle..."
    cd "$FRONTEND_DIR"
    if [ ! -d "node_modules" ]; then
        npm ci || npm install
    fi
    npm run build

    # Standalone output requires static/ and public/ copied into .next/standalone/
    if [ -d ".next/standalone" ]; then
        log_info "Synchronizing static assets into standalone directory..."
        mkdir -p .next/standalone/.next
        cp -r .next/static .next/standalone/.next/static
        if [ -d "public" ]; then
            cp -r public .next/standalone/public
        fi
    fi
    cd "$APP_DIR"

    log_success "Build completed successfully!"
}

# ------------------------------------------------------------------------------
# Start Services
# ------------------------------------------------------------------------------
start() {
    # Check if build has run
    if [ ! -d "$VENV_DIR" ] || [ ! -d "$FRONTEND_DIR/.next" ]; then
        log_warn "Application not built yet. Running build first..."
        build
    fi

    # 1. Start Backend
    if is_running "$BACKEND_PID_FILE"; then
        log_warn "Backend is already running (PID: $(cat "$BACKEND_PID_FILE"))."
    else
        log_info "Starting Backend on port $BACKEND_PORT..."
        cd "$APP_DIR/.."
        PYTHONPATH="$APP_DIR/.." setsid "$VENV_DIR/bin/python" -m uvicorn pdf_qa_agent.api.main:app \
            --host 0.0.0.0 \
            --port "$BACKEND_PORT" \
            </dev/null >> "$BACKEND_LOG" 2>&1 &
        local b_pid=$!
        disown "$b_pid" 2>/dev/null || true
        echo "$b_pid" > "$BACKEND_PID_FILE"
        cd "$APP_DIR"

        # Wait up to 5s for backend to become ready
        local started=false
        for _ in {1..10}; do
            if kill -0 "$b_pid" 2>/dev/null; then
                if curl -s "http://127.0.0.1:$BACKEND_PORT/" >/dev/null 2>&1; then
                    started=true
                    break
                fi
            else
                break
            fi
            sleep 0.5
        done

        if [ "$started" = true ]; then
            log_success "Backend started (PID: $b_pid) -> http://localhost:$BACKEND_PORT"
        else
            if kill -0 "$b_pid" 2>/dev/null; then
                log_success "Backend started (PID: $b_pid). Initializing background workers..."
            else
                log_error "Backend failed to start. Inspect logs: $BACKEND_LOG"
                tail -n 20 "$BACKEND_LOG"
                exit 1
            fi
        fi
    fi

    # 2. Start Frontend
    if is_running "$FRONTEND_PID_FILE"; then
        log_warn "Frontend is already running (PID: $(cat "$FRONTEND_PID_FILE"))."
    else
        log_info "Starting Frontend Dashboard on port $FRONTEND_PORT..."
        cd "$FRONTEND_DIR"
        
        # Ensure standalone directory has required static and public assets
        if [ -d ".next/standalone" ]; then
            mkdir -p .next/standalone/.next
            cp -r .next/static .next/standalone/.next/static 2>/dev/null || true
            [ -d "public" ] && cp -r public .next/standalone/public 2>/dev/null || true
        fi

        # Start Next.js standalone server
        PORT="$FRONTEND_PORT" HOSTNAME="0.0.0.0" BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" \
            setsid node .next/standalone/server.js \
            </dev/null >> "$FRONTEND_LOG" 2>&1 &

        local f_pid=$!
        disown "$f_pid" 2>/dev/null || true
        echo "$f_pid" > "$FRONTEND_PID_FILE"
        cd "$APP_DIR"

        # Wait up to 5s for frontend
        local f_started=false
        for _ in {1..10}; do
            if kill -0 "$f_pid" 2>/dev/null; then
                if curl -s "http://127.0.0.1:$FRONTEND_PORT/" >/dev/null 2>&1; then
                    f_started=true
                    break
                fi
            else
                break
            fi
            sleep 0.5
        done

        if [ "$f_started" = true ]; then
            log_success "Frontend Dashboard started (PID: $f_pid) -> http://localhost:$FRONTEND_PORT"
        else
            if kill -0 "$f_pid" 2>/dev/null; then
                log_success "Frontend starting up (PID: $f_pid)..."
            else
                log_error "Frontend failed to start. Inspect logs: $FRONTEND_LOG"
                tail -n 20 "$FRONTEND_LOG"
                exit 1
            fi
        fi
    fi

    echo ""
    log_success "All services are running smoothly!"
    echo "  - Web Dashboard:  http://localhost:$FRONTEND_PORT"
    echo "  - Backend API:    http://localhost:$BACKEND_PORT"
    echo "  - API Docs:       http://localhost:$BACKEND_PORT/docs"
    echo ""
}

# ------------------------------------------------------------------------------
# Stop Services
# ------------------------------------------------------------------------------
stop_process() {
    local name="$1"
    local pid_file="$2"

    if is_running "$pid_file"; then
        local pid
        pid=$(cat "$pid_file")
        log_info "Stopping $name (PID: $pid)..."
        kill "$pid" 2>/dev/null || true

        # Wait up to 10 seconds for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 20 ]; do
            sleep 0.5
            count=$((count + 1))
        done

        if kill -0 "$pid" 2>/dev/null; then
            log_warn "$name did not terminate gracefully; force killing (SIGKILL)..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        log_success "$name stopped."
    else
        log_info "$name is not running."
    fi
    rm -f "$pid_file"
}

stop() {
    stop_process "Frontend" "$FRONTEND_PID_FILE"
    stop_process "Backend" "$BACKEND_PID_FILE"
}

# ------------------------------------------------------------------------------
# Service Status
# ------------------------------------------------------------------------------
status() {
    echo "======================================================================"
    echo " PDF QA Agent - Service Status"
    echo "======================================================================"

    # Backend
    if is_running "$BACKEND_PID_FILE"; then
        local b_pid
        b_pid=$(cat "$BACKEND_PID_FILE")
        local b_mem
        b_mem=$(ps -o rss= -p "$b_pid" 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
        local b_cpu
        b_cpu=$(ps -o %cpu= -p "$b_pid" 2>/dev/null | tr -d ' ')
        echo -e " Backend (FastAPI):  ${GREEN}RUNNING${NC} (PID: $b_pid | CPU: ${b_cpu}% | RAM: $b_mem | Port: $BACKEND_PORT)"
    else
        echo -e " Backend (FastAPI):  ${RED}STOPPED${NC}"
    fi

    # Frontend
    if is_running "$FRONTEND_PID_FILE"; then
        local f_pid
        f_pid=$(cat "$FRONTEND_PID_FILE")
        local f_mem
        f_mem=$(ps -o rss= -p "$f_pid" 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
        local f_cpu
        f_cpu=$(ps -o %cpu= -p "$f_pid" 2>/dev/null | tr -d ' ')
        echo -e " Frontend (Next.js): ${GREEN}RUNNING${NC} (PID: $f_pid | CPU: ${f_cpu}% | RAM: $f_mem | Port: $FRONTEND_PORT)"
    else
        echo -e " Frontend (Next.js): ${RED}STOPPED${NC}"
    fi

    echo "======================================================================"
    echo " Logs:"
    echo "   Backend:  $BACKEND_LOG"
    echo "   Frontend: $FRONTEND_LOG"
    echo "======================================================================"
}

# ------------------------------------------------------------------------------
# Tail Logs
# ------------------------------------------------------------------------------
logs() {
    local target="$1"
    if [ "$target" = "backend" ]; then
        tail -f "$BACKEND_LOG"
    elif [ "$target" = "frontend" ]; then
        tail -f "$FRONTEND_LOG"
    else
        log_info "Streaming both backend and frontend logs (Ctrl+C to stop)..."
        tail -f "$BACKEND_LOG" "$FRONTEND_LOG"
    fi
}

# ------------------------------------------------------------------------------
# Install systemd Service Units
# ------------------------------------------------------------------------------
install_systemd() {
    local current_user
    current_user=$(whoami)
    local service_dir="/etc/systemd/system"

    if [ "$EUID" -ne 0 ]; then
        log_warn "Systemd installation requires root privileges. Please run with sudo:"
        echo "  sudo ./deploy.sh install-systemd"
        exit 1
    fi

    log_info "Generating systemd service unit files..."

    cat <<EOF > "$service_dir/pdf-qa-backend.service"
[Unit]
Description=PDF QA Agent Backend API (FastAPI)
After=network.target

[Service]
Type=simple
User=$current_user
WorkingDirectory=$APP_DIR/..
EnvironmentFile=-$APP_DIR/.env
Environment=PYTHONPATH=$APP_DIR/..
ExecStart=$VENV_DIR/bin/uvicorn pdf_qa_agent.api.main:app --host 0.0.0.0 --port $BACKEND_PORT
Restart=always
RestartSec=5
StandardOutput=append:$BACKEND_LOG
StandardError=append:$BACKEND_LOG

[Install]
WantedBy=multi-user.target
EOF

    cat <<EOF > "$service_dir/pdf-qa-frontend.service"
[Unit]
Description=PDF QA Agent Frontend (Next.js)
After=network.target pdf-qa-backend.service

[Service]
Type=simple
User=$current_user
WorkingDirectory=$FRONTEND_DIR
Environment=NODE_ENV=production
Environment=PORT=$FRONTEND_PORT
Environment=HOSTNAME=0.0.0.0
Environment=BACKEND_URL=http://127.0.0.1:$BACKEND_PORT
ExecStart=/usr/bin/node .next/standalone/server.js
Restart=always
RestartSec=5
StandardOutput=append:$FRONTEND_LOG
StandardError=append:$FRONTEND_LOG

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable pdf-qa-backend.service pdf-qa-frontend.service

    log_success "Systemd services successfully installed and enabled on system boot!"
    echo "  Control backend:  sudo systemctl [start|stop|restart|status] pdf-qa-backend"
    echo "  Control frontend: sudo systemctl [start|stop|restart|status] pdf-qa-frontend"
}

# ------------------------------------------------------------------------------
# CLI Router
# ------------------------------------------------------------------------------
case "$1" in
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    install-systemd)
        install_systemd
        ;;
    *)
        echo "Usage: $0 {build|start|stop|restart|status|logs [service]|install-systemd}"
        exit 1
        ;;
esac
