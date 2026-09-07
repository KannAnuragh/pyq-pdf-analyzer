#!/bin/bash
# run.sh

# Start backend
source venv/bin/activate
uvicorn pdf_qa_agent.api.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:3000"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
