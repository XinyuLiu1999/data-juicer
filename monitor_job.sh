#!/bin/bash
# Clean Ray Job Monitor for Data-Juicer

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Data-Juicer Ray Job Monitor ===${NC}"
echo ""

# Find latest Ray session
SESSION_DIR=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)

if [ -z "$SESSION_DIR" ]; then
    echo -e "${RED}No active Ray session found${NC}"
    echo ""
    echo "Ray may not be running. Start your job first:"
    echo "  python tools/process_data.py --config optimized_config.yaml > /tmp/dj.log 2>&1 &"
    echo ""
    echo "Or check if Ray is running with a different temp directory."
    exit 1
fi

echo -e "${GREEN}Ray session:${NC} $SESSION_DIR"

# Check for Ray dashboard
DASHBOARD_PORT=$(grep -r "dashboard" "$SESSION_DIR"/*.log 2>/dev/null | grep -oP "(?<=:)[0-9]{4,5}" | head -1)
if [ -n "$DASHBOARD_PORT" ]; then
    echo -e "${GREEN}Ray Dashboard:${NC} http://127.0.0.1:$DASHBOARD_PORT"
    echo -e "${YELLOW}TIP: Open this URL in your browser for the cleanest view!${NC}"
else
    echo -e "${YELLOW}Dashboard URL not found yet${NC}"
fi

echo ""
echo -e "${GREEN}Finding job logs...${NC}"

# Wait for job log to appear
MAX_WAIT=30
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    JOB_LOG=$(find "$SESSION_DIR/logs" -name "job-driver-*.log" 2>/dev/null | head -1)
    if [ -n "$JOB_LOG" ]; then
        break
    fi
    echo -e "${YELLOW}Waiting for job to start... (${WAITED}s)${NC}"
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ -z "$JOB_LOG" ]; then
    echo -e "${RED}No job log found after ${MAX_WAIT}s${NC}"
    echo ""
    echo "Available logs in session:"
    ls -lh "$SESSION_DIR/logs/" 2>/dev/null || echo "  (none)"
    exit 1
fi

echo -e "${GREEN}Following:${NC} $JOB_LOG"
echo ""
echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}Clean Output (filtered for readability)${NC}"
echo -e "${YELLOW}=========================================${NC}"
echo ""

# Follow log with clean filtering
tail -f "$JOB_LOG" | while IFS= read -r line; do
    # Skip very noisy lines
    if echo "$line" | grep -q "Tasks: 0; Actors: 0; Queued blocks: 0"; then
        continue
    fi

    # Color-code important lines
    if echo "$line" | grep -qi "error"; then
        echo -e "${RED}$line${NC}"
    elif echo "$line" | grep -qi "warning"; then
        echo -e "${YELLOW}$line${NC}"
    elif echo "$line" | grep -qi "completed\|finished\|100%"; then
        echo -e "${GREEN}$line${NC}"
    elif echo "$line" | grep -qE "INFO.*data_juicer|Loading|Running Dataset"; then
        echo "$line"
    elif echo "$line" | grep -qE "row/s|MapBatches.*%|Filter.*%"; then
        # Show progress but skip repetitive actor initialization
        if ! echo "$line" | grep -q "all objects local"; then
            echo "$line"
        fi
    fi
done
