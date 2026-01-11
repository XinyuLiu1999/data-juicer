# Clean Ray Monitoring Guide

The console output from Ray Data can be extremely messy. Here are **much cleaner** ways to monitor your data-juicer jobs.

---

## **Method 1: Ray Dashboard (Best Option) 🌟**

Ray provides a **web-based dashboard** that's much cleaner than console output.

### **Finding the Dashboard URL:**

When you start a Ray cluster, look for this message:
```
View the Ray dashboard at http://127.0.0.1:8265
```

Or check programmatically:
```bash
python -c "import ray; ray.init(address='auto'); print(ray.dashboard_url)"
```

### **What You Can See:**

1. **Jobs Tab**: Overall job progress
   - Job status (RUNNING, SUCCEEDED, FAILED)
   - Duration and resource usage
   - Clean progress bars for each operator

2. **Actors Tab**: GPU filter workers
   - Number of active actors
   - Resource allocation per actor
   - Actor states (PENDING, RUNNING, DEAD)

3. **Metrics Tab**: Real-time resource usage
   - CPU utilization per node
   - GPU utilization and memory
   - Object store memory usage
   - Clean graphs and charts

4. **Logs Tab**: Structured log viewing
   - Filter by worker/actor
   - Search functionality
   - Download logs

### **Accessing from Remote Server:**

If you're running on a remote server (like your CephFS setup), you need to forward the port:

```bash
# On your local machine:
ssh -L 8265:localhost:8265 your-server

# Then open in browser:
http://localhost:8265
```

---

## **Method 2: Ray Logs Directory**

Ray stores clean, structured logs in a temporary directory.

### **Finding Log Directory:**

```bash
# Find the Ray session directory
ls -lt /tmp/ray/session_* | head -1

# Example output:
# /tmp/ray/session_2026-01-11_09-30-45_123456/
```

### **Log Structure:**

```
/tmp/ray/session_*/
├── logs/
│   ├── job-driver-*.log              # Main job logs (cleanest!)
│   ├── worker-*.out                  # Worker stdout
│   ├── worker-*.err                  # Worker stderr
│   └── raylet.out                    # Ray system logs
└── dashboard_logs/
    └── dashboard.log                 # Dashboard logs
```

### **Reading Clean Job Logs:**

```bash
# Find your job's driver log (cleanest output)
tail -f /tmp/ray/session_*/logs/job-driver-*.log

# Or follow all job logs:
find /tmp/ray/session_latest_/logs -name "job-*.log" -exec tail -f {} +
```

### **Filtering Specific Information:**

```bash
# See only progress updates
tail -f /tmp/ray/session_*/logs/job-driver-*.log | grep -E "(Running|completed|row/s)"

# See only errors
tail -f /tmp/ray/session_*/logs/job-driver-*.log | grep -i error

# See only operator execution
tail -f /tmp/ray/session_*/logs/job-driver-*.log | grep -E "(MapBatches|Filter)"
```

---

## **Method 3: Redirect Output When Running**

Redirect the messy console output to a file and tail specific parts.

### **Option A: Separate stdout/stderr**

```bash
python tools/process_data.py --config optimized_config.yaml \
  > /tmp/dj_output.log 2> /tmp/dj_errors.log
```

Then monitor cleanly:
```bash
# In another terminal - clean progress only:
tail -f /tmp/dj_output.log | grep -E "(INFO|completed|%)"

# Or just errors:
tail -f /tmp/dj_errors.log
```

### **Option B: Combined with filtering**

```bash
python tools/process_data.py --config optimized_config.yaml 2>&1 | \
  tee /tmp/dj_full.log | \
  grep -v "MapBatches(compute_stats" | \
  grep -E "(INFO|WARNING|ERROR|Running Dataset|row/s)"
```

This filters out the repetitive MapBatches lines while showing important info.

---

## **Method 4: Data-Juicer's Built-in Monitoring**

Data-juicer has monitoring features you already enabled!

### **Your Config Already Has:**

```yaml
open_tracer: true     # Execution traces
open_monitor: true    # Resource monitoring
```

### **Monitoring Output Files:**

After the job completes, check:
```bash
# Tracer output (execution timeline)
ls -lh /cephfs/liuxinyu/my_dj/tests/testing-cc3m-1/tracer_*.json

# Monitor output (resource usage per operator)
ls -lh /cephfs/liuxinyu/my_dj/tests/testing-cc3m-1/monitor_*.json
```

### **Analyzing Tracer Data:**

```bash
# Pretty-print tracer output to see operator timeline
python -m json.tool /path/to/tracer_*.json | less

# Extract operator durations
cat tracer_*.json | jq '.[] | {op: .op_name, duration: .duration}'
```

### **Analyzing Monitor Data:**

```bash
# See resource usage per operator
python -m json.tool monitor_*.json | less

# Extract GPU usage
cat monitor_*.json | jq '.[] | select(.gpu_util != null) | {op: .op_name, gpu_util: .gpu_util, gpu_mem: .gpu_mem}'
```

---

## **Method 5: Custom Clean Output Script**

Create a simple script to monitor just what you need:

### **Create Monitor Script:**

```bash
cat > /home/user/data-juicer/monitor_job.sh <<'EOF'
#!/bin/bash

# Find latest Ray session
SESSION_DIR=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)

if [ -z "$SESSION_DIR" ]; then
    echo "No Ray session found"
    exit 1
fi

echo "Monitoring Ray session: $SESSION_DIR"
echo "========================================="

# Find job driver log
JOB_LOG=$(find "$SESSION_DIR/logs" -name "job-driver-*.log" 2>/dev/null | head -1)

if [ -z "$JOB_LOG" ]; then
    echo "No job log found yet, waiting..."
    sleep 5
    JOB_LOG=$(find "$SESSION_DIR/logs" -name "job-driver-*.log" 2>/dev/null | head -1)
fi

if [ -n "$JOB_LOG" ]; then
    echo "Following: $JOB_LOG"
    echo ""

    # Clean output - only show important lines
    tail -f "$JOB_LOG" | grep -E "(INFO|WARNING|ERROR|completed|progress|Dataset)" | \
        grep -v "Tasks: 0; Actors: 0" | \
        sed 's/^.*INFO.*- /[INFO] /' | \
        sed 's/^.*WARNING.*- /[WARN] /' | \
        sed 's/^.*ERROR.*- /[ERROR] /'
else
    echo "Could not find job log"
fi
EOF

chmod +x /home/user/data-juicer/monitor_job.sh
```

### **Usage:**

```bash
# In one terminal - run your job:
python tools/process_data.py --config optimized_config.yaml > /tmp/dj_full.log 2>&1

# In another terminal - clean monitoring:
./monitor_job.sh
```

---

## **Method 6: Use Ray CLI (If Available)**

If Ray CLI is in your conda environment:

```bash
# Activate your data-juicer environment
conda activate data-juicer-pyiqa-safe  # or your env name

# Check Ray status
ray status

# List jobs
ray job list

# Get specific job logs
ray job logs <job_id> --follow

# Monitor cluster resources
ray status --address auto
```

---

## **Quick Comparison:**

| Method | Cleanliness | Real-time | Ease | Best For |
|--------|-------------|-----------|------|----------|
| **Ray Dashboard** | ⭐⭐⭐⭐⭐ | ✅ | Easy | Visual monitoring, graphs |
| **Ray Logs** | ⭐⭐⭐⭐ | ✅ | Medium | Clean text logs |
| **Redirect + Filter** | ⭐⭐⭐ | ✅ | Easy | Quick filtering |
| **Tracer/Monitor** | ⭐⭐⭐⭐ | ❌ | Easy | Post-analysis |
| **Custom Script** | ⭐⭐⭐⭐ | ✅ | Medium | Customized view |
| **Ray CLI** | ⭐⭐⭐⭐ | ✅ | Easy | Command-line users |

---

## **Recommended Workflow:**

### **For Active Monitoring:**

1. **Start Ray Dashboard** (if accessible):
   ```bash
   # Note the dashboard URL when Ray starts
   # Access at http://127.0.0.1:8265
   ```

2. **Run job with output redirect**:
   ```bash
   python tools/process_data.py --config optimized_config.yaml \
     > /tmp/dj_output.log 2>&1 &
   ```

3. **Monitor cleanly in another terminal**:
   ```bash
   # Option 1: Clean grep
   tail -f /tmp/dj_output.log | grep -E "(INFO.*data_juicer|completed|ERROR)"

   # Option 2: Ray logs
   tail -f /tmp/ray/session_latest_/logs/job-driver-*.log | \
     grep -v "Tasks: 0; Actors: 0"

   # Option 3: GPU usage
   watch -n 2 nvidia-smi
   ```

### **For Post-Analysis:**

1. **Check tracer output**:
   ```bash
   cat tracer_*.json | jq -r '.[] | "\(.op_name): \(.duration)s"'
   ```

2. **Check monitor output**:
   ```bash
   cat monitor_*.json | jq -r '.[] | "\(.op_name): GPU=\(.gpu_util)%, Mem=\(.gpu_mem)MB"'
   ```

---

## **Example: Clean Monitoring Setup**

Here's a complete example for your current job:

### **Terminal 1: Run Job**
```bash
cd /home/user/data-juicer

# Run with output to file
python tools/process_data.py --config optimized_config.yaml \
  > /tmp/dj_job_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Note the PID
echo $! > /tmp/dj_job.pid
```

### **Terminal 2: Monitor Progress**
```bash
# Clean output
tail -f /tmp/dj_job_*.log | \
  grep -E "INFO.*data_juicer|completed|row/s" | \
  grep -v "Tasks: 0; Actors: 0" | \
  sed 's/.*INFO.*- //'
```

### **Terminal 3: Monitor GPU**
```bash
watch -n 1 nvidia-smi
```

### **Terminal 4: Ray Dashboard**
```bash
# If on remote server:
# ssh -L 8265:localhost:8265 your-server

# Then open in browser:
# http://localhost:8265
```

---

## **Summary:**

**The messy output you saw** is Ray Data's default console output, which includes:
- Every operator in the pipeline
- Every task and actor state
- Backpressure info
- Progress bars that overlap

**The cleanest solutions:**

1. 🥇 **Ray Dashboard** - Beautiful web UI (http://127.0.0.1:8265)
2. 🥈 **Ray Logs** - Clean text files in `/tmp/ray/session_*/logs/job-driver-*.log`
3. 🥉 **Redirect + grep** - Filter the noise yourself

**For your next run:**
```bash
# Best practice:
python tools/process_data.py --config optimized_config.yaml \
  > /tmp/dj_clean_$(date +%H%M%S).log 2>&1 &

# Then monitor cleanly:
tail -f /tmp/dj_clean_*.log | grep -E "(INFO.*data_juicer|completed|ERROR)"

# And in browser:
# Open http://127.0.0.1:8265
```
