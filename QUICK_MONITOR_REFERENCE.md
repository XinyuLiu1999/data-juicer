# Quick Monitoring Reference Card

## **TL;DR - Best Way to Monitor**

### **Option 1: Ray Dashboard (Best!) 🌟**

When you start your job, look for this line:
```
View the Ray dashboard at http://127.0.0.1:8265
```

**Open that URL in your browser** - it's 100x cleaner than console output!

If on a remote server, SSH tunnel:
```bash
ssh -L 8265:localhost:8265 your-server
# Then open: http://localhost:8265
```

---

### **Option 2: Clean Console Monitoring**

**Run your job in the background with logs:**
```bash
python tools/process_data.py --config optimized_config.yaml \
  > /tmp/dj.log 2>&1 &
```

**Monitor with the clean script:**
```bash
./monitor_job.sh
```

**Or manually with grep:**
```bash
tail -f /tmp/dj.log | grep -E "(INFO.*data_juicer|completed|ERROR)" | grep -v "Tasks: 0"
```

---

### **Option 3: Ray Logs (Cleanest text)**

```bash
# Find latest job log
tail -f /tmp/ray/session_latest_/logs/job-driver-*.log
```

---

## **One-Liners for Common Tasks**

### **See only progress:**
```bash
tail -f /tmp/dj.log | grep -E "[0-9]+%|row/s|completed"
```

### **See only errors:**
```bash
tail -f /tmp/dj.log | grep -i error
```

### **See only data-juicer INFO messages:**
```bash
tail -f /tmp/dj.log | grep "INFO.*data_juicer" | grep -v "Tasks: 0"
```

### **See GPU usage:**
```bash
watch -n 1 nvidia-smi
```

### **See which operators are running:**
```bash
tail -f /tmp/dj.log | grep -E "MapBatches|Filter" | grep -v "Tasks: 0; Actors: 0"
```

---

## **For Your Current Setup**

Based on your config, here's what to watch:

### **Expected Flow:**

1. **Text filters** (fast, CPU-only):
   ```
   alphanumeric_filter → character_repetition_filter →
   flagged_words_filter → perplexity_filter → ...
   ```
   Look for: `Loading sentencepiece model`, `Loading kenlm model`

2. **Simple image filters** (fast, CPU-only):
   ```
   image_aspect_ratio_filter → image_shape_filter →
   image_size_filter → image_blurriness_filter → ...
   ```
   Look for: `100%` completion bars

3. **GPU filters** (slow, this is where time is spent):
   ```
   image_text_similarity_filter → image_text_matching_filter →
   image_watermark_filter → image_nsfw_filter → ...
   ```
   Look for: `Actors: N`, GPU utilization in `nvidia-smi`

### **Key Metrics to Watch:**

| Metric | Command | Good Value |
|--------|---------|------------|
| **GPU Utilization** | `nvidia-smi` | 70-95% during GPU filters |
| **GPU Memory** | `nvidia-smi` | 3-8GB per GPU (with 2GB models) |
| **Processing Rate** | Log output | 10-100 row/s (varies by filter) |
| **Active Actors** | Log output | 1-4 actors per GPU filter |

---

## **Troubleshooting Output Issues**

### **If monitor_job.sh shows "No active Ray session":**

Check if Ray is actually running:
```bash
ps aux | grep ray
```

If not, your job may have finished or crashed. Check the job log:
```bash
tail -100 /tmp/dj.log
```

### **If you see "Tasks: 0; Actors: 0" spam:**

This is normal during initialization. It should stop once actors start processing.

Filter it out:
```bash
tail -f /tmp/dj.log | grep -v "Tasks: 0; Actors: 0"
```

### **If you want to see EVERYTHING (for debugging):**

```bash
tail -f /tmp/dj.log
```

Yes, it's messy, but sometimes you need it for debugging.

---

## **Quick Start Script**

Save this as `run_and_monitor.sh`:

```bash
#!/bin/bash

# Run job in background
echo "Starting data-juicer job..."
python tools/process_data.py --config optimized_config.yaml \
  > /tmp/dj_$(date +%Y%m%d_%H%M%S).log 2>&1 &

JOB_PID=$!
echo "Job PID: $JOB_PID"
echo "Log file: $(ls -t /tmp/dj_*.log | head -1)"

# Wait a moment for job to initialize
sleep 3

# Start clean monitoring
echo ""
echo "Starting clean monitor..."
./monitor_job.sh
```

Usage:
```bash
chmod +x run_and_monitor.sh
./run_and_monitor.sh
```

---

## **Ray Dashboard Quick Tour**

Once you open http://127.0.0.1:8265, here's where to look:

### **Jobs Tab:**
- See your current job status
- Overall progress percentage
- Total duration
- Resource usage summary

### **Metrics Tab:**
- **Node CPU**: See CPU usage per node
- **Node GPU**: See GPU usage per GPU
- **Object Store Memory**: See Ray's object store usage

### **Actors Tab:**
- See all active GPU filter workers
- Check their states (RUNNING, PENDING, DEAD)
- See resource allocation per actor

### **Logs Tab:**
- Search logs by keyword
- Filter by actor/worker
- Download logs for offline analysis

---

## **Summary**

**Messy console output?**
→ Use Ray Dashboard or `./monitor_job.sh`

**Want to see everything?**
→ `tail -f /tmp/dj.log`

**Want clean text output?**
→ `tail -f /tmp/ray/session_latest_/logs/job-driver-*.log | grep -v "Tasks: 0"`

**Want GPU stats?**
→ `watch -n 1 nvidia-smi`

**Want visual graphs?**
→ Ray Dashboard at http://127.0.0.1:8265
