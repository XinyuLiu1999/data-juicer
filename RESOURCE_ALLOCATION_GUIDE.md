# Resource Allocation Guide for Data-Juicer with Ray

## Understanding Your Error

```
ValueError: CPU resource is not enough for the current operators configuration.
At least 92.0 cpus are required, but only 32.0 cpus are available.
GPU resource is not enough for the current operators configuration.
At least 14.0 gpus are required, but only 2.0 gpus are available.
```

### Why This Happened

You specified explicit resource allocations (`num_proc`, `num_gpus`, `num_cpus`) for multiple GPU operators. The system validated that **all operators** can run simultaneously and found insufficient resources.

**Example calculation from your config:**
- 7 GPU filters × 2 workers each = 14 GPU worker instances
- Each worker needs 1 GPU → **14 GPUs required**
- Each worker needs 4-6 CPUs → **56-84 CPUs required**
- But you only have: **2 GPUs + 32 CPUs**

### The Key Insight: Sequential vs Concurrent Execution

**Important**: Operators in the pipeline run **SEQUENTIALLY**, not concurrently!

```
Data → Filter1 → Filter2 → Filter3 → ... → Output
       ↑ active  (waiting)  (waiting)
```

At any given time, only **ONE filter** is actively processing data. However, when you specify explicit `num_proc`, the system must validate that all operators **could** run simultaneously (for streaming mode).

## Solution 1: Auto-Parallelism (Recommended)

**File**: `optimized_config.yaml`

### Key Changes:
```yaml
# Enable automatic resource optimization
auto_op_parallelism: true

process:
  - image_text_similarity_filter:
      hf_clip: openai/clip-vit-base-patch32
      min_score: 0.18
      memory: '1500MB'  # ← Only specify memory
      # No num_proc, num_gpus, num_cpus - let system decide!
```

### How It Works:
1. System detects available resources (32 CPUs, 2 GPUs)
2. For each operator, calculates optimal concurrency based on:
   - Memory requirement
   - Available GPU memory (2 × 20GB = 40GB)
   - Available CPUs (32 cores)
3. Automatically sets fractional GPU allocations
4. Balances workload across resources

### Benefits:
- ✅ No manual tuning required
- ✅ Automatically adapts to your hardware
- ✅ Prevents over-allocation errors
- ✅ Optimizes throughput

## Solution 2: Manual Control (Conservative)

**File**: `manual_resource_config.yaml`

### Key Principles:

#### 1. Use Fractional GPUs
```yaml
- image_text_similarity_filter:
    memory: '1500MB'
    num_gpus: 0.25      # Each worker uses 25% of a GPU
    num_proc: 8         # 8 workers total (4 per GPU)
    num_cpus: 2         # 2 CPUs per worker
```

**GPU Math**:
- Each worker needs 0.25 GPU
- You have 2 GPUs total
- Can run: 2 / 0.25 = **8 workers maximum**
- Actual GPU memory used: 8 × 1.5GB = 12GB (fits in 40GB total)

**CPU Math**:
- Each worker needs 2 CPUs
- 8 workers × 2 CPUs = **16 CPUs required**
- You have 32 CPUs → **✅ Fits!**

#### 2. Adjust by Model Size

**Lighter models** (500MB - 1.5GB):
```yaml
num_gpus: 0.2 - 0.25
num_proc: 8 - 10
num_cpus: 2
```

**Heavier models** (2GB+):
```yaml
num_gpus: 0.5
num_proc: 4
num_cpus: 4
```

#### 3. CPU Filters

For CPU-only filters (text filters, simple image filters):
```yaml
num_proc: 16  # Use ~half your CPUs for balance
```

## Understanding the `memory` Parameter

### What It IS:
- A **scheduling hint** for resource planning
- Used to calculate how many workers can fit
- Helps prevent over-subscription

### What It Is NOT:
- ❌ A hard limit on memory usage
- ❌ A runtime memory constraint
- ❌ A guarantee that the model won't use more

### Example:

```yaml
- image_maniqa_filter:
    memory: '2GB'      # Scheduling hint
    num_gpus: 0.25     # Each worker gets 25% of a GPU
```

**What happens**:
1. System sees: 2GB memory requirement
2. Calculates: 40GB total / 2GB per worker = 20 workers could fit
3. But you specified `num_gpus: 0.25`, limiting to: 2 GPUs / 0.25 = 8 workers
4. Actual allocation: **8 workers**, each can use up to ~5GB GPU memory (20GB / 4 workers per GPU)
5. Model actually uses: ~2GB per worker
6. Remaining GPU memory: Available for batch processing

## Resource Allocation Formula

For a GPU filter with N workers on M GPUs:

### GPU Requirements:
```
gpu_per_worker = num_gpus
total_gpus_needed = N × gpu_per_worker
Must satisfy: total_gpus_needed ≤ M

Example:
N = 8 workers
gpu_per_worker = 0.25
total_gpus_needed = 8 × 0.25 = 2.0 ✅ (fits in 2 GPUs)
```

### CPU Requirements:
```
cpu_per_worker = num_cpus
total_cpus_needed = N × cpu_per_worker
Must satisfy: total_cpus_needed ≤ total_available_cpus

Example:
N = 8 workers
cpu_per_worker = 2
total_cpus_needed = 8 × 2 = 16 ✅ (fits in 32 CPUs)
```

### Memory Requirements (GPU):
```
Calculated based on memory hint:
gpu_fraction = memory / total_gpu_memory
workers_that_fit = total_gpu_memory / memory

Example:
memory = 2GB
total_gpu_memory = 40GB (2 × 20GB)
gpu_fraction = 2GB / 40GB = 0.05
workers_that_fit = 40GB / 2GB = 20 workers (theoretical max)
```

## Monitoring Resource Usage

Enable monitoring in your config:
```yaml
open_monitor: true  # Track CPU/GPU/memory per operator
open_tracer: true   # Detailed execution traces
```

During execution, watch GPU utilization:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

Look for:
- **GPU-Util**: Should be 70-95% when processing
- **Memory-Usage**: Should match your expectations
- **Power Usage**: Higher = more utilized

### Interpreting Results:

**Under-utilized** (< 50% GPU utilization):
- Increase `num_proc` (more workers)
- Decrease `num_cpus` (less CPU bottleneck)
- Increase batch size

**Over-utilized** (OOM errors):
- Decrease `num_proc` (fewer workers)
- Increase `num_gpus` (more GPU per worker)
- Increase `memory` specification

**CPU-bottlenecked** (low GPU util, high CPU util):
- Increase `num_cpus` per GPU worker
- Decrease `num_proc` for CPU filters

## Quick Reference: Your Hardware

**Available Resources:**
- CPUs: 32 cores
- GPUs: 2 × 20GB = 40GB total GPU memory

**Recommended Allocations:**

### For Light Models (500MB - 1.5GB):
```yaml
memory: '1500MB'
num_gpus: 0.2       # 5 workers per GPU
num_proc: 10        # 10 workers total
num_cpus: 2         # 20 CPUs total
```

### For Heavy Models (2GB+):
```yaml
memory: '2GB'
num_gpus: 0.5       # 2 workers per GPU
num_proc: 4         # 4 workers total
num_cpus: 4         # 16 CPUs total
```

### For CPU Filters:
```yaml
num_proc: 16        # Half your cores
# No num_cpus needed (defaults to 1)
```

## Common Mistakes to Avoid

### ❌ Mistake 1: Setting num_proc = num_cpus
```yaml
# WRONG - requests 32 workers × 4 CPUs = 128 CPUs!
num_proc: 32
num_cpus: 4
```

### ❌ Mistake 2: Using num_gpus = 1 for all filters
```yaml
# WRONG - 7 filters × 2 workers × 1 GPU = 14 GPUs needed!
- image_maniqa_filter:
    num_proc: 2
    num_gpus: 1  # ← This means 1 FULL GPU per worker!
```

### ❌ Mistake 3: Ignoring auto-parallelism
```yaml
# WRONG - manually specifying everything is error-prone
auto_op_parallelism: false  # ← Don't disable unless you know what you're doing
```

### ✅ Correct Approach:
```yaml
# RIGHT - let the system optimize
auto_op_parallelism: true

- image_maniqa_filter:
    memory: '2GB'  # Just specify memory hint
    # Let system calculate num_gpus, num_proc, num_cpus
```

## Troubleshooting

### Error: "At least X GPUs required"
**Cause**: Sum of (num_proc × num_gpus) across all GPU filters exceeds available GPUs

**Fix**:
1. Use auto-parallelism, OR
2. Reduce `num_gpus` to fractional values (0.1 - 0.5), OR
3. Reduce `num_proc` to fewer workers

### Error: "At least X CPUs required"
**Cause**: Sum of (num_proc × num_cpus) across all operators exceeds available CPUs

**Fix**:
1. Use auto-parallelism, OR
2. Reduce `num_cpus` per worker (try 1-2 instead of 4-6), OR
3. Reduce `num_proc` for CPU filters (try 8-16 instead of 32)

### Low GPU Utilization
**Symptoms**: `nvidia-smi` shows < 50% GPU-Util

**Causes**:
- Too few workers (`num_proc` too low)
- CPU bottleneck (not enough `num_cpus` per worker)
- Small batch size

**Fix**:
1. Increase `num_proc` (more workers)
2. Increase `num_cpus` for GPU workers (faster data loading)
3. Increase batch size in operator config

### OOM (Out of Memory) Errors
**Symptoms**: CUDA out of memory errors

**Causes**:
- Too many workers per GPU
- Batch size too large
- Model actually uses more memory than specified

**Fix**:
1. Reduce `num_proc` (fewer workers)
2. Increase `num_gpus` per worker (more GPU memory per worker)
3. Reduce batch size
4. Increase `memory` specification (better scheduling)

## Next Steps

1. **Try the optimized config first**:
   ```bash
   dj-process --config optimized_config.yaml
   ```

2. **Monitor resource usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **If you need manual control**, use `manual_resource_config.yaml` and adjust based on monitoring

4. **Iterate**: Tune based on actual GPU/CPU utilization observed

## Summary

| Config Type | When to Use | Pros | Cons |
|-------------|-------------|------|------|
| **Auto-parallelism** | Default, first try | Automatic, no tuning | Less control |
| **Manual (fractional GPU)** | Fine-tuning performance | Maximum control | Requires monitoring |
| **Manual (conservative)** | Debugging, stability | Predictable | May under-utilize |

**Recommendation**: Start with `optimized_config.yaml` (auto-parallelism). Only switch to manual tuning if you observe performance issues or want to maximize throughput.
