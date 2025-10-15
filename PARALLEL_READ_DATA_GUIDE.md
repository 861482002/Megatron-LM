# Parallel read_data Implementation Guide

## Overview

This guide explains how to integrate the parallelized `read_data` function into your PyTorch distributed checkpoint loading system. The parallel implementation uses Python's multiprocessing to read multiple checkpoint files concurrently.

## Key Changes

### Original vs. Parallel Approach

**Original (Sequential):**
- Processes files one by one in a single process
- For each file, reads all requests sequentially
- Updates planner as items are loaded

**Parallel (Multiprocessing):**
- Distributes files across multiple worker processes
- Each worker processes one file independently
- Results are collected and planner is updated in the main process

## Implementation Options

### Option 1: Using multiprocessing.Pool (Simple)

```python
def read_data_parallel(self, plan, planner, num_workers=None):
    # Group files
    # Create worker arguments
    # Use Pool.map() to process in parallel
    # Aggregate results in main process
```

**Pros:**
- Simple API
- Automatic process management
- Good for CPU-bound tasks

**Cons:**
- Less control over execution
- All tasks submitted at once

### Option 2: Using concurrent.futures.ProcessPoolExecutor (Recommended)

```python
def read_data_parallel_v2(self, plan, planner, max_workers=None):
    # Group files
    # Use ProcessPoolExecutor
    # Process results as they complete
    # Better exception handling
```

**Pros:**
- More flexible
- Better exception handling
- Can process results as they arrive
- Compatible with Future-based APIs

**Cons:**
- Slightly more complex

## Integration Steps

### Step 1: Adapt the Worker Function

The `_read_file_worker` function needs to be customized based on your specific:

1. **Filesystem abstraction**: Replace the simple `open()` call with your `fs.create_stream()`
2. **File slicing logic**: Implement your `_slice_file()` method
3. **Transform pipeline**: Integrate your `transforms.transform_load_stream()`
4. **Tensor narrowing**: Use your `narrow_tensor_by_index()` function

Example customization:

```python
def _read_file_worker(args):
    (relative_path, reqs, storage_data_dict, fs_config, base_path) = args
    
    # Reconstruct filesystem object
    fs = YourFilesystem(**fs_config)
    
    new_path = fs.concat_path(base_path, relative_path)
    results = []
    
    with fs.create_stream(new_path, "rb") as stream:
        for req in reqs:
            item_md = storage_data_dict[req.storage_index]
            
            # Use your actual implementation
            file_slice = your_slice_file(stream, item_md)
            
            # Apply transformations
            transform_from = your_transforms.transform_load_stream(
                req,
                item_md.transform_descriptors or (),
                file_slice,
            )
            
            # ... rest of the logic
    
    return results
```

### Step 2: Handle Serialization

Multiprocessing requires all arguments to be picklable. You need to:

1. **Serialize filesystem config**: Instead of passing the `fs` object, pass configuration dict
2. **Serialize transforms**: Pass transform configuration, not transform objects
3. **Storage data**: Already serializable (dicts and dataclasses)

```python
# Prepare serializable arguments
worker_args = []
for relative_path, reqs in per_file.items():
    storage_data_dict = {
        req.storage_index: self.storage_data[req.storage_index]
        for req in reqs
    }
    
    worker_args.append((
        relative_path,
        reqs,
        storage_data_dict,
        self.fs.get_config(),  # Serializable config
        self.path,
    ))
```

### Step 3: Optimize Worker Count

Choose the number of workers based on:

```python
import multiprocessing as mp

def get_optimal_workers(num_files, max_workers=None):
    """
    Determine optimal number of workers.
    
    Args:
        num_files: Number of files to process
        max_workers: Maximum workers to use (None = cpu_count)
    
    Returns:
        Optimal number of workers
    """
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # Don't create more workers than files
    optimal = min(max_workers, num_files)
    
    # For small numbers of files, might be better to go sequential
    if num_files == 1:
        return 1
    
    return optimal
```

## Performance Considerations

### When to Use Parallel Reading

**Good candidates:**
- ✅ Multiple large checkpoint files
- ✅ I/O bound operations (reading from slow storage)
- ✅ CPU-bound transformations (decompression, decryption)
- ✅ Network file systems with good parallel read support

**Poor candidates:**
- ❌ Single file checkpoints
- ❌ Very small files (overhead > benefit)
- ❌ Memory-constrained environments
- ❌ File systems that don't support parallel reads well

### Memory Management

Each worker process holds its own copy of loaded data. Monitor memory usage:

```python
# Limit workers based on available memory
import psutil

def get_memory_aware_workers(avg_file_size_mb, max_workers=None):
    available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    # Leave 20% headroom
    usable_memory = available_memory_mb * 0.8
    
    # Estimate workers that fit in memory
    workers_by_memory = int(usable_memory / avg_file_size_mb)
    
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    return min(workers_by_memory, max_workers)
```

### I/O Optimization

For optimal I/O performance:

1. **Sort by offset**: Read files in sequential order when possible
2. **Batch small reads**: Combine small reads from the same file
3. **Prefetch**: Consider adding prefetch logic

```python
# Sort requests by offset before processing
for relative_path, reqs in per_file.items():
    # Sort requests by file offset to improve sequential reads
    sorted_reqs = sorted(
        reqs,
        key=lambda r: getattr(
            self.storage_data[r.storage_index],
            'offset',
            0
        )
    )
    # Use sorted_reqs instead of reqs
```

## Error Handling

Enhanced error handling:

```python
def _read_file_worker(args):
    try:
        # ... processing logic ...
        return results
    except Exception as e:
        import traceback
        error_info = {
            'file': args[0],
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return [(None, 'error', error_info)]

# In main process:
for req, data_type, loaded_data in all_results:
    if data_type == 'error':
        print(f"Error in file {loaded_data['file']}:")
        print(loaded_data['traceback'])
        raise RuntimeError(f"Worker failed: {loaded_data['error']}")
```

## Testing

### Unit Test Example

```python
def test_parallel_vs_sequential():
    """Verify parallel implementation produces same results as sequential."""
    # Create test plan and planner
    plan = create_test_plan()
    planner_seq = create_planner()
    planner_par = create_planner()
    
    # Run both versions
    reader_seq = YourReader()
    reader_seq.read_data(plan, planner_seq)
    
    reader_par = YourReader()
    reader_par.read_data_parallel(plan, planner_par, num_workers=4)
    
    # Compare results
    assert_planners_equal(planner_seq, planner_par)
```

### Performance Benchmark

```python
import time

def benchmark_read_data(checkpoint_dir, num_workers_list=[1, 2, 4, 8]):
    """Benchmark different worker counts."""
    for num_workers in num_workers_list:
        start = time.time()
        
        reader = YourReader()
        reader.read_data_parallel(plan, planner, num_workers=num_workers)
        
        elapsed = time.time() - start
        print(f"Workers: {num_workers}, Time: {elapsed:.2f}s")
```

## Migration Path

### Step 1: Add parallel version alongside original

```python
class YourReader:
    def read_data(self, plan, planner):
        """Original sequential implementation (keep for compatibility)."""
        # ... existing code ...
    
    def read_data_parallel(self, plan, planner, num_workers=None):
        """New parallel implementation."""
        # ... new code ...
```

### Step 2: Add feature flag

```python
class YourReader:
    def __init__(self, use_parallel=False, num_workers=None):
        self.use_parallel = use_parallel
        self.num_workers = num_workers
    
    def read_data(self, plan, planner):
        if self.use_parallel:
            return self.read_data_parallel(plan, planner, self.num_workers)
        else:
            return self._read_data_sequential(plan, planner)
```

### Step 3: Gradual rollout

1. Test in development with `use_parallel=True`
2. Run A/B tests in staging
3. Monitor performance and errors
4. Gradually increase parallel usage
5. Eventually make parallel the default

## Troubleshooting

### Issue: Slower than sequential

**Possible causes:**
- Too many workers (context switching overhead)
- Small files (multiprocessing overhead)
- Poor I/O parallelization support

**Solutions:**
- Reduce worker count
- Use sequential for small checkpoints
- Benchmark on your specific hardware

### Issue: Out of memory

**Possible causes:**
- Too many workers loading large tensors
- Worker processes not releasing memory

**Solutions:**
- Reduce worker count
- Process fewer files at once
- Monitor with memory profiling

### Issue: Pickling errors

**Possible causes:**
- Non-serializable objects in arguments
- Lambda functions or local functions
- Complex objects

**Solutions:**
- Use serializable data structures
- Define worker function at module level
- Pass configuration instead of objects

## Advanced: Shared Memory

For large tensors, consider using shared memory:

```python
from multiprocessing import shared_memory
import numpy as np

def use_shared_memory_tensor(tensor):
    """Place tensor in shared memory for zero-copy access."""
    # Convert to numpy (shared layout)
    np_array = tensor.numpy()
    
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
    
    # Copy data to shared memory
    shared_array = np.ndarray(
        np_array.shape,
        dtype=np_array.dtype,
        buffer=shm.buf
    )
    shared_array[:] = np_array[:]
    
    return shm.name, np_array.shape, str(np_array.dtype)

# In worker: reconstruct from shared memory
def reconstruct_tensor(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    tensor = torch.from_numpy(np_array)
    return tensor
```

## Conclusion

The parallel implementation can significantly speed up checkpoint loading when:
- You have multiple checkpoint files
- I/O is a bottleneck
- You have CPU headroom

Remember to:
- ✅ Test thoroughly
- ✅ Monitor performance
- ✅ Handle errors gracefully
- ✅ Consider memory constraints
- ✅ Keep the sequential version as fallback
