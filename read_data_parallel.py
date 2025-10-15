"""
Parallelized version of read_data using multiprocessing.
This implementation processes multiple files concurrently using a process pool.
"""

import io
import multiprocessing as mp
from concurrent.futures import Future
from typing import Any, Dict, List, Tuple, cast

import torch
from torch import Tensor


def _read_file_worker(args: Tuple) -> List[Tuple[Any, Any, Any]]:
    """
    Worker function for processing a single file's read requests.
    
    Args:
        args: Tuple containing (relative_path, reqs, storage_data, transform_descriptors, 
              fs_path, base_path, fs_create_stream_fn)
    
    Returns:
        List of tuples: [(req, data_type, loaded_data), ...]
        where data_type is 'bytes' or 'tensor', and loaded_data is the actual data
    """
    from torch.distributed.checkpoint import LoadItemType
    
    (
        relative_path,
        reqs,
        storage_data_dict,
        fs_path,
        base_path,
    ) = args
    
    results = []
    
    # Reconstruct the filesystem object (simplified version - adjust based on your fs class)
    # Note: You may need to pass serializable fs parameters instead
    import os
    new_path = os.path.join(base_path, relative_path)
    
    try:
        with open(new_path, "rb") as stream:
            for req in reqs:
                item_md = storage_data_dict[req.storage_index]
                
                # Slice file - this is a simplified version, adjust based on your _slice_file impl
                if hasattr(item_md, 'offset') and hasattr(item_md, 'length'):
                    stream.seek(item_md.offset)
                    file_data = stream.read(item_md.length)
                    file_slice = io.BytesIO(file_data)
                else:
                    stream.seek(0)
                    file_slice = stream
                
                # Apply transformations (simplified - you may need to pass transform logic)
                transform_descriptors = getattr(item_md, 'transform_descriptors', ()) or ()
                # For simplicity, assuming no complex transforms here
                # You may need to serialize transform logic differently
                transform_from = file_slice
                
                if req.type == LoadItemType.BYTE_IO:
                    read_bytes = io.BytesIO(transform_from.read(-1))
                    read_bytes.seek(0)
                    results.append((req, 'bytes', read_bytes))
                else:
                    # Read data for tensor
                    if hasattr(transform_from, 'seekable') and transform_from.seekable():
                        seekable = transform_from
                    else:
                        seekable = io.BytesIO(transform_from.read(-1))
                        seekable.seek(0)
                    
                    tensor = cast(
                        Tensor,
                        torch.load(
                            seekable,
                            map_location="cpu",
                            weights_only=True,
                        ),
                    )
                    
                    # Apply narrowing (simplified - adjust based on your narrow_tensor_by_index)
                    if hasattr(req, 'storage_offsets') and hasattr(req, 'lengths'):
                        # Narrow tensor based on offsets and lengths
                        for dim, (offset, length) in enumerate(
                            zip(req.storage_offsets, req.lengths)
                        ):
                            if length != tensor.size(dim):
                                tensor = tensor.narrow(dim, offset, length)
                    
                    results.append((req, 'tensor', tensor))
    
    except Exception as e:
        # Return error information
        results.append((None, 'error', str(e)))
    
    return results


def read_data_parallel(
    self,
    plan: Any,  # LoadPlan
    planner: Any,  # LoadPlanner
    num_workers: int = None,
) -> Future[None]:
    """
    Parallel version of read_data using multiprocessing.
    
    Args:
        plan: LoadPlan containing items to read
        planner: LoadPlanner for coordinating loads
        num_workers: Number of worker processes (default: cpu_count)
    
    Returns:
        Future that resolves to None when loading is complete
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Group requests by file
    per_file: Dict[str, List] = {}
    for read_item in plan.items:
        item_md = self.storage_data[read_item.storage_index]
        path = item_md.relative_path
        per_file.setdefault(path, []).append(read_item)
    
    # Prepare worker arguments
    worker_args = []
    for relative_path, reqs in per_file.items():
        # Serialize storage_data for this file's requests
        storage_data_dict = {
            req.storage_index: self.storage_data[req.storage_index]
            for req in reqs
        }
        
        worker_args.append((
            relative_path,
            reqs,
            storage_data_dict,
            self.path,
            self.path,  # base_path
        ))
    
    # Process files in parallel
    results_list = []
    
    if len(worker_args) == 1 or num_workers == 1:
        # Single file or single worker - no need for multiprocessing
        for args in worker_args:
            results_list.extend(_read_file_worker(args))
    else:
        # Use multiprocessing pool
        with mp.Pool(processes=min(num_workers, len(worker_args))) as pool:
            all_results = pool.map(_read_file_worker, worker_args)
            for file_results in all_results:
                results_list.extend(file_results)
    
    # Process results in the main process
    for req, data_type, loaded_data in results_list:
        if data_type == 'error':
            raise RuntimeError(f"Error loading data: {loaded_data}")
        elif data_type == 'bytes':
            planner.load_bytes(req, loaded_data)
        elif data_type == 'tensor':
            target_tensor = planner.resolve_tensor(req).detach()
            
            assert target_tensor.size() == loaded_data.size(), (
                f"req {req.storage_index} mismatch sizes "
                f"{target_tensor.size()} vs {loaded_data.size()}"
            )
            target_tensor.copy_(loaded_data)
            planner.commit_tensor(req, target_tensor)
    
    fut: Future = Future()
    fut.set_result(None)
    return fut


# Alternative implementation using concurrent.futures for better control
def read_data_parallel_v2(
    self,
    plan: Any,
    planner: Any,
    max_workers: int = None,
) -> Future[None]:
    """
    Alternative parallel implementation using concurrent.futures.ProcessPoolExecutor.
    This version provides better resource management and exception handling.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # Group requests by file
    per_file: Dict[str, List] = {}
    for read_item in plan.items:
        item_md = self.storage_data[read_item.storage_index]
        path = item_md.relative_path
        per_file.setdefault(path, []).append(read_item)
    
    # Prepare worker arguments
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
            self.path,
            self.path,
        ))
    
    # Process files in parallel using ProcessPoolExecutor
    all_results = []
    
    if len(worker_args) == 1 or max_workers == 1:
        # Single file - process directly
        for args in worker_args:
            all_results.extend(_read_file_worker(args))
    else:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(worker_args))) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(_read_file_worker, args): args 
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_args):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as exc:
                    args = future_to_args[future]
                    raise RuntimeError(
                        f"Worker failed for file {args[0]}: {exc}"
                    ) from exc
    
    # Process results in the main process
    for req, data_type, loaded_data in all_results:
        if data_type == 'error':
            raise RuntimeError(f"Error loading data: {loaded_data}")
        elif data_type == 'bytes':
            planner.load_bytes(req, loaded_data)
        elif data_type == 'tensor':
            target_tensor = planner.resolve_tensor(req).detach()
            
            assert target_tensor.size() == loaded_data.size(), (
                f"req {req.storage_index} mismatch sizes "
                f"{target_tensor.size()} vs {loaded_data.size()}"
            )
            target_tensor.copy_(loaded_data)
            planner.commit_tensor(req, target_tensor)
    
    fut: Future = Future()
    fut.set_result(None)
    return fut
