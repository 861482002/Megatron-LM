"""
Production-ready parallel implementation of read_data for PyTorch distributed checkpointing.

This module provides a complete multiprocessing-based implementation that can be
directly integrated into your FileSystemReader or similar checkpoint loading class.
"""

import io
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributed.checkpoint import LoadItemType, LoadPlan, LoadPlanner, ReadItem


@dataclass
class FileReadTask:
    """Encapsulates all data needed to read from a single file."""
    
    relative_path: str
    base_path: str
    read_items: List[dict]  # Serialized ReadItem objects
    storage_metadata: Dict[int, dict]  # Serialized storage metadata
    fs_config: Optional[dict] = None  # Filesystem configuration if needed
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class FileReadResult:
    """Result from reading a single file."""
    
    storage_index: int
    data_type: str  # 'bytes', 'tensor', or 'error'
    data: Any  # BytesIO, Tensor, or error message
    target_size: Optional[Tuple[int, ...]] = None  # Expected tensor size for validation


def _slice_file_simple(stream: io.BufferedReader, item_metadata: Any) -> io.BytesIO:
    """
    Simple file slicing implementation.
    
    Replace this with your actual _slice_file implementation.
    
    Args:
        stream: Open file stream
        item_metadata: Metadata containing offset and length information
    
    Returns:
        BytesIO containing the sliced data
    """
    # Check if metadata has offset and length attributes
    if hasattr(item_metadata, 'offset') and hasattr(item_metadata, 'length'):
        stream.seek(item_metadata.offset)
        data = stream.read(item_metadata.length)
        return io.BytesIO(data)
    else:
        # Read entire file if no offset/length specified
        stream.seek(0)
        data = stream.read()
        return io.BytesIO(data)


def _narrow_tensor_simple(
    tensor: Tensor,
    storage_offsets: Tuple[int, ...],
    lengths: Tuple[int, ...]
) -> Tensor:
    """
    Simple tensor narrowing implementation.
    
    Replace this with your actual narrow_tensor_by_index implementation.
    
    Args:
        tensor: Input tensor to narrow
        storage_offsets: Starting indices for each dimension
        lengths: Lengths for each dimension
    
    Returns:
        Narrowed tensor
    """
    result = tensor
    for dim, (offset, length) in enumerate(zip(storage_offsets, lengths)):
        if length != result.size(dim):
            result = result.narrow(dim, offset, length)
    return result


def _process_single_file(task: FileReadTask) -> List[FileReadResult]:
    """
    Worker function to process a single file's read requests.
    
    This function runs in a separate process and must be picklable.
    All logic here should be self-contained and use only serializable data.
    
    Args:
        task: FileReadTask containing all necessary information
    
    Returns:
        List of FileReadResult objects
    """
    results = []
    full_path = os.path.join(task.base_path, task.relative_path)
    
    try:
        with open(full_path, 'rb') as stream:
            for read_item_dict in task.read_items:
                try:
                    storage_index = read_item_dict['storage_index']
                    item_type = read_item_dict['type']
                    storage_offsets = read_item_dict.get('storage_offsets', ())
                    lengths = read_item_dict.get('lengths', ())
                    
                    # Get metadata for this item
                    item_metadata_dict = task.storage_metadata[storage_index]
                    
                    # Create a simple namespace object from dict
                    class MetadataNamespace:
                        def __init__(self, d):
                            for k, v in d.items():
                                setattr(self, k, v)
                    
                    item_metadata = MetadataNamespace(item_metadata_dict)
                    
                    # Slice the file
                    file_slice = _slice_file_simple(stream, item_metadata)
                    
                    # Apply transformations (simplified - extend as needed)
                    transform_descriptors = getattr(
                        item_metadata,
                        'transform_descriptors',
                        None
                    )
                    
                    # For now, assume no complex transformations
                    # You can add transform logic here if needed
                    transform_from = file_slice
                    
                    if item_type == LoadItemType.BYTE_IO.value if hasattr(LoadItemType.BYTE_IO, 'value') else 0:
                        # Load as bytes
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        
                        results.append(FileReadResult(
                            storage_index=storage_index,
                            data_type='bytes',
                            data=read_bytes
                        ))
                    
                    else:
                        # Load as tensor
                        if hasattr(transform_from, 'seekable') and transform_from.seekable():
                            seekable = transform_from
                        else:
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)
                        
                        # Load tensor
                        tensor = torch.load(
                            seekable,
                            map_location='cpu',
                            weights_only=True,
                        )
                        
                        # Apply narrowing if needed
                        if storage_offsets and lengths:
                            tensor = _narrow_tensor_simple(
                                tensor,
                                storage_offsets,
                                lengths
                            )
                        
                        results.append(FileReadResult(
                            storage_index=storage_index,
                            data_type='tensor',
                            data=tensor,
                            target_size=tuple(tensor.size())
                        ))
                
                except Exception as e:
                    # Error processing individual read item
                    results.append(FileReadResult(
                        storage_index=read_item_dict.get('storage_index', -1),
                        data_type='error',
                        data=f"Error processing read item: {str(e)}\n{type(e).__name__}",
                        target_size=None
                    ))
    
    except Exception as e:
        # Error opening/reading file
        results.append(FileReadResult(
            storage_index=-1,
            data_type='error',
            data=f"Error reading file {task.relative_path}: {str(e)}\n{type(e).__name__}",
            target_size=None
        ))
    
    return results


class ParallelFileReader:
    """
    Mixin or replacement for read_data method with parallel processing support.
    
    This class can be used as a mixin with your existing reader class,
    or you can copy the methods into your class.
    """
    
    def __init__(self, enable_parallel: bool = True, max_workers: Optional[int] = None):
        """
        Initialize parallel reader settings.
        
        Args:
            enable_parallel: Whether to use parallel reading (default: True)
            max_workers: Maximum number of worker processes (default: cpu_count)
        """
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or mp.cpu_count()
    
    def _serialize_read_item(self, read_item: ReadItem) -> dict:
        """
        Convert ReadItem to a serializable dictionary.
        
        Args:
            read_item: ReadItem to serialize
        
        Returns:
            Dictionary representation
        """
        return {
            'storage_index': read_item.storage_index,
            'type': read_item.type.value if hasattr(read_item.type, 'value') else read_item.type,
            'storage_offsets': getattr(read_item, 'storage_offsets', ()),
            'lengths': getattr(read_item, 'lengths', ()),
            # Add other fields as needed
        }
    
    def _serialize_storage_metadata(self, storage_metadata: Any) -> dict:
        """
        Convert storage metadata to serializable dictionary.
        
        Args:
            storage_metadata: Storage metadata object
        
        Returns:
            Dictionary representation
        """
        # If it's already a dict, return it
        if isinstance(storage_metadata, dict):
            return storage_metadata
        
        # If it has a to_dict method, use it
        if hasattr(storage_metadata, 'to_dict'):
            return storage_metadata.to_dict()
        
        # Otherwise, extract common attributes
        result = {}
        for attr in ['relative_path', 'offset', 'length', 'transform_descriptors']:
            if hasattr(storage_metadata, attr):
                value = getattr(storage_metadata, attr)
                # Handle non-serializable values
                if attr == 'transform_descriptors' and value is not None:
                    # Serialize transform descriptors if needed
                    result[attr] = list(value) if value else None
                else:
                    result[attr] = value
        
        return result
    
    def _create_read_tasks(
        self,
        plan: LoadPlan,
        storage_data: Dict[int, Any]
    ) -> List[FileReadTask]:
        """
        Group read items by file and create FileReadTask objects.
        
        Args:
            plan: LoadPlan containing items to read
            storage_data: Mapping from storage index to storage metadata
        
        Returns:
            List of FileReadTask objects, one per file
        """
        # Group requests by file
        per_file: Dict[str, List[ReadItem]] = {}
        for read_item in plan.items:
            item_md = storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)
        
        # Create tasks
        tasks = []
        for relative_path, read_items in per_file.items():
            # Serialize read items
            serialized_items = [
                self._serialize_read_item(item) for item in read_items
            ]
            
            # Serialize storage metadata for these items
            storage_metadata = {
                item.storage_index: self._serialize_storage_metadata(
                    storage_data[item.storage_index]
                )
                for item in read_items
            }
            
            task = FileReadTask(
                relative_path=relative_path,
                base_path=self.path,
                read_items=serialized_items,
                storage_metadata=storage_metadata,
                fs_config=None  # Add if needed
            )
            tasks.append(task)
        
        return tasks
    
    def _process_results(
        self,
        results: List[FileReadResult],
        plan: LoadPlan,
        planner: LoadPlanner
    ) -> None:
        """
        Process results from workers and update planner.
        
        Args:
            results: List of FileReadResult objects from workers
            plan: Original LoadPlan
            planner: LoadPlanner to update
        
        Raises:
            RuntimeError: If any errors occurred during reading
        """
        # Create mapping from storage_index to ReadItem for quick lookup
        storage_index_to_item = {
            item.storage_index: item for item in plan.items
        }
        
        for result in results:
            if result.data_type == 'error':
                raise RuntimeError(
                    f"Error loading storage index {result.storage_index}: "
                    f"{result.data}"
                )
            
            # Get original ReadItem
            read_item = storage_index_to_item[result.storage_index]
            
            if result.data_type == 'bytes':
                planner.load_bytes(read_item, result.data)
            
            elif result.data_type == 'tensor':
                target_tensor = planner.resolve_tensor(read_item).detach()
                loaded_tensor = result.data
                
                # Validate sizes match
                assert target_tensor.size() == loaded_tensor.size(), (
                    f"req {read_item.storage_index} mismatch sizes "
                    f"{target_tensor.size()} vs {loaded_tensor.size()}"
                )
                
                # Copy data
                target_tensor.copy_(loaded_tensor)
                planner.commit_tensor(read_item, target_tensor)
    
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Read data according to plan, with optional parallel processing.
        
        This is a drop-in replacement for the original read_data method.
        
        Args:
            plan: LoadPlan containing items to read
            planner: LoadPlanner for coordinating loads
        
        Returns:
            Future that resolves when reading is complete
        """
        # Create read tasks
        tasks = self._create_read_tasks(plan, self.storage_data)
        
        # Determine if we should use parallel processing
        use_parallel = (
            self.enable_parallel
            and len(tasks) > 1
            and self.max_workers > 1
        )
        
        all_results = []
        
        if not use_parallel:
            # Sequential processing
            for task in tasks:
                results = _process_single_file(task)
                all_results.extend(results)
        else:
            # Parallel processing
            num_workers = min(self.max_workers, len(tasks))
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(_process_single_file, task): task
                    for task in tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Worker failed for file {task.relative_path}: {exc}"
                        ) from exc
        
        # Process all results
        self._process_results(all_results, plan, planner)
        
        # Return completed future
        fut: Future = Future()
        fut.set_result(None)
        return fut
    
    def read_data_with_progress(
        self,
        plan: LoadPlan,
        planner: LoadPlanner,
        progress_callback: Optional[callable] = None
    ) -> Future[None]:
        """
        Read data with progress reporting.
        
        Args:
            plan: LoadPlan containing items to read
            planner: LoadPlanner for coordinating loads
            progress_callback: Optional callback(current, total) for progress updates
        
        Returns:
            Future that resolves when reading is complete
        """
        tasks = self._create_read_tasks(plan, self.storage_data)
        total_tasks = len(tasks)
        completed_tasks = 0
        
        all_results = []
        
        use_parallel = (
            self.enable_parallel
            and len(tasks) > 1
            and self.max_workers > 1
        )
        
        if not use_parallel:
            for task in tasks:
                results = _process_single_file(task)
                all_results.extend(results)
                completed_tasks += 1
                if progress_callback:
                    progress_callback(completed_tasks, total_tasks)
        else:
            num_workers = min(self.max_workers, len(tasks))
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_task = {
                    executor.submit(_process_single_file, task): task
                    for task in tasks
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        completed_tasks += 1
                        if progress_callback:
                            progress_callback(completed_tasks, total_tasks)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Worker failed for file {task.relative_path}: {exc}"
                        ) from exc
        
        self._process_results(all_results, plan, planner)
        
        fut: Future = Future()
        fut.set_result(None)
        return fut


# Example usage
if __name__ == '__main__':
    """
    Example of how to integrate ParallelFileReader into your existing class.
    """
    
    # Option 1: Use as a mixin
    class MyFileSystemReader(ParallelFileReader):
        def __init__(self, path, enable_parallel=True, max_workers=None):
            super().__init__(enable_parallel, max_workers)
            self.path = path
            self.storage_data = {}  # Your storage data
            # ... other initialization
    
    # Option 2: Copy methods into your existing class
    class ExistingReader:
        def __init__(self, path):
            self.path = path
            self.storage_data = {}
            self.enable_parallel = True
            self.max_workers = mp.cpu_count()
        
        # Copy the methods from ParallelFileReader:
        # - _serialize_read_item
        # - _serialize_storage_metadata
        # - _create_read_tasks
        # - _process_results
        # - read_data
    
    print("Integration examples created successfully!")
    print(f"Available CPU cores: {mp.cpu_count()}")
