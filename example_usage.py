"""
示例：如何将并行读取集成到您的代码中

这个文件展示了几种不同的集成方式和使用场景。
"""

import io
import multiprocessing as mp
from concurrent.futures import Future
from typing import Any, Dict

import torch
from torch.distributed.checkpoint import LoadPlan, LoadPlanner

# 导入并行实现
from parallel_reader_implementation import ParallelFileReader, _process_single_file


# ============================================================================
# 示例 1: 将原始函数改为并行版本（最小改动）
# ============================================================================

def read_data_original(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
    """原始的顺序实现（您提供的代码）"""
    # group requests by file
    per_file: dict[str, list] = {}
    for read_item in plan.items:
        item_md = self.storage_data[read_item.storage_index]
        path = item_md.relative_path
        per_file.setdefault(path, []).append(read_item)

    for relative_path, reqs in per_file.items():
        new_path = self.fs.concat_path(self.path, relative_path)
        with self.fs.create_stream(new_path, "rb") as stream:
            for req in reqs:
                item_md = self.storage_data[req.storage_index]
                file_slice = self._slice_file(stream, item_md)
                transform_from = self.transforms.transform_load_stream(
                    req,
                    item_md.transform_descriptors or (),
                    file_slice,
                )

                if req.type == LoadItemType.BYTE_IO:
                    read_bytes = io.BytesIO(transform_from.read(-1))
                    read_bytes.seek(0)
                    planner.load_bytes(req, read_bytes)
                else:
                    if transform_from.seekable():
                        seekable = transform_from
                    else:
                        seekable = io.BytesIO(transform_from.read(-1))
                        seekable.seek(0)

                    tensor = torch.load(
                        seekable,
                        map_location="cpu",
                        weights_only=True,
                    )
                    tensor = narrow_tensor_by_index(
                        tensor, req.storage_offsets, req.lengths
                    )
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert target_tensor.size() == tensor.size()
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

    fut: Future = Future()
    fut.set_result(None)
    return fut


def read_data_parallel_simple(self, plan: LoadPlan, planner: LoadPlanner, 
                              num_workers: int = None) -> Future[None]:
    """
    简单的并行版本 - 在原始代码基础上最小改动
    
    主要变化：
    1. 添加 num_workers 参数
    2. 将文件处理逻辑提取到工作函数
    3. 使用进程池并行处理
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # group requests by file (与原始相同)
    per_file: dict[str, list] = {}
    for read_item in plan.items:
        item_md = self.storage_data[read_item.storage_index]
        path = item_md.relative_path
        per_file.setdefault(path, []).append(read_item)
    
    # 如果只有一个文件，直接使用原始方法
    if len(per_file) <= 1:
        return read_data_original(self, plan, planner)
    
    # 准备工作任务
    from parallel_reader_implementation import FileReadTask
    
    tasks = []
    for relative_path, read_items in per_file.items():
        # 序列化必要数据
        serialized_items = [
            {
                'storage_index': item.storage_index,
                'type': item.type.value if hasattr(item.type, 'value') else item.type,
                'storage_offsets': getattr(item, 'storage_offsets', ()),
                'lengths': getattr(item, 'lengths', ()),
            }
            for item in read_items
        ]
        
        storage_metadata = {
            item.storage_index: {
                'relative_path': self.storage_data[item.storage_index].relative_path,
                'offset': getattr(self.storage_data[item.storage_index], 'offset', 0),
                'length': getattr(self.storage_data[item.storage_index], 'length', -1),
            }
            for item in read_items
        }
        
        task = FileReadTask(
            relative_path=relative_path,
            base_path=self.path,
            read_items=serialized_items,
            storage_metadata=storage_metadata,
        )
        tasks.append(task)
    
    # 并行处理
    all_results = []
    with ProcessPoolExecutor(max_workers=min(num_workers, len(tasks))) as executor:
        future_to_task = {
            executor.submit(_process_single_file, task): task
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            results = future.result()
            all_results.extend(results)
    
    # 处理结果（与原始类似，但批量处理）
    storage_index_to_item = {item.storage_index: item for item in plan.items}
    
    for result in all_results:
        if result.data_type == 'error':
            raise RuntimeError(f"Error loading: {result.data}")
        
        read_item = storage_index_to_item[result.storage_index]
        
        if result.data_type == 'bytes':
            planner.load_bytes(read_item, result.data)
        elif result.data_type == 'tensor':
            target_tensor = planner.resolve_tensor(read_item).detach()
            assert target_tensor.size() == result.data.size()
            target_tensor.copy_(result.data)
            planner.commit_tensor(read_item, target_tensor)
    
    fut: Future = Future()
    fut.set_result(None)
    return fut


# ============================================================================
# 示例 2: 使用 ParallelFileReader 作为 Mixin
# ============================================================================

class MyFileSystemReader(ParallelFileReader):
    """
    继承 ParallelFileReader 来获得并行读取能力
    """
    
    def __init__(self, path: str, enable_parallel: bool = True, max_workers: int = None):
        # 初始化并行功能
        super().__init__(enable_parallel, max_workers)
        
        # 您的自定义初始化
        self.path = path
        self.storage_data: Dict[int, Any] = {}
        self.fs = None  # 您的文件系统对象
        self.transforms = None  # 您的转换对象
        
        # 加载元数据等
        self._load_metadata()
    
    def _load_metadata(self):
        """加载检查点元数据"""
        # 您的元数据加载逻辑
        pass
    
    def _slice_file(self, stream, item_metadata):
        """您的文件切片实现"""
        # 实现您的逻辑
        pass
    
    # read_data 方法已经从 ParallelFileReader 继承
    # 可以直接使用，或者覆盖以添加自定义逻辑


# ============================================================================
# 示例 3: 在现有类中添加并行功能
# ============================================================================

class ExistingFileSystemReader:
    """
    现有的文件系统读取器，添加并行功能
    """
    
    def __init__(self, path: str, enable_parallel: bool = True, max_workers: int = None):
        self.path = path
        self.storage_data: Dict[int, Any] = {}
        self.fs = None
        self.transforms = None
        
        # 并行配置
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or mp.cpu_count()
    
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        支持并行的 read_data 实现
        
        根据配置自动选择并行或顺序执行
        """
        if self.enable_parallel:
            return self._read_data_parallel(plan, planner)
        else:
            return self._read_data_sequential(plan, planner)
    
    def _read_data_sequential(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """原始的顺序实现"""
        # 您的原始实现
        pass
    
    def _read_data_parallel(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """并行实现 - 从 ParallelFileReader 复制相关方法"""
        # 复制以下方法到这个类：
        # - _serialize_read_item
        # - _serialize_storage_metadata
        # - _create_read_tasks
        # - _process_results
        # 然后实现并行逻辑
        pass


# ============================================================================
# 示例 4: 带进度显示的使用
# ============================================================================

def example_with_progress():
    """展示如何使用进度回调"""
    
    def progress_callback(current: int, total: int):
        """进度回调函数"""
        percentage = (current / total) * 100
        print(f"\rLoading checkpoint: {current}/{total} files ({percentage:.1f}%)", end='')
        if current == total:
            print()  # 完成后换行
    
    # 创建读取器
    reader = MyFileSystemReader(
        path="/path/to/checkpoint",
        enable_parallel=True,
        max_workers=8
    )
    
    # 使用进度回调读取数据
    future = reader.read_data_with_progress(
        plan=your_plan,
        planner=your_planner,
        progress_callback=progress_callback
    )
    
    # 等待完成
    future.result()
    print("Checkpoint loaded successfully!")


# ============================================================================
# 示例 5: 根据检查点大小自适应选择
# ============================================================================

class AdaptiveFileSystemReader(ParallelFileReader):
    """
    根据检查点特征自动选择最佳策略
    """
    
    def __init__(self, path: str, max_workers: int = None):
        super().__init__(enable_parallel=True, max_workers=max_workers)
        self.path = path
        self.storage_data: Dict[int, Any] = {}
    
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        智能选择并行或顺序执行
        """
        # 统计文件数量和大小
        num_files = len(set(
            self.storage_data[item.storage_index].relative_path
            for item in plan.items
        ))
        
        # 估算总大小（如果可用）
        total_size_mb = sum(
            getattr(self.storage_data[item.storage_index], 'length', 0)
            for item in plan.items
        ) / (1024 * 1024)
        
        # 决策逻辑
        if num_files == 1:
            # 单文件：使用顺序
            self.enable_parallel = False
            print("Using sequential read (single file)")
        elif total_size_mb < 10:
            # 小检查点：使用顺序
            self.enable_parallel = False
            print(f"Using sequential read (small checkpoint: {total_size_mb:.1f}MB)")
        else:
            # 大检查点：使用并行
            self.enable_parallel = True
            # 根据文件大小调整工作进程数
            avg_file_size = total_size_mb / num_files
            if avg_file_size > 100:
                # 大文件：使用较少的工作进程
                self.max_workers = min(self.max_workers, num_files, 4)
            else:
                # 中等文件：使用更多工作进程
                self.max_workers = min(self.max_workers, num_files)
            
            print(f"Using parallel read ({num_files} files, {self.max_workers} workers)")
        
        # 调用父类方法
        return super().read_data(plan, planner)


# ============================================================================
# 示例 6: 性能测试和比较
# ============================================================================

def benchmark_parallel_vs_sequential():
    """
    性能测试：比较并行和顺序实现
    """
    import time
    
    checkpoint_path = "/path/to/checkpoint"
    
    # 准备测试数据
    plan = create_test_plan()
    
    print("=" * 60)
    print("Performance Benchmark: Parallel vs Sequential")
    print("=" * 60)
    
    # 测试顺序读取
    print("\n1. Sequential Read:")
    planner_seq = create_planner()
    reader_seq = MyFileSystemReader(
        checkpoint_path,
        enable_parallel=False
    )
    
    start_time = time.time()
    reader_seq.read_data(plan, planner_seq)
    seq_time = time.time() - start_time
    
    print(f"   Time: {seq_time:.2f}s")
    
    # 测试不同工作进程数的并行读取
    print("\n2. Parallel Read (different worker counts):")
    
    for num_workers in [2, 4, 8, 16]:
        if num_workers > mp.cpu_count():
            continue
        
        planner_par = create_planner()
        reader_par = MyFileSystemReader(
            checkpoint_path,
            enable_parallel=True,
            max_workers=num_workers
        )
        
        start_time = time.time()
        reader_par.read_data(plan, planner_par)
        par_time = time.time() - start_time
        
        speedup = seq_time / par_time
        print(f"   {num_workers} workers: {par_time:.2f}s (speedup: {speedup:.2f}x)")
    
    print("\n" + "=" * 60)


# ============================================================================
# 辅助函数
# ============================================================================

def create_test_plan():
    """创建测试用的 LoadPlan"""
    # 实现您的测试数据创建逻辑
    pass


def create_planner():
    """创建测试用的 LoadPlanner"""
    # 实现您的 planner 创建逻辑
    pass


# ============================================================================
# 主函数：运行示例
# ============================================================================

if __name__ == '__main__':
    print("并行读取数据示例")
    print("=" * 60)
    
    # 显示系统信息
    print(f"\n系统信息:")
    print(f"  CPU 核心数: {mp.cpu_count()}")
    
    try:
        import psutil
        mem_gb = psutil.virtual_memory().total / (1024**3)
        print(f"  总内存: {mem_gb:.1f} GB")
    except ImportError:
        print("  (安装 psutil 可显示内存信息)")
    
    print("\n" + "=" * 60)
    print("\n可用示例:")
    print("  1. read_data_parallel_simple - 简单的并行版本")
    print("  2. MyFileSystemReader - 使用 Mixin 的实现")
    print("  3. ExistingFileSystemReader - 在现有类中添加并行")
    print("  4. example_with_progress - 带进度显示的使用")
    print("  5. AdaptiveFileSystemReader - 自适应选择策略")
    print("  6. benchmark_parallel_vs_sequential - 性能测试")
    
    print("\n选择您需要的示例集成到您的代码中！")
    print("=" * 60)
