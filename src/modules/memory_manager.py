"""
Memory Management Tool
Provides unified memory cleanup and monitoring functionality
"""

import gc
import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime


class MemoryManager:
    """Memory Manager"""
    
    def __init__(self, cleanup_interval: float = 60.0, max_history: int = 100):
        """
        Initialize Memory Manager
        
        Args:
            cleanup_interval: Automatic cleanup interval (seconds)
            max_history: Maximum number of history records
        """
        self.cleanup_interval = cleanup_interval
        self.max_history = max_history
        
        # Memory monitoring history
        self.memory_history = []
        
        # Registered objects
        self.registered_objects = []
        
        # Cleanup thread
        self.cleanup_thread = None
        self.should_cleanup = False
        
        # Statistics
        self.stats = {
            'total_cleanups': 0,
            'objects_cleaned': 0,
            'last_cleanup_time': None,
            'total_memory_freed': 0
        }
        
    def register_object(self, obj: Any, name: Optional[str] = None) -> str:
        """
        Register object for memory management
        
        Args:
            obj: Object to register
            name: Object name
            
        Returns:
            Object ID
        """
        obj_id = f"{name}_{id(obj)}_{time.time()}"
        
        self.registered_objects.append({
            'id': obj_id,
            'object': obj,
            'name': name or 'unnamed',
            'register_time': time.time(),
            'last_access': time.time()
        })
        
        return obj_id
        
    def unregister_object(self, obj_id: str) -> bool:
        """
        Unregister object
        
        Args:
            obj_id: Object ID
            
        Returns:
            Whether unregistration was successful
        """
        for i, obj_info in enumerate(self.registered_objects):
            if obj_info['id'] == obj_id:
                del self.registered_objects[i]
                return True
        return False
        
    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection
        
        Returns:
            Cleanup statistics
        """
        try:
            # Record memory state before cleanup
            gc.collect()
            stats_before = gc.get_stats()
            
            # Perform garbage collection
            collected = gc.collect()
            
            # Record memory state after cleanup
            stats_after = gc.get_stats()
            
            # Calculate collected objects
            objects_collected = 0
            for generation_before, generation_after in zip(stats_before, stats_after):
                objects_collected += generation_before['collected']
                
            # Clean up registered invalid objects
            cleaned_objects = 0
            valid_objects = []
            
            for obj_info in self.registered_objects:
                try:
                    # Check if object is still valid
                    obj_info['object'].__class__
                    obj_info['last_access'] = time.time()
                    valid_objects.append(obj_info)
                except (ReferenceError, AttributeError):
                    # Object has been garbage collected
                    cleaned_objects += 1
                    
            self.registered_objects = valid_objects
            
            # Update statistics
            self.stats['total_cleanups'] += 1
            self.stats['objects_cleaned'] += cleaned_objects + objects_collected
            self.stats['last_cleanup_time'] = time.time()
            
            # Record to history
            self.memory_history.append({
                'timestamp': time.time(),
                'objects_collected': objects_collected,
                'registered_objects_cleaned': cleaned_objects,
                'total_objects_collected': objects_collected + cleaned_objects
            })
            
            # Limit history length
            if len(self.memory_history) > self.max_history:
                self.memory_history = self.memory_history[-self.max_history:]
                
            return {
                'success': True,
                'objects_collected': objects_collected,
                'registered_objects_cleaned': cleaned_objects,
                'total_cleaned': objects_collected + cleaned_objects,
                'registered_objects_count': len(self.registered_objects)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'objects_collected': 0,
                'registered_objects_cleaned': 0,
                'total_cleaned': 0
            }
            
    def start_auto_cleanup(self) -> None:
        """Start auto-cleanup thread"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.should_cleanup = True
            self.cleanup_thread = threading.Thread(target=self._auto_cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            print(f"Memory auto-cleanup started (interval: {self.cleanup_interval}s)")
            
    def stop_auto_cleanup(self) -> None:
        """Stop auto-cleanup thread"""
        self.should_cleanup = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
            print("Memory auto-cleanup stopped")
            
    def _auto_cleanup_loop(self) -> None:
        """Auto-cleanup loop"""
        while self.should_cleanup:
            try:
                time.sleep(self.cleanup_interval)
                if self.should_cleanup:  # Check again to avoid being stopped during wait
                    self.force_garbage_collection()
            except Exception as e:
                print(f"Auto-cleanup error: {e}")
                
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            gc_stats = gc.get_stats()
            
            # Calculate total objects
            total_objects = sum(stat['count'] for stat in gc_stats)
            total_collected = sum(stat['collected'] for stat in gc_stats)
            
            return {
                'gc_stats': gc_stats,
                'total_objects': total_objects,
                'total_collected': total_collected,
                'registered_objects': len(self.registered_objects),
                'memory_history_count': len(self.memory_history),
                'cleanup_stats': self.stats.copy(),
                'last_cleanup': self.memory_history[-1] if self.memory_history else None
            }
        except Exception as e:
            return {'error': str(e)}
            
    def get_memory_report(self) -> str:
        """Generate memory report"""
        stats = self.get_memory_stats()
        
        if 'error' in stats:
            return f"Error generating memory report: {stats['error']}"
            
        report = [
            "=== Memory Management Report ===",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Registered Objects: {stats['registered_objects']}",
            f"Total Python Objects: {stats['total_objects']}",
            f"Total Collected Objects: {stats['total_collected']}",
            f"Memory History Records: {stats['memory_history_count']}",
            "",
            "Cleanup Statistics:",
            f"  Total Cleanups: {stats['cleanup_stats']['total_cleanups']}",
            f"  Total Objects Cleaned: {stats['cleanup_stats']['objects_cleaned']}",
            f"  Last Cleanup Time: {datetime.fromtimestamp(stats['cleanup_stats']['last_cleanup_time']).strftime('%Y-%m-%d %H:%M:%S') if stats['cleanup_stats']['last_cleanup_time'] else 'Never'}",
        ]
        
        # Add recent cleanup details
        if stats['last_cleanup']:
            last_cleanup = stats['last_cleanup']
            report.extend([
                "",
                "Recent Cleanup Details:",
                f"  Total Cleaned: {last_cleanup['total_cleaned']}",
                f"  GC Collected: {last_cleanup['objects_collected']}",
                f"  Registered Objects Cleaned: {last_cleanup['registered_objects_cleaned']}",
            ])
            
        # Add memory usage trend
        if len(self.memory_history) > 1:
            recent_cleanups = self.memory_history[-5:]
            avg_cleaned = sum(c['total_objects_collected'] for c in recent_cleanups) / len(recent_cleanups)
            report.extend([
                "",
                "Average of Last 5 Cleanups: {:.1f} objects/cleanup".format(avg_cleaned)
            ])
            
        return "\n".join(report)
        
    def __del__(self):
        """Destructor"""
        try:
            self.stop_auto_cleanup()
            self.force_garbage_collection()
        except:
            pass


# Global memory manager instance
_global_memory_manager = None


def get_global_memory_manager() -> MemoryManager:
    """Get global memory manager"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
        _global_memory_manager.start_auto_cleanup()
    return _global_memory_manager


def cleanup_memory() -> Dict[str, Any]:
    """Perform memory cleanup"""
    manager = get_global_memory_manager()
    return manager.force_garbage_collection()


def register_for_cleanup(obj: Any, name: Optional[str] = None) -> str:
    """Register object for memory management"""
    manager = get_global_memory_manager()
    return manager.register_object(obj, name)


def unregister_from_cleanup(obj_id: str) -> bool:
    """Unregister object from memory management"""
    manager = get_global_memory_manager()
    return manager.unregister_object(obj_id)


def get_memory_report() -> str:
    """Get memory report"""
    manager = get_global_memory_manager()
    return manager.get_memory_report()


if __name__ == "__main__":
    # Test Memory Manager
    manager = MemoryManager()
    
    print("=== Memory Manager Test ===")
    
    # Test garbage collection
    print("\nTesting garbage collection...")
    result = manager.force_garbage_collection()
    print(f"Cleanup result: {result}")
    
    # Test object registration
    print("\nTesting object registration...")
    test_data = [list(range(1000)) for _ in range(10)]
    obj_ids = []
    
    for i, data in enumerate(test_data):
        obj_id = manager.register_object(data, f"test_data_{i}")
        obj_ids.append(obj_id)
        
    print(f"Registered {len(obj_ids)} test objects")
    
    # Cleanup again
    print("\nCleaning registered objects...")
    result = manager.force_garbage_collection()
    print(f"Cleanup result: {result}")
    
    # Generate report
    print("\n" + manager.get_memory_report())
    
    # Cleanup
    del test_data
    del obj_ids
    manager.stop_auto_cleanup()