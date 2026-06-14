//! Persistent thread pool for KV cache preload operations.
//!
//! Eliminates per-token thread spawn/join overhead in `forward_into_offload`.
//! Workers are created once and reused across tokens via channel-based task dispatch.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::Result;

// `PreloadResult`/`PreloadAccess` 는 L2 로 격상되었다(§13.8-O INV-LAYER-003,
// `crate::preload_access`). 기존 `kv::offload::preload_pool::{PreloadResult,
// PreloadAccess}` 경로 참조를 깨지 않도록 re-export 한다. 구현체(`PreloadPool`)는
// 여기(pressure 도메인)에 잔존한다.
pub use crate::preload_access::{PreloadAccess, PreloadResult};

/// A preload task with type-erased cache pointer.
///
/// # Safety
/// The caller must ensure:
/// 1. `cache_ptr` points to a valid, properly aligned object
/// 2. The pointed-to object outlives the task execution
/// 3. No two concurrent tasks access the same `cache_ptr`
struct PreloadTask {
    cache_ptr: *mut (),
    preload_fn: unsafe fn(*mut ()) -> Result<()>,
    result_tx: mpsc::SyncSender<PreloadResult>,
}

// SAFETY: PreloadTask is sent across threads. The raw pointer is
// dereferenced only by one worker at a time, and the caller guarantees
// the pointed-to data outlives the task and has exclusive access.
// This matches the safety model of the previous thread::scope + raw pointer code.
unsafe impl Send for PreloadTask {}

/// Persistent thread pool for KV cache preload operations.
///
/// Workers block on a shared task queue and process preload requests.
/// Each submitted task gets a dedicated result channel, allowing the caller
/// to wait for specific layers in order.
pub struct PreloadPool {
    task_tx: Option<mpsc::Sender<PreloadTask>>,
    workers: Vec<thread::JoinHandle<()>>,
    size: usize,
}

impl PreloadPool {
    /// Create a pool with `num_workers` persistent threads.
    pub fn new(num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        let (task_tx, task_rx) = mpsc::channel::<PreloadTask>();
        let task_rx = Arc::new(Mutex::new(task_rx));

        let workers = (0..num_workers)
            .map(|id| {
                let rx = task_rx.clone();
                thread::Builder::new()
                    .name(format!("preload-{id}"))
                    .spawn(move || {
                        loop {
                            let task = {
                                let rx = rx.lock().unwrap();
                                rx.recv()
                            };
                            match task {
                                Ok(task) => {
                                    let t0 = Instant::now();
                                    let result = unsafe { (task.preload_fn)(task.cache_ptr) };
                                    let _ = task.result_tx.send(PreloadResult {
                                        result,
                                        duration: t0.elapsed(),
                                    });
                                }
                                Err(_) => break, // Channel closed → shutdown
                            }
                        }
                    })
                    .expect("failed to spawn preload worker")
            })
            .collect();

        PreloadPool {
            task_tx: Some(task_tx),
            workers,
            size: num_workers,
        }
    }

    /// Number of worker threads.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Submit a preload task. Returns a receiver for the one-shot result.
    ///
    /// # Safety
    /// - `cache_ptr` must point to a valid, properly aligned object
    /// - The object must remain valid until the result is received
    /// - No concurrent task may access the same `cache_ptr`
    /// - `preload_fn` must correctly cast `*mut ()` back to the original type
    pub unsafe fn submit(
        &self,
        cache_ptr: *mut (),
        preload_fn: unsafe fn(*mut ()) -> Result<()>,
    ) -> mpsc::Receiver<PreloadResult> {
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        if let Some(tx) = &self.task_tx {
            let _ = tx.send(PreloadTask {
                cache_ptr,
                preload_fn,
                result_tx,
            });
        }
        result_rx
    }
}

impl PreloadAccess for PreloadPool {
    fn size(&self) -> usize {
        self.size
    }

    unsafe fn submit_raw(
        &self,
        cache_ptr: *mut (),
        preload_fn: unsafe fn(*mut ()) -> anyhow::Result<()>,
    ) -> mpsc::Receiver<PreloadResult> {
        // SAFETY: delegated to PreloadPool::submit; all safety requirements
        // of PreloadAccess::submit_raw apply here unchanged.
        unsafe { self.submit(cache_ptr, preload_fn) }
    }
}

impl Drop for PreloadPool {
    fn drop(&mut self) {
        // Close the task channel (causes workers to exit)
        self.task_tx.take();
        // Join all worker threads
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Type-erased preload function for `PrefetchableCache` implementors.
///
/// # Safety
/// `ptr` must point to a valid, properly aligned `C: PrefetchableCache`.
pub unsafe fn preload_erased<C: crate::kv::kv_cache::PrefetchableCache>(
    ptr: *mut (),
) -> Result<()> {
    unsafe { (*(ptr as *mut C)).preload() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn test_pool_basic() {
        let pool = PreloadPool::new(2);
        assert_eq!(pool.size(), 2);

        // Submit a trivial task
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        COUNTER.store(0, Ordering::SeqCst);

        unsafe fn increment(ptr: *mut ()) -> Result<()> {
            unsafe {
                let counter = &*(ptr as *const AtomicUsize);
                counter.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }

        let rx = unsafe { pool.submit(&COUNTER as *const AtomicUsize as *mut (), increment) };
        let result = rx.recv().unwrap();
        assert!(result.result.is_ok());
        assert_eq!(COUNTER.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_pool_concurrent_tasks() {
        let pool = PreloadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));

        unsafe fn increment(ptr: *mut ()) -> Result<()> {
            unsafe {
                let counter = &*(ptr as *const AtomicUsize);
                counter.fetch_add(1, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(5));
            Ok(())
        }

        // Submit 8 tasks
        let receivers: Vec<_> = (0..8)
            .map(|_| unsafe { pool.submit(Arc::as_ptr(&counter) as *mut (), increment) })
            .collect();

        // Collect all results
        for rx in receivers {
            let result = rx.recv().unwrap();
            assert!(result.result.is_ok());
        }
        assert_eq!(counter.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn test_pool_drop_joins_workers() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        unsafe fn slow_task(ptr: *mut ()) -> Result<()> {
            unsafe {
                let counter = &*(ptr as *const AtomicUsize);
                std::thread::sleep(Duration::from_millis(20));
                counter.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }

        {
            let pool = PreloadPool::new(2);
            let _rx = unsafe { pool.submit(Arc::as_ptr(&counter_clone) as *mut (), slow_task) };
            // Drop pool here — should wait for the in-flight task
        }

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "task should complete before drop returns"
        );
    }

    #[test]
    fn test_pool_result_timing() {
        let pool = PreloadPool::new(1);

        unsafe fn sleep_task(_ptr: *mut ()) -> Result<()> {
            std::thread::sleep(Duration::from_millis(10));
            Ok(())
        }

        let rx = unsafe { pool.submit(std::ptr::null_mut(), sleep_task) };
        let result = rx.recv().unwrap();
        assert!(result.result.is_ok());
        assert!(
            result.duration >= Duration::from_millis(8),
            "duration should reflect actual work: {:?}",
            result.duration
        );
    }
}
