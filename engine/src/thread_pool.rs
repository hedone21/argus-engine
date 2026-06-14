//! Low-latency thread pool for matmul dispatch.
//!
//! Workers park between dispatches (zero CPU when idle) and are woken
//! via thread::unpark (~1-3µs). During dispatch, work-stealing via
//! atomic fetch_add naturally balances load across heterogeneous cores.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Function type for work items: (context_pointer, chunk_id)
pub type WorkFn = unsafe fn(*const u8, usize);

/// 64-byte aligned atomic wrapper to prevent false sharing.
/// Each hot field gets its own cache line (ggml GGML_CACHE_ALIGN pattern).
#[repr(C, align(64))]
struct CacheAligned<T>(T);

#[repr(C)]
struct SharedState {
    /// Incremented each dispatch — workers check this after unpark.
    generation: CacheAligned<AtomicU64>,
    /// Workers fetch_add to grab the next chunk (work-stealing).
    next_chunk: CacheAligned<AtomicUsize>,
    /// Workers increment when done with current dispatch.
    done_count: CacheAligned<AtomicUsize>,
    // --- cold fields below (shared cache line OK) ---
    /// Total chunks for current dispatch.
    total_chunks: AtomicUsize,
    /// Shutdown signal.
    shutdown: AtomicBool,
    /// When true, workers spin instead of park after completing work.
    /// Used during batch dispatches to keep workers hot.
    batch_mode: AtomicBool,
    /// Number of worker threads (excludes main thread).
    n_workers: usize,
    /// Work function pointer (stored as usize for atomicity).
    work_fn: AtomicUsize,
    /// Work context pointer (stored as usize for atomicity).
    work_ctx: AtomicUsize,
}

pub struct SpinPool {
    shared: Arc<SharedState>,
    worker_threads: Vec<std::thread::Thread>,
    _join_handles: Vec<std::thread::JoinHandle<()>>,
}

impl SpinPool {
    /// Create a pool with `n_workers` background threads.
    /// The main thread also participates during dispatch (total = n_workers + 1).
    pub fn new(n_workers: usize) -> Self {
        let shared = Arc::new(SharedState {
            generation: CacheAligned(AtomicU64::new(0)),
            next_chunk: CacheAligned(AtomicUsize::new(0)),
            done_count: CacheAligned(AtomicUsize::new(0)),
            total_chunks: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            batch_mode: AtomicBool::new(false),
            n_workers,
            work_fn: AtomicUsize::new(0),
            work_ctx: AtomicUsize::new(0),
        });

        let (tx, rx) = std::sync::mpsc::channel();
        let mut join_handles = Vec::with_capacity(n_workers);

        for _ in 0..n_workers {
            let s = shared.clone();
            let tx = tx.clone();
            join_handles.push(std::thread::spawn(move || {
                // Send our Thread handle back for unpark, then drop sender
                // so rx.into_iter() terminates after all handles are collected.
                tx.send(std::thread::current()).unwrap();
                drop(tx);
                Self::worker_loop(&s);
            }));
        }
        drop(tx);

        let worker_threads: Vec<_> = rx.into_iter().collect();

        SpinPool {
            shared,
            worker_threads,
            _join_handles: join_handles,
        }
    }

    fn worker_loop(shared: &SharedState) {
        let mut last_gen = 0u64;

        loop {
            // Hybrid wait: brief spin then park.
            // Spin catches rapid back-to-back dispatches (matmul chains).
            // Park yields CPU for non-matmul ops after the spin window.
            let mut spins = 0u32;
            loop {
                if shared.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                let g = shared.generation.0.load(Ordering::Acquire);
                if g != last_gen {
                    last_gen = g;
                    break;
                }
                spins += 1;
                if spins < 500 || shared.batch_mode.load(Ordering::Relaxed) {
                    // In batch mode: pure spin (keep workers hot between dispatches)
                    // Normal mode: brief spin then park
                    std::hint::spin_loop();
                    if spins > 500 {
                        spins = 500; // prevent overflow
                    }
                } else {
                    std::thread::park();
                    spins = 0;
                }
            }

            // Work-stealing: grab chunks via atomic counter
            Self::steal_work(shared);

            shared.done_count.0.fetch_add(1, Ordering::Release);
        }
    }

    #[inline]
    fn steal_work(shared: &SharedState) {
        let work_fn: WorkFn =
            unsafe { std::mem::transmute(shared.work_fn.load(Ordering::Relaxed)) };
        let work_ctx = shared.work_ctx.load(Ordering::Relaxed) as *const u8;
        let total = shared.total_chunks.load(Ordering::Relaxed);

        loop {
            let chunk = shared.next_chunk.0.fetch_add(1, Ordering::Relaxed);
            if chunk >= total {
                break;
            }
            unsafe {
                work_fn(work_ctx, chunk);
            }
        }
    }

    /// Enter batch mode: workers will spin instead of park after finishing work.
    /// Call before a sequence of rapid dispatch() calls (e.g., QKV matmuls).
    pub fn begin_batch(&self) {
        self.shared.batch_mode.store(true, Ordering::Release);
    }

    /// Exit batch mode: workers will park normally after finishing work.
    pub fn end_batch(&self) {
        self.shared.batch_mode.store(false, Ordering::Release);
    }

    /// Dispatch a batch of work items sequentially, keeping workers spinning
    /// between items (no park/unpark). Emulates ggml's graph execution model.
    ///
    /// # Safety
    /// All work_fn/ctx pairs must be valid for their respective chunk ranges.
    pub unsafe fn dispatch_batch(&self, items: &[(WorkFn, *const u8, usize)]) {
        if items.is_empty() {
            return;
        }

        // Wake workers for first item
        let (wf, ctx, nc) = items[0];
        self.shared.work_fn.store(wf as usize, Ordering::Relaxed);
        self.shared.work_ctx.store(ctx as usize, Ordering::Relaxed);
        self.shared.total_chunks.store(nc, Ordering::Relaxed);
        self.shared.next_chunk.0.store(0, Ordering::Relaxed);
        self.shared.done_count.0.store(0, Ordering::Relaxed);
        self.shared.generation.0.fetch_add(1, Ordering::Release);
        for wt in &self.worker_threads {
            wt.unpark();
        }
        Self::steal_work(&self.shared);
        while self.shared.done_count.0.load(Ordering::Acquire) < self.shared.n_workers {
            std::hint::spin_loop();
        }

        // Subsequent items: workers are spinning (within spin window),
        // just update work params and bump generation — no unpark needed.
        for &(wf, ctx, nc) in &items[1..] {
            self.shared.work_fn.store(wf as usize, Ordering::Relaxed);
            self.shared.work_ctx.store(ctx as usize, Ordering::Relaxed);
            self.shared.total_chunks.store(nc, Ordering::Relaxed);
            self.shared.next_chunk.0.store(0, Ordering::Relaxed);
            self.shared.done_count.0.store(0, Ordering::Relaxed);
            self.shared.generation.0.fetch_add(1, Ordering::Release);
            // Workers are still spinning — they'll catch the generation change
            Self::steal_work(&self.shared);
            while self.shared.done_count.0.load(Ordering::Acquire) < self.shared.n_workers {
                std::hint::spin_loop();
            }
        }
    }

    /// Dispatch `n_chunks` work items. Blocks until all are processed.
    ///
    /// # Safety
    /// `work_fn(ctx, chunk_id)` must be safe to call for all chunk_id in 0..n_chunks.
    /// `ctx` must remain valid until dispatch returns.
    pub unsafe fn dispatch(&self, n_chunks: usize, work_fn: WorkFn, ctx: *const u8) {
        if n_chunks == 0 {
            return;
        }

        // Set up work (Relaxed: visible via generation Release below)
        self.shared
            .work_fn
            .store(work_fn as usize, Ordering::Relaxed);
        self.shared.work_ctx.store(ctx as usize, Ordering::Relaxed);
        self.shared.total_chunks.store(n_chunks, Ordering::Relaxed);
        self.shared.next_chunk.0.store(0, Ordering::Relaxed);
        self.shared.done_count.0.store(0, Ordering::Relaxed);

        // Publish work (Release ensures setup is visible to workers)
        self.shared.generation.0.fetch_add(1, Ordering::Release);

        // Wake all parked workers
        for wt in &self.worker_threads {
            wt.unpark();
        }

        // Main thread also steals work
        Self::steal_work(&self.shared);

        // Wait for all workers to finish
        while self.shared.done_count.0.load(Ordering::Acquire) < self.shared.n_workers {
            std::hint::spin_loop();
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.0.fetch_add(1, Ordering::Release);
        for wt in &self.worker_threads {
            wt.unpark();
        }
        for h in self._join_handles.drain(..) {
            let _ = h.join();
        }
    }
}

// Global singleton — initialized lazily on first matmul call.
static POOL: std::sync::OnceLock<SpinPool> = std::sync::OnceLock::new();

/// Get the global SpinPool.
pub fn get_pool() -> &'static SpinPool {
    POOL.get_or_init(|| {
        let n = rayon::current_num_threads();
        // n-1 workers + main = n total. Park/unpark means idle workers
        // don't consume CPU, so we can safely use all available threads.
        SpinPool::new(n.saturating_sub(1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_basic() {
        let pool = SpinPool::new(3);
        let counter = AtomicU64::new(0);

        unsafe fn add_work(ctx: *const u8, _chunk_id: usize) {
            // SAFETY: ctx is a valid pointer to AtomicU64 for the duration of dispatch.
            let c = unsafe { &*(ctx as *const AtomicU64) };
            c.fetch_add(1, Ordering::Relaxed);
        }

        unsafe {
            pool.dispatch(100, add_work, &counter as *const _ as *const u8);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_work_stealing() {
        let pool = SpinPool::new(4);
        let results: Vec<AtomicU64> = (0..1000).map(|_| AtomicU64::new(0)).collect();

        unsafe fn mark_work(ctx: *const u8, chunk_id: usize) {
            // SAFETY: ctx is a valid pointer to Vec<AtomicU64> for the duration of dispatch.
            let results = unsafe { &*(ctx as *const Vec<AtomicU64>) };
            results[chunk_id].fetch_add(1, Ordering::Relaxed);
        }

        unsafe {
            pool.dispatch(1000, mark_work, &results as *const _ as *const u8);
        }

        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.load(Ordering::Relaxed),
                1,
                "chunk {} processed {} times",
                i,
                r.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn test_cache_line_alignment() {
        // Hot atomic fields must sit on separate 64-byte cache lines
        // to prevent false sharing (ggml Issue #9588: +30% on ARM).
        let state = SharedState {
            generation: CacheAligned(AtomicU64::new(0)),
            next_chunk: CacheAligned(AtomicUsize::new(0)),
            done_count: CacheAligned(AtomicUsize::new(0)),
            total_chunks: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            batch_mode: AtomicBool::new(false),
            n_workers: 0,
            work_fn: AtomicUsize::new(0),
            work_ctx: AtomicUsize::new(0),
        };

        let base = &state as *const _ as usize;
        let gen_off = &state.generation.0 as *const _ as usize - base;
        let next_off = &state.next_chunk.0 as *const _ as usize - base;
        let done_off = &state.done_count.0 as *const _ as usize - base;

        // Each hot field on a distinct 64-byte cache line
        assert_ne!(
            gen_off / 64,
            next_off / 64,
            "generation and next_chunk share a cache line"
        );
        assert_ne!(
            next_off / 64,
            done_off / 64,
            "next_chunk and done_count share a cache line"
        );
        assert_ne!(
            gen_off / 64,
            done_off / 64,
            "generation and done_count share a cache line"
        );
    }

    #[test]
    fn test_multiple_dispatches() {
        let pool = SpinPool::new(2);
        let counter = AtomicU64::new(0);

        unsafe fn add_work(ctx: *const u8, _chunk_id: usize) {
            // SAFETY: ctx is a valid pointer to AtomicU64 for the duration of dispatch.
            let c = unsafe { &*(ctx as *const AtomicU64) };
            c.fetch_add(1, Ordering::Relaxed);
        }

        for _ in 0..50 {
            unsafe {
                pool.dispatch(20, add_work, &counter as *const _ as *const u8);
            }
        }
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }
}
