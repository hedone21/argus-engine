// Backward-compat re-export (B-4 sprint, 1 sprint 한정). 다음 cleanup sprint에서 제거.
#[deprecated(
    note = "observability::eval moved to session::eval (B-4 sprint, §13.8-I). Update imports."
)]
pub use crate::session::eval;
pub mod profile;
pub mod rss_trace;
