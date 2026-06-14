//! KV read 축(`KVReadStage`) 빌트인 등록 모듈.
//!
//! read 축의 빌트인 read stage(Quest)를 production binary 에 force-link 하고, fat-LTO
//! `--gc-sections` silent drop 을 startup self-test 로 fail-fast 한다. stage 축의
//! `ensure_builtin_stages_registered`(eviction) / format 축의 `ensure_builtin_kv_formats_registered`
//! 거울.

pub mod read_stage_registry;
