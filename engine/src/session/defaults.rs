//! No-op default for the optional command source slot.
//!
//! `DecodeLoopBuilder::build` substitutes [`NoOpCommandSource`] whenever the
//! caller did not provide one (`with_cmd_source` / `with_resilience`).
//!
//! Phase β-7 removed the other no-op slots (eviction / swap / observer / report
//! / tick) along with their v1 traits — those responsibilities moved to the
//! `PipelineStage` registry.

use argus_shared::EngineCommand;

use crate::session::command_dispatcher::CommandSource;

/// Command source that never yields a command.
pub struct NoOpCommandSource;

impl CommandSource for NoOpCommandSource {
    fn poll(&mut self) -> anyhow::Result<Vec<EngineCommand>> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_command_source_yields_empty_commands() {
        let mut s = NoOpCommandSource;
        let cmds = s.poll().unwrap();
        assert!(cmds.is_empty());
    }
}
