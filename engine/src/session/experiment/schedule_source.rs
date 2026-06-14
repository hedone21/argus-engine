//! [`ScheduleCommandSource`] — 정적 `ExperimentSchedule` 을 β-4 `CommandSource` seam 으로 연결.
//!
//! `ExperimentSchedule` 의 JSON directive schedule 을 token-pos 기반으로 drain 하여
//! `DecodeLoop` 의 `CommandDispatcher` 에 전달한다. legacy mpsc/`CommandExecutor::poll` 채널
//! **없이** `CommandSource::poll` 1개로만 구현한다.
//!
//! ## poll ↔ token-pos 시맨틱
//!
//! `DecodeLoop::run` 은 매 decode step 에서 `cmd_source.poll()` 을 1회 호출한 뒤
//! forward 를 수행한다 (`decode_loop.rs` run 루프). 따라서 `poll()` 호출 시점의
//! `cur_token` 은 **직후에 생성될 토큰의 위치**와 대응한다.
//!
//! legacy `directives_at(step)` 시맨틱: `at_token == cur_token` 인 entry 를 drain.
//! `cur_token` 은 poll 반환 후 +1 된다.
//!
//! ### off-by-one 분석
//!
//! `DecodeLoop::run(budget, first_token)` 에서 `first_token` 은 prefill 직후 이미
//! 생성된 토큰이고 `run` 은 decode 스텝 0 부터 시작한다. poll 이 발화하는 시점은
//! decode step 0, 1, 2, … 순서이므로 `cur_token = 0` 이 첫 번째 decode token(=
//! prompt_len 번째 위치)에 대응한다. legacy `generate.rs` 도 `decode_step` 카운터를
//! 0 에서 시작해 `directives_at(decode_step)` 를 호출했으므로 **off-by-one 없음** —
//! schedule 파일의 `at_token` 은 decode step 인덱스(0-based)를 의미한다.

use argus_shared::EngineCommand;

use crate::experiment::ExperimentSchedule;
use crate::session::command_dispatcher::CommandSource;

/// 정적 JSON schedule 을 decode-step token-pos 기반으로 emit 하는 `CommandSource`.
///
/// `ExperimentSchedule` 을 보유하고 내부 step counter 로 token-pos 를 추적한다.
/// 매 `poll()` 호출 시 해당 pos 의 directive commands 를 drain 하고 counter 를 +1.
pub struct ScheduleCommandSource {
    schedule: ExperimentSchedule,
    /// 다음 poll 에서 조회할 decode step 위치 (0-based).
    cur_token: usize,
}

impl ScheduleCommandSource {
    /// 새 `ScheduleCommandSource` 를 만든다. `cur_token` 은 0 에서 시작.
    pub fn new(schedule: ExperimentSchedule) -> Self {
        Self {
            schedule,
            cur_token: 0,
        }
    }
}

impl CommandSource for ScheduleCommandSource {
    /// 현재 `cur_token` 위치에 등록된 directive 의 commands 를 모두 drain 하고
    /// counter 를 +1 한다.
    ///
    /// - 동일 pos 복수 entry: 정의 순서대로 flatten 하여 반환한다.
    /// - 미도달 pos: 빈 `Vec` 반환 (counter 는 여전히 +1).
    /// - schedule 소진(모든 entry 의 at_token < cur_token) 후: 빈 `Vec`.
    fn poll(&mut self) -> anyhow::Result<Vec<EngineCommand>> {
        let pos = self.cur_token;
        self.cur_token += 1;

        // 변환 로직 0: EngineDirective.commands 를 그대로 flatten.
        let cmds: Vec<EngineCommand> = self
            .schedule
            .directives_at(pos)
            .flat_map(|e| e.directive.commands.iter().cloned())
            .collect();
        Ok(cmds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_shared::{EngineCommand, EngineDirective};

    use crate::experiment::{DirectiveEntry, ExperimentSchedule};

    fn make_schedule(entries: Vec<(usize, Vec<EngineCommand>)>) -> ExperimentSchedule {
        let directives = entries
            .into_iter()
            .map(|(at_token, commands)| DirectiveEntry {
                at_token,
                directive: EngineDirective {
                    seq_id: 0,
                    commands,
                },
            })
            .collect();
        ExperimentSchedule {
            name: "test".to_string(),
            description: String::new(),
            directives,
        }
    }

    // (a) pos 도달 시 commands 반환
    #[test]
    fn poll_returns_commands_at_target_pos() {
        let schedule = make_schedule(vec![(3, vec![EngineCommand::Suspend])]);
        let mut src = ScheduleCommandSource::new(schedule);

        // pos 0, 1, 2: 빈 Vec
        for _ in 0..3 {
            let cmds = src.poll().unwrap();
            assert!(cmds.is_empty());
        }
        // pos 3: Suspend 반환
        let cmds = src.poll().unwrap();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Suspend));
    }

    // (b) 미도달 pos: 빈 Vec
    #[test]
    fn poll_returns_empty_before_scheduled_pos() {
        let schedule = make_schedule(vec![(10, vec![EngineCommand::Suspend])]);
        let mut src = ScheduleCommandSource::new(schedule);

        let cmds = src.poll().unwrap();
        assert!(cmds.is_empty(), "pos 0 → empty (entry is at 10)");
    }

    // (c) 동일 pos 복수 directive 순서 보존
    #[test]
    fn poll_same_pos_multiple_directives_order_preserved() {
        let schedule = make_schedule(vec![
            (2, vec![EngineCommand::Throttle { delay_ms: 10 }]),
            (2, vec![EngineCommand::Suspend]),
        ]);
        let mut src = ScheduleCommandSource::new(schedule);

        // pos 0, 1: 빈 Vec
        src.poll().unwrap();
        src.poll().unwrap();

        // pos 2: Throttle 먼저, Suspend 다음 (정의 순서 보존)
        let cmds = src.poll().unwrap();
        assert_eq!(cmds.len(), 2);
        assert!(matches!(cmds[0], EngineCommand::Throttle { delay_ms: 10 }));
        assert!(matches!(cmds[1], EngineCommand::Suspend));
    }

    // (d) 소진 후 빈 Vec
    #[test]
    fn poll_returns_empty_after_exhausted() {
        let schedule = make_schedule(vec![(0, vec![EngineCommand::Suspend])]);
        let mut src = ScheduleCommandSource::new(schedule);

        // pos 0: Suspend
        let cmds = src.poll().unwrap();
        assert_eq!(cmds.len(), 1);

        // pos 1, 2, ...: 빈 Vec
        for _ in 0..5 {
            let cmds = src.poll().unwrap();
            assert!(cmds.is_empty());
        }
    }

    // (e) ExperimentSchedule 파싱 왕복 (SSOT: experiment.rs 기존 포맷)
    #[test]
    fn experiment_schedule_parse_roundtrip() {
        let json = r#"{
            "name": "throttle_test",
            "description": "test schedule",
            "directives": [
                {
                    "at_token": 10,
                    "directive": {
                        "seq_id": 1,
                        "commands": [{"type": "throttle", "delay_ms": 100}]
                    }
                },
                {
                    "at_token": 20,
                    "directive": {
                        "seq_id": 2,
                        "commands": [{"type": "suspend"}]
                    }
                }
            ]
        }"#;

        let schedule: ExperimentSchedule = serde_json::from_str(json).expect("parse");
        assert_eq!(schedule.name, "throttle_test");
        assert_eq!(schedule.directives.len(), 2);

        let mut src = ScheduleCommandSource::new(schedule);

        // pos 0..9: empty
        for _ in 0..10 {
            assert!(src.poll().unwrap().is_empty());
        }
        // pos 10: Throttle
        let cmds = src.poll().unwrap();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Throttle { delay_ms: 100 }));

        // pos 11..19: empty
        for _ in 0..9 {
            assert!(src.poll().unwrap().is_empty());
        }
        // pos 20: Suspend
        let cmds = src.poll().unwrap();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Suspend));
    }

    // pos 역순 정의: 높은 pos 먼저 정의해도 각 pos 에서 정상 반환
    #[test]
    fn poll_out_of_order_definition() {
        let schedule = make_schedule(vec![
            (5, vec![EngineCommand::Suspend]),
            (1, vec![EngineCommand::Resume]),
        ]);
        let mut src = ScheduleCommandSource::new(schedule);

        // pos 0: empty
        assert!(src.poll().unwrap().is_empty());
        // pos 1: Resume
        let cmds = src.poll().unwrap();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Resume));
        // pos 2, 3, 4: empty
        for _ in 0..3 {
            assert!(src.poll().unwrap().is_empty());
        }
        // pos 5: Suspend
        let cmds = src.poll().unwrap();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Suspend));
    }
}
