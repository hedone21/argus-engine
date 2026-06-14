use crate::model_config::ModelArch;
use anyhow::{Result, anyhow};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Copy)]
pub struct ChatTemplate {
    arch: ModelArch,
}

impl ChatTemplate {
    pub fn new(arch: ModelArch) -> Result<Self> {
        match arch {
            ModelArch::Llama | ModelArch::Qwen2 => Ok(Self { arch }),
            ModelArch::Gemma3 => Err(anyhow!(
                "chat template for Gemma3 is not implemented (use Llama or Qwen2)"
            )),
        }
    }

    pub fn arch(&self) -> ModelArch {
        self.arch
    }

    pub fn render_system(&self, content: &str) -> String {
        match self.arch {
            ModelArch::Llama => {
                format!("<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
            }
            ModelArch::Qwen2 => format!("<|im_start|>system\n{content}<|im_end|>\n"),
            ModelArch::Gemma3 => unreachable!(),
        }
    }

    pub fn render_user_and_assistant_header(&self, user: &str) -> String {
        match self.arch {
            ModelArch::Llama => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            ModelArch::Qwen2 => {
                format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
            }
            ModelArch::Gemma3 => unreachable!(),
        }
    }

    pub fn assistant_eot(&self) -> &'static str {
        match self.arch {
            ModelArch::Llama => "<|eot_id|>",
            ModelArch::Qwen2 => "<|im_end|>",
            ModelArch::Gemma3 => unreachable!(),
        }
    }

    pub fn bos_needed_on_first_prefill(&self) -> bool {
        matches!(self.arch, ModelArch::Llama)
    }

    pub fn bos_literal(&self) -> Option<&'static str> {
        match self.arch {
            ModelArch::Llama => Some("<|begin_of_text|>"),
            ModelArch::Qwen2 => None,
            ModelArch::Gemma3 => None,
        }
    }

    pub fn stop_token_literals(&self) -> &'static [&'static str] {
        match self.arch {
            ModelArch::Llama => &["<|eot_id|>", "<|end_of_text|>"],
            ModelArch::Qwen2 => &["<|im_end|>", "<|endoftext|>"],
            ModelArch::Gemma3 => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_system_golden() {
        let t = ChatTemplate::new(ModelArch::Llama).unwrap();
        assert_eq!(
            t.render_system("You are helpful."),
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"
        );
    }

    #[test]
    fn llama_user_golden() {
        let t = ChatTemplate::new(ModelArch::Llama).unwrap();
        assert_eq!(
            t.render_user_and_assistant_header("Hi"),
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn qwen2_system_golden() {
        let t = ChatTemplate::new(ModelArch::Qwen2).unwrap();
        assert_eq!(
            t.render_system("You are helpful."),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        );
    }

    #[test]
    fn qwen2_user_golden() {
        let t = ChatTemplate::new(ModelArch::Qwen2).unwrap();
        assert_eq!(
            t.render_user_and_assistant_header("Hi"),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn gemma3_returns_error() {
        assert!(ChatTemplate::new(ModelArch::Gemma3).is_err());
    }

    #[test]
    fn bos_policy() {
        assert!(
            ChatTemplate::new(ModelArch::Llama)
                .unwrap()
                .bos_needed_on_first_prefill()
        );
        assert!(
            !ChatTemplate::new(ModelArch::Qwen2)
                .unwrap()
                .bos_needed_on_first_prefill()
        );
    }

    #[test]
    fn stop_tokens() {
        assert_eq!(
            ChatTemplate::new(ModelArch::Llama)
                .unwrap()
                .stop_token_literals()
                .len(),
            2
        );
        assert_eq!(
            ChatTemplate::new(ModelArch::Qwen2)
                .unwrap()
                .stop_token_literals()
                .len(),
            2
        );
    }
}
