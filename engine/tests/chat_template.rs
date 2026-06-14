//! Integration tests for the chat template module — byte-exact goldens.

use argus_engine::model_config::ModelArch;
use argus_engine::session::chat_template::ChatTemplate;

#[test]
fn llama_system_golden() {
    let t = ChatTemplate::new(ModelArch::Llama).unwrap();
    assert_eq!(
        t.render_system("You are a helpful assistant."),
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
    );
}

#[test]
fn llama_user_and_assistant_header_golden() {
    let t = ChatTemplate::new(ModelArch::Llama).unwrap();
    assert_eq!(
        t.render_user_and_assistant_header("What is Rust?"),
        "<|start_header_id|>user<|end_header_id|>\n\nWhat is Rust?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );
}

#[test]
fn llama_multi_turn_concat_golden() {
    let t = ChatTemplate::new(ModelArch::Llama).unwrap();
    let s1 = t.render_system("Be concise.");
    let u1 = t.render_user_and_assistant_header("hi");
    let eot1 = t.assistant_eot();
    let u2 = t.render_user_and_assistant_header("again");
    let combined = format!("{s1}{u1}ok{eot1}{u2}");
    assert_eq!(
        combined,
        "<|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\nok<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\nagain<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    );
}

#[test]
fn qwen2_system_golden() {
    let t = ChatTemplate::new(ModelArch::Qwen2).unwrap();
    assert_eq!(
        t.render_system("You are a helpful assistant."),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    );
}

#[test]
fn qwen2_user_and_assistant_header_golden() {
    let t = ChatTemplate::new(ModelArch::Qwen2).unwrap();
    assert_eq!(
        t.render_user_and_assistant_header("What is Rust?"),
        "<|im_start|>user\nWhat is Rust?<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn qwen2_multi_turn_concat_golden() {
    let t = ChatTemplate::new(ModelArch::Qwen2).unwrap();
    let s1 = t.render_system("Be concise.");
    let u1 = t.render_user_and_assistant_header("hi");
    let eot1 = t.assistant_eot();
    let u2 = t.render_user_and_assistant_header("again");
    let combined = format!("{s1}{u1}ok{eot1}\n{u2}");
    assert_eq!(
        combined,
        "<|im_start|>system\nBe concise.<|im_end|>\n\
         <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n\
         ok<|im_end|>\n\
         <|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn gemma3_is_rejected() {
    let err = ChatTemplate::new(ModelArch::Gemma3).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("Gemma3"),
        "error should mention Gemma3, got: {msg}"
    );
}

#[test]
fn assistant_eot_strings() {
    assert_eq!(
        ChatTemplate::new(ModelArch::Llama).unwrap().assistant_eot(),
        "<|eot_id|>"
    );
    assert_eq!(
        ChatTemplate::new(ModelArch::Qwen2).unwrap().assistant_eot(),
        "<|im_end|>"
    );
}

#[test]
fn bos_policy() {
    assert!(
        ChatTemplate::new(ModelArch::Llama)
            .unwrap()
            .bos_needed_on_first_prefill()
    );
    assert_eq!(
        ChatTemplate::new(ModelArch::Llama).unwrap().bos_literal(),
        Some("<|begin_of_text|>")
    );
    assert!(
        !ChatTemplate::new(ModelArch::Qwen2)
            .unwrap()
            .bos_needed_on_first_prefill()
    );
    assert_eq!(
        ChatTemplate::new(ModelArch::Qwen2).unwrap().bos_literal(),
        None
    );
}

#[test]
fn stop_tokens_count_and_content() {
    let llama = ChatTemplate::new(ModelArch::Llama).unwrap();
    assert_eq!(
        llama.stop_token_literals(),
        &["<|eot_id|>", "<|end_of_text|>"]
    );
    let qwen = ChatTemplate::new(ModelArch::Qwen2).unwrap();
    assert_eq!(qwen.stop_token_literals(), &["<|im_end|>", "<|endoftext|>"]);
}
