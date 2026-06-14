use super::{LayerWeightNames, WeightMapper};

pub struct LlamaMapper {
    pub prefix: String,
}

impl WeightMapper for LlamaMapper {
    fn weight_names(&self, i: usize) -> LayerWeightNames {
        let p = &self.prefix;
        LayerWeightNames {
            wq: format!("{p}model.layers.{i}.self_attn.q_proj.weight"),
            wk: format!("{p}model.layers.{i}.self_attn.k_proj.weight"),
            wv: format!("{p}model.layers.{i}.self_attn.v_proj.weight"),
            wo: format!("{p}model.layers.{i}.self_attn.o_proj.weight"),
            w_gate: format!("{p}model.layers.{i}.mlp.gate_proj.weight"),
            w_up: format!("{p}model.layers.{i}.mlp.up_proj.weight"),
            w_down: format!("{p}model.layers.{i}.mlp.down_proj.weight"),
            attention_norm: format!("{p}model.layers.{i}.input_layernorm.weight"),
            ffn_norm: format!("{p}model.layers.{i}.post_attention_layernorm.weight"),
            pre_ffn_norm: None,
            post_ffn_norm: None,
            q_norm: None,
            k_norm: None,
        }
    }
    fn embed_name(&self) -> String {
        format!("{}model.embed_tokens.weight", self.prefix)
    }
    fn norm_name(&self) -> String {
        format!("{}model.norm.weight", self.prefix)
    }
    fn lm_head_name(&self) -> String {
        format!("{}lm_head.weight", self.prefix)
    }
}
