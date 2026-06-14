use super::{LayerBiasNames, LayerWeightNames, WeightMapper};

pub struct Qwen2Mapper {
    pub prefix: String,
}

impl WeightMapper for Qwen2Mapper {
    fn weight_names(&self, i: usize) -> LayerWeightNames {
        let p = &self.prefix;
        // Qwen2 uses the same weight names as Llama
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

    fn bias_names(&self, i: usize) -> Option<LayerBiasNames> {
        let p = &self.prefix;
        Some(LayerBiasNames {
            bq: format!("{p}model.layers.{i}.self_attn.q_proj.bias"),
            bk: format!("{p}model.layers.{i}.self_attn.k_proj.bias"),
            bv: format!("{p}model.layers.{i}.self_attn.v_proj.bias"),
        })
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
