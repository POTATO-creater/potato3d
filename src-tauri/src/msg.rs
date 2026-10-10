//! Mensagens de progresso traduzíveis. A interface procura `key` no catálogo
//! do idioma escolhido (`src/i18n/`) e preenche o texto com `args`; um
//! argumento pode ser outra `Msg`.

use serde::Serialize;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize)]
pub struct Msg {
    pub key: &'static str,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub args: Map<String, Value>,
}

/// `msg!("poses.matching", pairs = n)`: a chave tem que existir em todos os
/// catálogos de `src/i18n/`.
#[macro_export]
macro_rules! msg {
    ($key:literal $(, $name:ident = $val:expr)* $(,)?) => {{
        #[allow(unused_mut)]
        let mut args = ::serde_json::Map::new();
        $( args.insert(stringify!($name).to_string(), ::serde_json::to_value(&$val).unwrap()); )*
        $crate::msg::Msg { key: $key, args }
    }};
}

/// Forma crua para a CLI e os logs: `poses.matching {"pairs":120}`.
impl std::fmt::Display for Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.args.is_empty() {
            f.write_str(self.key)
        } else {
            write!(f, "{} {}", self.key, Value::Object(self.args.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    /// Formato lido por `formatMsg` em src/i18n.
    #[test]
    fn serializes_key_and_nested_args() {
        let m = crate::msg!("poses.retry", step = crate::msg!("poses.matching", pairs = 12));
        assert_eq!(
            serde_json::to_value(&m).unwrap(),
            serde_json::json!({ "key": "poses.retry", "args": { "step": { "key": "poses.matching", "args": { "pairs": 12 } } } })
        );
        assert_eq!(serde_json::to_string(&crate::msg!("train.done")).unwrap(), r#"{"key":"train.done"}"#);
    }
}
