//! eat-core — language rubrics, word model, and the Ollama-backed NLP engine.
//! Rust port of language_prompts.py + nlp_engine.py from Project-EAT.

pub mod languages;
pub mod logging;
pub mod nlp;
pub mod words;

pub use nlp::{EatError, NlpEngine};
pub use words::{Pages, Word};
