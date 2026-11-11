fn main() {
    // 1) 本地 Ollama
    let (ok, models, tag) = eat_core::NlpEngine::check_endpoint("http://localhost:11434/api/generate", None);
    println!("local ollama: ok={} via={} models={:?}", ok, tag, &models[..models.len().min(3)]);
    // 2) 無效 OpenAI key
    let (ok, _, msg) = eat_core::NlpEngine::check_endpoint("https://api.openai.com/v1/chat/completions", Some("sk-invalid-xxx"));
    println!("openai bad key: ok={} msg={}", ok, msg);
    // 3) 無 key
    let (ok, _, msg) = eat_core::NlpEngine::check_endpoint("https://api.openai.com/v1/chat/completions", None);
    println!("openai no key: ok={} msg={}", ok, msg);
}
