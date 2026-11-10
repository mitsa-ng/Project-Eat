//! eat-gui — teacher-facing desktop app (Rust port of gui.py's batch flow).
//!
//! Folder of essay PDFs → every file corrected (OCR + Ollama + annotated PDF)
//! → all outputs packed into one ZIP. Includes a first-run check that
//! verifies Ollama is reachable and offers to pull the default model.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;

use eat_app::PipelineOpts;
use eat_core::languages::LANGUAGES;
use eat_core::nlp::{default_language, default_model_name, default_ollama_url};

// ── shared state between the GUI thread and worker threads ──────────────────

#[derive(Default)]
struct BatchState {
    running: bool,
    log: Vec<String>,
    results: Vec<(String, usize, String)>,
    zip_path: Option<String>,
}

static GUI_WANTS_LOG: AtomicBool = AtomicBool::new(false);

fn install_log_sink(state: Arc<Mutex<BatchState>>) {
    eat_core::logging::subscribe(Some(Box::new(move |line: &str| {
        // only mirror into the GUI buffer while the GUI wants logs
        if GUI_WANTS_LOG.load(Ordering::Relaxed) {
            if let Ok(mut s) = state.lock() {
                s.log.push(line.to_string());
                if s.log.len() > 500 {
                    s.log.drain(..100);
                }
            }
        }
    })));
}

enum OllamaStatus {
    Checking,
    Reachable(Vec<String>), // model tags
    Unreachable,
}

/// First-run Ollama auto-install pipeline (macOS).
#[derive(Clone, Debug)]
enum InstallState {
    Idle,
    Downloading { done: u64, total: u64 },
    Installing,
    Failed(String),
}

struct EatGui {
    input_folder: String,
    zip_out: String,
    model: String,
    ollama_url: String,
    language: String,
    save_intermediate: bool,
    reader: String, // "ppocr" | "vlm"
    vlm_model: String,
    state: Arc<Mutex<BatchState>>,
    ollama: Arc<Mutex<Option<OllamaStatus>>>,
    pull_running: Arc<AtomicBool>,
    install: Arc<Mutex<InstallState>>,
    live: LiveState,
}

/// 即時模式（live camera）狀態：worker 執行緒 ⇄ UI 的通道。
struct LiveState {
    running: Arc<AtomicBool>,
    preview_rx: std::sync::mpsc::Receiver<LivePreview>,
    result_rx: std::sync::mpsc::Receiver<String>,
    texture: Option<egui::TextureHandle>,
    last_words: usize,
}

struct LivePreview {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

impl LiveState {
    fn new() -> Self {
        let (_ptx, preview_rx) = std::sync::mpsc::channel();
        let (_rtx, result_rx) = std::sync::mpsc::channel();
        LiveState { running: Arc::new(AtomicBool::new(false)), preview_rx, result_rx, texture: None, last_words: 0 }
    }
}

impl EatGui {
    fn new(cc: &eframe::CreationContext) -> Self {
        install_cjk_font(&cc.egui_ctx);
        let state = Arc::new(Mutex::new(BatchState::default()));
        install_log_sink(state.clone());
        let gui = EatGui {
            input_folder: String::new(),
            zip_out: String::new(),
            model: default_model_name(),
            ollama_url: default_ollama_url(),
            language: default_language(),
            save_intermediate: false,
            reader: "vlm".into(),
            vlm_model: "qwen2.5vl:7b".into(),
            state,
            ollama: Arc::new(Mutex::new(Some(OllamaStatus::Checking))),
            pull_running: Arc::new(AtomicBool::new(false)),
            install: Arc::new(Mutex::new(InstallState::Idle)),
            live: LiveState::new(),
        };
        gui.spawn_ollama_check();
        gui
    }
}

/// egui 預設只附 Ubuntu-Light（無 CJK 字形），介面中文會變成方框。
/// 這裡把隨附的 Noto Sans CJK 註冊進 egui 字型表（常規＋等寬家族）。
fn install_cjk_font(ctx: &egui::Context) {
    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Ok(p) = std::env::var("EAT_CJK_FONT") {
            v.push(p.into());
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                v.push(dir.join("fonts").join(CJK_FONT_FILE));
                v.push(dir.join(CJK_FONT_FILE));
            }
        }
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            v.push(std::path::Path::new(&manifest)
                .join("../../vendor/fonts")
                .join(CJK_FONT_FILE));
        }
        v
    };
    let path = match candidates.iter().find(|p| p.exists()) {
        Some(p) => p.clone(),
        None => {
            eprintln!("eat-gui: CJK font not found — UI Chinese text will render as boxes");
            return;
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("eat-gui: cannot read {}: {}", path.display(), e);
            return;
        }
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "noto_cjk".into(),
        std::sync::Arc::new(egui::FontData::from_owned(bytes)),
    );
    // prepend so CJK takes priority over the default Latin face for matching glyphs
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(list) = fonts.families.get_mut(&family) {
            list.insert(0, "noto_cjk".into());
        }
    }
    ctx.set_fonts(fonts);
}

const CJK_FONT_FILE: &str = "NotoSansCJKtc-Regular.otf";

impl EatGui {
    fn spawn_ollama_check(&self) {
        let url = self.ollama_url.clone();
        let ollama = self.ollama.clone();
        std::thread::spawn(move || {
            let tags_url = url
                .replace("/api/generate", "/api/tags")
                .trim_end_matches('/')
                .to_string();
            let tags_url = if tags_url.contains("/api/tags") {
                tags_url
            } else {
                format!("{}/api/tags", tags_url)
            };
            let status = match ureq_local::get(&tags_url) {
                Ok(body) => {
                    let models: Vec<String> = serde_json_parse_models(&body);
                    OllamaStatus::Reachable(models)
                }
                Err(_) => OllamaStatus::Unreachable,
            };
            *ollama.lock().unwrap() = Some(status);
        });
    }

    fn spawn_pull(&self) {
        if self.pull_running.load(Ordering::Relaxed) {
            return;
        }
        self.pull_running.store(true, Ordering::Relaxed);
        eat_core::logging::info("啟動模型下載：ollama pull qwen2.5:7b（背景執行）");
        let pull = self.pull_running.clone();
        std::thread::spawn(move || {
            use std::process::{Command, Stdio};
            let ok = Command::new("ollama")
                .args(["pull", "qwen2.5:7b"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                eat_core::logging::info("模型下載完成（qwen2.5:7b）");
            } else {
                eat_core::logging::error("模型下載失敗 — 請確認已安裝 Ollama，並手動執行：ollama pull qwen2.5:7b");
            }
            pull.store(false, Ordering::Relaxed);
        });
    }

    /// One-click Ollama install (macOS): download the official zip, extract
    /// Ollama.app, copy it to /Applications (falling back to ~/Applications),
    /// launch it, then poll the API until the server answers.
    fn spawn_install(&self) {
        {
            let mut st = self.install.lock().unwrap();
            if matches!(*st, InstallState::Downloading { .. } | InstallState::Installing) {
                return;
            }
            *st = InstallState::Downloading { done: 0, total: 0 };
        }
        eat_core::logging::info("開始下載 Ollama 安裝包（Ollama-darwin.zip，約 229 MB）…");
        let install = self.install.clone();
        let ollama = self.ollama.clone();
        let url = self.ollama_url.clone();
        std::thread::spawn(move || {
            let set = |st: InstallState| *install.lock().unwrap() = st;
            let zip = std::env::temp_dir().join("Ollama-darwin.zip");

            // total size via HEAD (follows redirects to the CDN)
            let total: u64 = std::process::Command::new("curl")
                .args(["-sIL", "--max-time", "15", "https://ollama.com/download/Ollama-darwin.zip"])
                .output()
                .ok()
                .and_then(|o| {
                    let head = String::from_utf8_lossy(&o.stdout).into_owned();
                    head.lines()
                        .rev()
                        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                        .and_then(|l| l.split(':').nth(1).and_then(|v| v.trim().parse().ok()))
                })
                .unwrap_or(229_245_253);

            // background download; poll the file size for progress
            let mut dl = std::process::Command::new("curl")
                .args(["-sL", "--max-time", "1800", "-o"])
                .arg(&zip)
                .arg("https://ollama.com/download/Ollama-darwin.zip")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("curl");
            loop {
                if matches!(install.lock().unwrap().clone(), InstallState::Failed(_)) {
                    let _ = dl.kill();
                    return;
                }
                match dl.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => {}
                    Err(_) => break,
                }
                let done = std::fs::metadata(&zip).map(|m| m.len()).unwrap_or(0);
                *install.lock().unwrap() = InstallState::Downloading { done, total };
                std::thread::sleep(std::time::Duration::from_millis(600));
            }
            let done = std::fs::metadata(&zip).map(|m| m.len()).unwrap_or(0);
            if done < 1_000_000 {
                set(InstallState::Failed(format!("下載不完整（{} bytes）", done)));
                eat_core::logging::error("Ollama 下載失敗");
                return;
            }
            set(InstallState::Installing);
            eat_core::logging::info("下載完成，解壓並安裝 Ollama.app …");

            // extract (ditto handles zip + resource forks correctly)
            let work = std::env::temp_dir().join("OllamaInstall");
            let _ = std::fs::remove_dir_all(&work);
            let _ = std::fs::create_dir_all(&work);
            let ok = std::process::Command::new("ditto")
                .args(["-x", "-k"])
                .arg(&zip)
                .arg(&work)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            let app = work.join("Ollama.app");
            if !ok || !app.exists() {
                set(InstallState::Failed("解壓失敗".into()));
                eat_core::logging::error("Ollama 安裝包解壓失敗");
                return;
            }

            // install: /Applications first, ~/Applications as user fallback
            let mut dest = std::path::PathBuf::from("/Applications/Ollama.app");
            let cp = |dst: &std::path::Path| {
                std::process::Command::new("cp")
                    .args(["-R"])
                    .arg(&app)
                    .arg(dst)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
            };
            if !cp(&dest) {
                let home_apps = std::env::var("HOME")
                    .map(|h| std::path::PathBuf::from(h).join("Applications/Ollama.app"))
                    .unwrap_or_else(|_| dest.clone());
                if !cp(&home_apps) {
                    set(InstallState::Failed("無法複製到 /Applications 或 ~/Applications".into()));
                    eat_core::logging::error("Ollama 安裝失敗：無法複製 App（請手動安裝）");
                    return;
                }
                dest = home_apps;
            }
            eat_core::logging::info(&format!("已安裝 → {}", dest.display()));

            // launch the menu-bar app (starts `ollama serve` automatically)
            if !std::process::Command::new("open")
                .arg(&dest)
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
            {
                set(InstallState::Failed("啟動 Ollama 失敗".into()));
                return;
            }

            // poll the API until the server answers (menu-bar app startup)
            let tags_url = url.replace("/api/generate", "/api/tags");
            for _ in 0..40 {
                std::thread::sleep(std::time::Duration::from_secs(3));
                if let Ok(body) = ureq_local::get(&tags_url) {
                    let models = serde_json_parse_models(&body);
                    *ollama.lock().unwrap() = Some(OllamaStatus::Reachable(models));
                    set(InstallState::Idle);
                    eat_core::logging::info("Ollama 已安裝並啟動 ✔ 接下來請下載模型 qwen2.5:7b");
                    return;
                }
            }
            set(InstallState::Failed("已安裝但伺服器未在 2 分鐘內回應".into()));
        });
    }

    /// 即時模式 worker：開相機 → 預覽串流 → 節流 OCR → 文字有變才跑 LLM
    /// （移植 live_camera/live_mode.py：conf ≥ 0.35、<3 字跳過、文字未變
    /// 跳過昂貴的 NLP 呼叫）。
    fn spawn_live(&mut self, camera_index: usize) {
        if self.live.running.load(Ordering::Relaxed) {
            return;
        }
        self.live.running.store(true, Ordering::Relaxed);
        eat_core::logging::info(&format!("即時模式啟動（相機 {}）…", camera_index));

        let running = self.live.running.clone();
        let (ptx, preview_rx) = std::sync::mpsc::channel();
        let (rtx, result_rx) = std::sync::mpsc::channel();
        self.live.preview_rx = preview_rx;
        self.live.result_rx = result_rx;
        let language = self.language.clone();
        let model = self.model.clone();
        let url = self.ollama_url.clone();

        std::thread::spawn(move || {
            use nokhwa::Camera;
            use nokhwa::pixel_format::RgbFormat;
            use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};

            let mut camera = match Camera::new(
                CameraIndex::Index(camera_index as u32),
                RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(RequestedFormatType::None),
            ) {
                Ok(c) => c,
                Err(e) => {
                    eat_core::logging::error(&format!("無法開啟相機：{:?}", e));
                    running.store(false, Ordering::Relaxed);
                    return;
                }
            };
            if let Err(e) = camera.open_stream() {
                eat_core::logging::error(&format!("相機串流開啟失敗：{:?}", e));
                running.store(false, Ordering::Relaxed);
                return;
            }

            let ocr_lock = match eat_app::shared_ocr() {
                Ok(o) => o,
                Err(e) => {
                    eat_core::logging::error(&e.0);
                    running.store(false, Ordering::Relaxed);
                    return;
                }
            };
            let mut ocr = ocr_lock.lock().unwrap();
            let nlp = eat_core::NlpEngine::new(&model, &url, &language);

            let mut last_text = String::new();
            let mut last_ocr = std::time::Instant::now();
            while running.load(Ordering::Relaxed) {
                let frame = match camera.frame() {
                    Ok(f) => f,
                    Err(_) => break,
                };
                let img = match frame.decode_image::<nokhwa::pixel_format::RgbFormat>() {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                let (w, h) = (img.width() as usize, img.height() as usize);
                let rgb = img.into_raw();
                let _ = ptx.send(LivePreview { w, h, rgb: rgb.clone() });

                // 節流 OCR（~每 2.5 秒；Python 版每 3 幀＋文字比對）
                if last_ocr.elapsed() < std::time::Duration::from_millis(2500) {
                    continue;
                }
                last_ocr = std::time::Instant::now();
                let eat_img = eat_ocr::imgops::Img::from_rgb(w, h, &rgb);
                let lines = match ocr.run(&eat_img) {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let lines: Vec<_> = lines
                    .into_iter()
                    .filter(|l| l.score >= 0.35 && !l.text.trim().is_empty())
                    .collect();
                if lines.len() < 3 {
                    continue; // 文件不在畫面中
                }
                let frame_text = lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" ");
                if frame_text == last_text {
                    continue; // 同一頁 — 跳過 LLM
                }
                last_text = frame_text.clone();
                eat_core::logging::info(&format!(
                    "即時畫面：{} 行文字，分析中 …",
                    lines.len()
                ));
                let pages: eat_core::words::Pages = vec![lines
                    .iter()
                    .map(|l| eat_core::words::Word {
                        text: l.text.clone(),
                        box_: [
                            [l.box_[0].0, l.box_[0].1],
                            [l.box_[1].0, l.box_[1].1],
                            [l.box_[2].0, l.box_[2].1],
                            [l.box_[3].0, l.box_[3].1],
                        ],
                    })
                    .collect()];
                match nlp.analyse_pages(&pages) {
                    Ok(errors) => {
                        let msg = if errors.is_empty() {
                            "✔ 未發現錯誤".to_string()
                        } else {
                            errors
                                .iter()
                                .map(|e| format!("[{}] {} → {}", e.kind, e.original, e.suggestion))
                                .collect::<Vec<_>>()
                                .join("\n  ")
                        };
                        let _ = rtx.send(format!("即時分析（{} 個錯誤）：\n  {}", errors.len(), msg));
                    }
                    Err(e) => eat_core::logging::error(&e.to_string()),
                }
            }
            let _ = camera.stop_stream();
            eat_core::logging::info("即時模式結束");
        });
    }

    fn spawn_batch(&self) {
        let opts = PipelineOpts {
            input_pdf: None,
            output: None,
            reader: self.reader.clone(),
            vlm_model: self.vlm_model.clone(),
            model: Some(self.model.clone()),
            ollama_url: Some(self.ollama_url.clone()),
            language: Some(self.language.clone()),
            save_intermediate: true, // error counts for the results table
            no_spell_correct: true,
            no_gpu: false,
        };
        let dir = self.input_folder.clone();
        let zip = self.zip_out.clone();
        let state = self.state.clone();
        {
            let mut s = state.lock().unwrap();
            s.running = true;
            s.results.clear();
            s.log.clear();
        }
        std::thread::spawn(move || {
            GUI_WANTS_LOG.store(true, Ordering::Relaxed);
            let (_zip, results) = eat_app::run_batch(&dir, &zip, &opts);
            GUI_WANTS_LOG.store(false, Ordering::Relaxed);
            let mut s = state.lock().unwrap();
            s.running = false;
            s.results = results;
            s.zip_path = Some(zip);
        });
    }
}

// tiny local HTTP helpers (avoid a workspace-wide ureq dep in the gui crate)
mod ureq_local {
    pub fn get(url: &str) -> Result<String, String> {
        let resp = std::thread::spawn({
            let url = url.to_string();
            move || {
                // blocking read with a 3 s budget via std only is not possible;
                // shell out to curl which is present on macOS out of the box
                std::process::Command::new("curl")
                    .args(["-s", "--max-time", "3", &url])
                    .output()
            }
        })
        .join()
        .map_err(|_| "thread panic".to_string())?
        .map_err(|e| format!("curl failed: {}", e))?;
        if resp.status.success() {
            Ok(String::from_utf8_lossy(&resp.stdout).into_owned())
        } else {
            Err(format!("curl exit {}", resp.status))
        }
    }
}

fn serde_json_parse_models(body: &str) -> Vec<String> {
    // {"models":[{"name":"qwen2.5:7b:latest", ...}, ...]}
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(p) = rest.find("\"name\"") {
        let after = &rest[p + 6..];
        if let Some(q1) = after.find('"') {
            if let Some(q2) = after[q1 + 1..].find('"') {
                out.push(after[q1 + 1..q1 + 1 + q2].to_string());
            }
        }
        rest = &after[6.min(after.len())..];
        if out.len() > 50 {
            break;
        }
    }
    out
}

impl eframe::App for EatGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.heading("AI 作文批改 AI Essay Corrector");
            ui.label("選一個資料夾（內含學生作文 PDF）→ 批次批改 → 打包成 ZIP");
            ui.add_space(4.0);
        });

        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let status = self.ollama.lock().unwrap().take();
                match &status {
                    Some(OllamaStatus::Reachable(models)) => {
                        ui.colored_label(egui::Color32::from_rgb(0x2e, 0x7d, 0x32), "● Ollama 已連線");
                        if models.is_empty() {
                            ui.colored_label(egui::Color32::from_rgb(0xb2, 0x6a, 0x00), "— 尚未下載模型");
                            if ui.button("下載預設模型 qwen2.5:7b（4.7 GB）").clicked() {
                                self.spawn_pull();
                            }
                        } else if !models.iter().any(|m| m.starts_with("qwen2.5:7b")) {
                            ui.label(format!("已安裝模型：{}", models.join(", ")));
                        }
                    }
                    Some(OllamaStatus::Unreachable) => {
                        ui.colored_label(egui::Color32::from_rgb(0xc6, 0x28, 0x28), "● 找不到 Ollama");
                        let inst = self.install.lock().unwrap().clone();
                        match inst {
                            InstallState::Downloading { done, total } => {
                                let pct = if total > 0 { (done as f32 / total as f32 * 100.0).min(100.0) } else { 0.0 };
                                ui.spinner();
                                ui.label(format!("下載 Ollama 中 … {:.0}%（{:.0}/{} MB）", pct, done as f64 / 1e6, total / 1_000_000));
                            }
                            InstallState::Installing => {
                                ui.spinner();
                                ui.label("安裝中 …");
                            }
                            InstallState::Failed(e) => {
                                ui.colored_label(egui::Color32::from_rgb(0xc6, 0x28, 0x28), format!("安裝失敗：{}", e));
                                if ui.button("重試安裝").clicked() {
                                    self.spawn_install();
                                }
                            }
                            InstallState::Idle => {
                                if ui.button("⤓ 一鍵安裝 Ollama（229 MB，自動下載・安裝・啟動）").clicked() {
                                    self.spawn_install();
                                }
                            }
                        }
                        if ui.button("重試連線").clicked() {
                            *self.ollama.lock().unwrap() = Some(OllamaStatus::Checking);
                            self.spawn_ollama_check();
                        }
                    }
                    _ => { ui.label("○ 正在檢查 Ollama …"); }
                }
                if self.pull_running.load(Ordering::Relaxed) {
                    ui.spinner();
                    ui.label("模型下載中 … 完成後按「重試連線」");
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label("輸入資料夾");
                    ui.text_edit_singleline(&mut self.input_folder);
                    if ui.button("瀏覽…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_folder() {
                            self.input_folder = p.to_string_lossy().into_owned();
                            if self.zip_out.is_empty() {
                                self.zip_out = format!("{}/annotated.zip", self.input_folder);
                            }
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("輸出 ZIP   ");
                    ui.text_edit_singleline(&mut self.zip_out);
                });
            });

            ui.add_space(6.0);
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label("模型");
                    ui.text_edit_singleline(&mut self.model);
                    ui.label("語言");
                    egui::ComboBox::from_id_salt("lang")
                        .selected_text(self.language.clone())
                        .show_ui(ui, |ui| {
                            for (code, name) in LANGUAGES {
                                ui.selectable_value(&mut self.language, code.to_string(), format!("{} ({})", name, code));
                            }
                        });
                    ui.checkbox(&mut self.save_intermediate, "ZIP 內含中間 JSON");
                });
                ui.horizontal(|ui| {
                    ui.label("Ollama URL");
                    ui.text_edit_singleline(&mut self.ollama_url);
                });
                ui.horizontal(|ui| {
                    ui.label("掃描讀取引擎");
                    egui::ComboBox::from_id_salt("reader")
                        .selected_text(if self.reader == "vlm" { "VLM 手寫精讀（準，每頁約 30 分鐘）".to_string() } else { "PP-OCR 本機（快）".to_string() })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.reader, "vlm".into(), "VLM 手寫精讀（準，每頁約 30 分鐘）");
                            ui.selectable_value(&mut self.reader, "ppocr".into(), "PP-OCR 本機（快）");
                        });
                    if self.reader == "vlm" {
                        ui.label("視覺模型");
                        ui.text_edit_singleline(&mut self.vlm_model);
                        if !models_available(&self.ollama).contains(&self.vlm_model) {
                            ui.colored_label(egui::Color32::from_rgb(0xb2, 0x6a, 0x00), format!("需先 ollama pull {}", self.vlm_model));
                        }
                    }
                });
            });

            ui.add_space(8.0);
            let can_run = !self.input_folder.is_empty()
                && !self.zip_out.is_empty()
                && !self.state.lock().unwrap().running;
            ui.add_enabled_ui(can_run, |ui| {
                let btn = egui::Button::new("▶ 開始批改");
                if ui.add_sized([160.0, 32.0], btn).clicked() {
                    self.spawn_batch();
                }
            });
            if self.state.lock().unwrap().running {
                ui.add(egui::Spinner::new());
                ui.label("批次處理中 … 詳見下方記錄");
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
            }

            // results table
            {
                let s = self.state.lock().unwrap();
                if !s.results.is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.heading("結果");
                    egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
                        egui::Grid::new("results").striped(true).show(ui, |ui| {
                            ui.strong("檔案");
                            ui.strong("錯誤數");
                            ui.strong("狀態");
                            ui.end_row();
                            for (name, n, status) in &s.results {
                                ui.label(name);
                                ui.label(if *n > 0 { n.to_string() } else { "—".into() });
                                ui.label(status);
                                ui.end_row();
                            }
                        });
                    });
                    if let Some(zip) = &s.zip_path {
                        if ui.button("📂 開啟輸出 ZIP 所在資料夾").clicked() {
                            let _ = std::process::Command::new("open")
                                .arg("-R")
                                .arg(zip)
                                .spawn();
                        }
                    }
                }
            }

            // 即時模式（live camera）
            ui.add_space(8.0);
            ui.separator();
            ui.heading("即時模式（攝影機對準作文）");
            ui.horizontal(|ui| {
                let running = self.live.running.load(Ordering::Relaxed);
                if !running {
                    if ui.button("● 開啟相機").clicked() {
                        self.spawn_live(0);
                    }
                } else {
                    if ui.button("■ 關閉相機").clicked() {
                        self.live.running.store(false, Ordering::Relaxed);
                    }
                    ui.spinner();
                    ui.label("預覽與分析中 …");
                }
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(120));
            });
            // drain preview frames → texture
            loop {
                match self.live.preview_rx.try_recv() {
                    Ok(p) => {
                        let size = [p.w as usize, p.h as usize];
                        let color_image = egui::ColorImage::from_rgba_unmultiplied(
                            size,
                            &rgba_from_rgb(&p.rgb),
                        );
                        if let Some(tex) = self.live.texture.as_mut() {
                            tex.set(color_image, egui::TextureOptions::LINEAR);
                        } else {
                            self.live.texture = Some(ui.ctx().load_texture(
                                "live-preview",
                                color_image,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }
                    Err(_) => break,
                }
            }
            if let Some(tex) = &self.live.texture {
                let avail = ui.available_width();
                ui.image((tex.id(), egui::vec2(avail.min(640.0), avail.min(640.0) * 0.5625)));
            }
            // drain analysis results into the log
            loop {
                match self.live.result_rx.try_recv() {
                    Ok(msg) => eat_core::logging::info(&msg),
                    Err(_) => break,
                }
            }

            // log panel
            ui.add_space(8.0);
            ui.separator();
            ui.heading("記錄");
            egui::ScrollArea::vertical()
                .max_height(ui.available_height() - 4.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    let s = self.state.lock().unwrap();
                    for line in &s.log {
                        ui.monospace(line);
                    }
                });
        });
    }
}

fn models_available(ollama: &Arc<Mutex<Option<OllamaStatus>>>) -> Vec<String> {
    match &*ollama.lock().unwrap() {
        Some(OllamaStatus::Reachable(m)) => m.clone(),
        _ => Vec::new(),
    }
}

fn rgba_from_rgb(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks_exact(3) {
        out.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    out
}

fn main() -> eframe::Result {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 640.0])
            .with_title("AI 作文批改 Project-EAT"),
        ..Default::default()
    };
    eframe::run_native("eat-gui", native, Box::new(|cc| Ok(Box::new(EatGui::new(cc)))))
}
