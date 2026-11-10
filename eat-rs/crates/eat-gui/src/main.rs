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

struct EatGui {
    input_folder: String,
    zip_out: String,
    model: String,
    ollama_url: String,
    language: String,
    save_intermediate: bool,
    state: Arc<Mutex<BatchState>>,
    ollama: Arc<Mutex<Option<OllamaStatus>>>,
    pull_running: Arc<AtomicBool>,
}

impl Default for EatGui {
    fn default() -> Self {
        let state = Arc::new(Mutex::new(BatchState::default()));
        install_log_sink(state.clone());
        let gui = EatGui {
            input_folder: String::new(),
            zip_out: String::new(),
            model: default_model_name(),
            ollama_url: default_ollama_url(),
            language: default_language(),
            save_intermediate: false,
            state,
            ollama: Arc::new(Mutex::new(Some(OllamaStatus::Checking))),
            pull_running: Arc::new(AtomicBool::new(false)),
        };
        gui.spawn_ollama_check();
        gui
    }
}

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

    fn spawn_batch(&self) {
        let opts = PipelineOpts {
            input_pdf: None,
            output: None,
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
                        ui.hyperlink_to("安裝 Ollama", "https://ollama.com/download");
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

fn main() -> eframe::Result {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([760.0, 640.0])
            .with_title("AI 作文批改 Project-EAT"),
        ..Default::default()
    };
    eframe::run_native("eat-gui", native, Box::new(|_cc| Ok(Box::new(EatGui::default()))))
}
