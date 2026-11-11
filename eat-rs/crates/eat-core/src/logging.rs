//! Minimal stderr logger with HH:MM:SS timestamps plus an optional
//! subscriber hook (installed by the GUI to mirror log lines into its panel).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static LEVEL: AtomicU8 = AtomicU8::new(1); // 0=debug 1=info

type Sink = Box<dyn Fn(&str) + Send + Sync>;
static SINK: Mutex<Option<Sink>> = Mutex::new(None);

pub fn set_debug(debug: bool) {
    LEVEL.store(if debug { 0 } else { 1 }, Ordering::Relaxed);
}

/// Install a log subscriber (e.g. the GUI log panel). Pass None to remove.
pub fn subscribe(sink: Option<Sink>) {
    *SINK.lock().unwrap() = sink;
}

fn emit(level: &str, msg: &str) {
    let line = format!("{}  {:<7} eat  {}", timestamp(), level, msg);
    eprintln!("{}", line);
    if let Ok(guard) = SINK.lock() {
        if let Some(f) = guard.as_ref() {
            f(&line);
        }
    }
}

fn timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let day = secs % 86_400;
    format!("{:02}:{:02}:{:02}", day / 3600, (day % 3600) / 60, day % 60)
}

pub fn debug(msg: &str) {
    if LEVEL.load(Ordering::Relaxed) == 0 {
        emit("DEBUG", msg);
    }
}

pub fn info(msg: &str) {
    emit("INFO", msg);
}

pub fn warn(msg: &str) {
    emit("WARNING", msg);
}

pub fn error(msg: &str) {
    emit("ERROR", msg);
}
