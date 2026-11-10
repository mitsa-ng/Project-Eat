//! Minimal dependency-free stderr logger with HH:MM:SS timestamps,
//! matching the Python pipeline's log format.

use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static LEVEL: AtomicU8 = AtomicU8::new(1); // 0=debug 1=info

pub fn set_debug(debug: bool) {
    LEVEL.store(if debug { 0 } else { 1 }, Ordering::Relaxed);
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
        eprintln!("{}  DEBUG   eat  {}", timestamp(), msg);
    }
}

pub fn info(msg: &str) {
    eprintln!("{}  INFO    eat  {}", timestamp(), msg);
}

pub fn warn(msg: &str) {
    eprintln!("{}  WARNING eat  {}", timestamp(), msg);
}

pub fn error(msg: &str) {
    eprintln!("{}  ERROR   eat  {}", timestamp(), msg);
}
