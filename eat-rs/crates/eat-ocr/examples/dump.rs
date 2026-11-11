//! Debug helper: run OCR on an image and print lines as JSON.
use eat_ocr::imgops::Img;
fn main() {
    let path = std::env::args().nth(1).expect("usage: dump <img>");
    let img = image::open(&path).unwrap().to_rgb8();
    let img = Img::from_rgb(img.width() as usize, img.height() as usize, &img);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/ppocr");
    std::env::set_var("EAT_PPROCR_DIR", std::fs::canonicalize(&dir).unwrap_or(dir));
    let mut ocr = eat_ocr::RapidOcr::load().unwrap();
    let t0 = std::time::Instant::now();
    let lines = ocr.run(&img).unwrap();
    eprintln!("{} lines in {:.1}s", lines.len(), t0.elapsed().as_secs_f32());
    for l in &lines {
        let b: Vec<String> = l.box_.iter().map(|(x, y)| format!("{:.0},{:.0}", x, y)).collect();
        println!("{}\t{}\t{}", format!("{:.2}", l.score), l.text, b.join(" "));
    }
}
