//! Debug: run the isolated DB post-process on a dumped .npy probmap.
//! usage: dbpp <pred.npy> <pred_w> <pred_h> <img_w> <img_h>
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&args[1]).unwrap();
    // minimal .npy parse: magic(6) + ver(2) + header_len(2 LE) + header + data
    let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
    let data = &raw[10 + hlen..];
    let pred: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (pw, ph, iw, ih): (usize, usize, usize, usize) = (
        args[2].parse().unwrap(),
        args[3].parse().unwrap(),
        args[4].parse().unwrap(),
        args[5].parse().unwrap(),
    );
    assert_eq!(pred.len(), pw * ph, "npy size mismatch");
    let boxes = eat_ocr::db_postprocess(&pred, pw, ph, iw, ih);
    eprintln!("rust boxes: {}", boxes.len());
    for b in &boxes {
        let pts: Vec<String> = b.iter().map(|(x, y)| format!("{:.0},{:.0}", x, y)).collect();
        println!("{}", pts.join(" "));
    }
}
