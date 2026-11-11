fn main() {
    let r = eat_app::report::RubricCfg::default();
    println!("整體要求: {}", r.criteria);
    for d in &r.dims {
        println!("[{}] {:.0} 分 — {}", d.name, d.max, d.desc);
    }
    println!("總滿分: {:.0}", r.total_max());
}
