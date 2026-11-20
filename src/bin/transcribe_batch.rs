//! Deprecated: `transcribe` now takes a list of `--wav` files, a JSONL
//! manifest, and `--batch`, so this binary only forwards to it.
//!
//! Kept so existing scripts keep working; it prints one line to stderr and then
//! runs exactly what `transcribe` would run for the same arguments.

#[path = "transcribe.rs"]
mod cli;

fn main() -> anyhow::Result<()> {
    eprintln!(
        "[transcribe_batch] deprecated: use `transcribe --batch N --wav a.wav b.wav ...` or `--manifest items.jsonl`; forwarding."
    );
    let mut args: Vec<String> = std::env::args().collect();
    // `transcribe_batch` took a variadic `--wav` list already, which is what
    // `transcribe` now does too, so the arguments need no translation.
    if args.len() > 1 {
        args.remove(0);
    }
    cli::run(args)
}
