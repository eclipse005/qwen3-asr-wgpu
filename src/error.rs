/// Everything a public entry point can fail with.
#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum AsrError {
    #[error("Model load failed: {0}")]
    ModelLoad(#[source] anyhow::Error),

    #[error("Audio decode failed: {0}")]
    AudioDecode(#[source] anyhow::Error),

    #[error("Inference failed: {0}")]
    Inference(#[source] anyhow::Error),

    #[error("Invalid options: {0}")]
    InvalidOptions(String),
}

/// [`AsrError`]-typed result — the return of every public entry point.
pub type Result<T> = std::result::Result<T, AsrError>;
