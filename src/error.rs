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

    /// `BatchEvents::should_stop` asked for the remaining items to be abandoned.
    /// Only the items that had not finished get this; the ones already delivered
    /// keep their transcript.
    #[error("cancelled")]
    Cancelled,
}

/// [`AsrError`]-typed result — the return of every public entry point.
pub type Result<T> = std::result::Result<T, AsrError>;
