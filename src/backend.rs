use crate::gpu::DeviceSelector;
use crate::inference::EncoderBackend;

/// Compute backend selection.
///
/// The GPU here is whatever wgpu can drive — Vulkan, D3D12, Metal or GL — and
/// there is deliberately no variant naming one vendor's runtime.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl Backend {
    /// The best backend this build/machine can offer — i.e. [`Backend::Auto`].
    ///
    /// Exists so a caller can write the intent ("give me the good one") without
    /// hard-coding a variant; it is infallible, because choosing between two
    /// enum variants cannot fail.
    #[must_use]
    pub const fn best() -> Self {
        Backend::Auto
    }

    /// Short human label — for logs and for asserting what got picked.
    #[must_use]
    pub const fn tag(&self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }

    pub(crate) fn resolve(self) -> crate::Result<(DeviceSelector, EncoderBackend)> {
        match self {
            Backend::Auto => Ok((DeviceSelector::Auto, EncoderBackend::Gpu)),
            Backend::Cpu => Ok((DeviceSelector::Cpu, EncoderBackend::Cpu)),
            Backend::Gpu => {
                if !has_gpu_adapter() {
                    return Err(crate::AsrError::ModelLoad(anyhow::anyhow!(
                        "no GPU adapter: wgpu sees no device that is not itself a CPU \
                         implementation — use Backend::Cpu, or Backend::Auto to fall back"
                    )));
                }
                Ok((DeviceSelector::Auto, EncoderBackend::Gpu))
            }
        }
    }
}

fn has_gpu_adapter() -> bool {
    crate::inference::AsrInference::device_targets()
        .iter()
        .any(|t| t.info.device_type != wgpu::DeviceType::Cpu)
}
