use crate::gpu::DeviceSelector;
use crate::inference::EncoderBackend;

/// Compute backend selection.
///
/// The GPU here is whatever wgpu can drive — Vulkan, D3D12, Metal or GL — and
/// there is deliberately no variant naming one vendor's runtime.
///
/// * [`Self::Auto`] — one real GPU, or the CPU towers when none can be opened.
/// * [`Self::Cpu`] — the CPU towers. No adapter.
/// * [`Self::Gpu`] — the best GPU. No adapter, a failed open, or a GPU audio
///   tower that will not build is an error, not a silent CPU run.
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

    /// Device, audio tower, and whether a failed open may use the CPU towers.
    ///
    /// Only [`Self::Auto`] sets the fallback. [`Self::Gpu`] still selects
    /// [`DeviceSelector::Auto`] (the best card) and leaves the error to the load.
    pub(crate) fn plan(self) -> DevicePlan {
        match self {
            Backend::Auto => DevicePlan {
                selector: DeviceSelector::Auto,
                encoder: EncoderBackend::Gpu,
                allow_fallback: true,
            },
            Backend::Cpu => DevicePlan {
                selector: DeviceSelector::Cpu,
                encoder: EncoderBackend::Cpu,
                allow_fallback: false,
            },
            Backend::Gpu => DevicePlan {
                selector: DeviceSelector::Auto,
                encoder: EncoderBackend::Gpu,
                allow_fallback: false,
            },
        }
    }
}

/// What [`Backend::plan`] decided, before any weight is read.
pub(crate) struct DevicePlan {
    pub selector: DeviceSelector,
    pub encoder: EncoderBackend,
    pub allow_fallback: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_fallback_is_only_auto() {
        let auto = Backend::Auto.plan();
        assert!(auto.allow_fallback);
        assert_eq!(auto.selector, DeviceSelector::Auto);
        assert_eq!(auto.encoder, EncoderBackend::Gpu);

        let cpu = Backend::Cpu.plan();
        assert!(!cpu.allow_fallback);
        assert_eq!(cpu.selector, DeviceSelector::Cpu);
        assert_eq!(cpu.encoder, EncoderBackend::Cpu);

        let gpu = Backend::Gpu.plan();
        assert!(!gpu.allow_fallback);
        assert_eq!(gpu.selector, DeviceSelector::Auto);
        assert_eq!(gpu.encoder, EncoderBackend::Gpu);
    }
}
