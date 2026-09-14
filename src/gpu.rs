//! Device / queue wrapper and buffer plumbing.
//!
//! Everything the engine needs from wgpu lives here so the rest of the crate can
//! stay in terms of buffers + dispatches.  Two conventions matter for the whole
//! engine and are enforced by the constructors below:
//!
//! * storage buffers are always allocated 16-byte padded, so a `array<vec4<u32>>`
//!   view over an f16 payload never runs off the end;
//! * every activation tensor is stored the way CUDA stores it — f16, indexed as
//!   `__half2` words (2 halves per `u32`).  Keeping the byte layout identical to
//!   the CUDA backend is what lets the two store bit-comparable values.

use anyhow::{bail, Context, Result};

/// A wgpu device plus the queue, adapter info and negotiated limits.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
}

impl Gpu {
    /// Enumerate adapters and pick one.  `prefer` matches a case-insensitive
    /// substring of the adapter name (e.g. `"nvidia"`, `"intel"`); without it the
    /// first discrete GPU wins, falling back to whatever is available.
    pub async fn new(prefer: Option<&str>) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        if adapters.is_empty() {
            bail!("no wgpu adapters found");
        }

        let mut chosen = None;
        if let Some(w) = prefer {
            let w = w.to_lowercase();
            chosen = adapters
                .iter()
                .find(|a| a.get_info().name.to_lowercase().contains(&w));
            if chosen.is_none() {
                bail!("no adapter matching {:?}", w);
            }
        }
        let adapter = match chosen {
            Some(a) => a,
            None => adapters
                .iter()
                .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
                .unwrap_or(&adapters[0]),
        };

        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("qwen3-asr-wgpu"),
                // Timestamp queries gate the per-op step profiler; SUBGROUP lets
                // `gemv` use warp shuffles instead of shared-memory butterflies.
                // All three are intersected with what the adapter reports, so an
                // adapter without them still gets a device (callers branch on
                // `Gpu::features`).  See `docs/wgpu-best-practices-audit.md`.
                required_features: features
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                        | wgpu::Features::SUBGROUP),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .context("request_device")?;

        // surface async validation errors / device loss instead of dying later
        // at an unrelated map with a bare "async map a buffer"
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            eprintln!("[wgpu uncaptured error] {e}");
        }));

        Ok(Self {
            device,
            queue,
            info,
            limits,
            features,
        })
    }

    /// One-line description for logs and reports.
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}) | {} {} | wg workgroup storage {} B, maxStorageBufferBindingSize {} MiB",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info,
            self.limits.max_compute_workgroup_storage_size,
            self.limits.max_storage_buffer_binding_size / (1024 * 1024),
        )
    }

    /// Storage buffer sized `bytes` rounded up to a 16-byte multiple.
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 15) & !15;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Runtime-sized deferred write (queue ordering makes it visible to work
    /// submitted afterwards).  For multi-megabyte *load-time* transfers use
    /// [`Gpu::uploader`] instead — deferred staging must be bounded.
    pub fn upload(&self, buf: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buf, 0, data);
    }

    /// Submit an empty command buffer and wait for the queue to drain.  This
    /// retires every deferred `write_buffer` copy and frees their staging.
    pub fn flush(&self) -> Result<()> {
        let enc = self.device.create_command_encoder(&Default::default());
        self.queue.submit([enc.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for flush")?;
        Ok(())
    }

    /// Start a bulk upload session (load-time weight transfer).
    pub fn uploader(&self) -> BulkUpload<'_> {
        BulkUpload { gpu: self, pending: 0 }
    }

    pub fn readback(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let size = (bytes + 3) & !3;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size.max(4));
        self.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll for readback")?;
        rx.recv().context("map callback dropped")?.context("map buffer")?;
        let mut data = slice.get_mapped_range()?.to_vec();
        drop(slice);
        staging.unmap();
        data.truncate(bytes as usize);
        Ok(data)
    }

    /// Compile a WGSL module + compute pipeline, surfacing validation errors
    /// instead of letting them turn into an opaque panic later.  `layout`
    /// attaches an explicit pipeline layout — required whenever one bind group
    /// is shared across sibling pipelines, because wgpu's implicit layouts are
    /// pipeline-exclusive.
    pub fn pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let err = pollster::block_on(guard.pop());
        if let Some(e) = err {
            bail!("pipeline {label} failed validation: {e}");
        }
        Ok(pipe)
    }
}

/// Outstanding deferred-copy bytes tolerated before a flush.
///
/// wgpu defers every `queue.write_buffer` to the next submit and keeps the
/// device-side staging alive until then.  Letting a multi-GiB model load
/// accumulate into one submit overflows VRAM on WDDM drivers: copies silently
/// land as zeros and the device is lost on the next dispatch (observed on
/// Pascal for a 3.4 GiB load; 1.2 GiB happened to survive).  The CUDA engine
/// never holds staging because `htod` copies are synchronous — this budget is
/// the wgpu counterpart of that discipline: stage a bounded amount, then
/// submit + wait, which retires the staging, before staging more.
const STAGING_BUDGET: u64 = 256 << 20;

/// Load-time upload session with bounded outstanding staging.
///
/// Route every model-load transfer through one of these (`Gpu::uploader`);
/// runtime-sized writes (rope tables, token slots, KV seeding) stay on the
/// deferred [`Gpu::upload`] path, where queue ordering already guarantees
/// visibility and the volumes are tiny.
pub struct BulkUpload<'a> {
    gpu: &'a Gpu,
    pending: u64,
}

impl<'a> BulkUpload<'a> {
    /// Allocate a 16B-padded storage buffer (as [`Gpu::storage`]).
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.gpu.storage(label, bytes)
    }

    /// Allocate a uniform buffer (as [`Gpu::uniform`]).
    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.gpu.uniform(label, bytes)
    }

    /// Stage `data` into `buf`.  Pieces larger than the budget are split so a
    /// single staging buffer can never exceed it.
    pub fn upload(&mut self, buf: &wgpu::Buffer, data: &[u8]) -> Result<()> {
        let budget = (STAGING_BUDGET as usize).max(1);
        let mut off = 0usize;
        for piece in data.chunks(budget) {
            if self.pending + piece.len() as u64 > STAGING_BUDGET {
                self.gpu.flush()?;
                self.pending = 0;
            }
            self.gpu.queue.write_buffer(buf, off as u64, piece);
            self.pending += piece.len() as u64;
            off += piece.len();
        }
        Ok(())
    }

    /// Retire whatever is still staged.  Call before the first real dispatch.
    pub fn finish(self) -> Result<()> {
        if self.pending > 0 {
            self.gpu.flush()?;
        }
        Ok(())
    }
}
