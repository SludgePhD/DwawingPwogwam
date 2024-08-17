use std::{
    mem,
    ops::Range,
    process,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::bail;
use bytemuck::NoUninit;
use half::f16;
use wgpu::{
    util::{DeviceExt, TextureDataOrder},
    Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent,
    BlendFactor, BlendOperation, BlendState, Buffer, BufferBindingType, BufferDescriptor,
    BufferUsages, Color, ColorTargetState, ColorWrites, CompositeAlphaMode, Device,
    DeviceDescriptor, Extent3d, FilterMode, FragmentState, InstanceDescriptor, LoadOp, MemoryHints,
    MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, Sampler, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, Surface, SurfaceError,
    SurfaceTarget, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureView, TextureViewDimension, VertexState,
};
use winit::{
    application::ApplicationHandler,
    event::{StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    window::{Fullscreen, Window, WindowId, WindowLevel},
};

use crate::{
    cmd::{Cmd, Tool},
    config::Config,
    math::{lerp, vec2, vec3, vec4, Vec2f, Vec2u, Vec3f, Vec4f, Vec4h},
};

const ALPHA_MODE: CompositeAlphaMode = CompositeAlphaMode::PreMultiplied;

pub struct App {
    config: Config,
    instance: wgpu::Instance,
    win: Option<Win>,
}

/// Global renderer data.
struct Gpu {
    adapter: Adapter,
    device: Device,
    queue: Queue,
    /// Format of the window surface, used as the format of every render target.
    format: TextureFormat,

    /// Scheduled instances.
    instances: Vec<Instance>,
    /// Global instance buffer. Holds every drawable instance that will be drawn in the dispatch.
    instance_buf: Buffer,

    render_pipeline: RenderPipeline,
    erase_pipeline: RenderPipeline,
    sampler: Sampler,

    global_bgl: BindGroupLayout,
    pass_bgl: BindGroupLayout,
    drawable_bgl: BindGroupLayout,

    global_bg: Option<BindGroup>,
}

impl Gpu {
    fn new(
        instance: &wgpu::Instance,
        surface: &Surface<'_>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }));
        let Some(adapter) = adapter else {
            bail!("failed to find a supported graphics adapter")
        };

        let surface_caps = surface.get_capabilities(&adapter);
        if !surface_caps.alpha_modes.contains(&ALPHA_MODE) {
            bail!(
                "surface does not support required alpha compositing mode {:?} (supported: {:?})",
                ALPHA_MODE,
                surface_caps.alpha_modes,
            );
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                memory_hints: MemoryHints::MemoryUsage,
                ..Default::default()
            },
            None,
        ))?;

        let config = surface
            .get_default_config(&adapter, width, height)
            .expect("adapter does not support surface");

        // Shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // BGLs
        let global_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("global"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        });
        let pass_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pass"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                count: None,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }],
        });
        let drawable_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("drawable"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                count: None,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            }],
        });

        // Pipeline.
        let make_pipeline = |blend: BlendState| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&global_bgl, &pass_bgl, &drawable_bgl],
                    ..Default::default()
                })),
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vertex",
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: "fragment",
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: config.format,
                        blend: Some(blend),
                        write_mask: ColorWrites::all(),
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        let render_pipeline = make_pipeline(BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let erase_pipeline = make_pipeline(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::ReverseSubtract,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::ReverseSubtract,
            },
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let instance_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: mem::size_of::<Instance>() as u64, // 1 instance preallocated
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Gpu {
            adapter,
            device,
            queue,
            format: config.format,

            instances: Vec::new(),
            instance_buf,

            render_pipeline,
            erase_pipeline,
            sampler,

            global_bgl,
            pass_bgl,
            drawable_bgl,

            global_bg: None,
        })
    }

    fn schedule_instances(&mut self, instances: &[Instance]) -> Range<u32> {
        let start = self.instances.len() as u32;
        let end = start + instances.len() as u32;

        self.instances.extend_from_slice(instances);

        start..end
    }

    /// Writes all scheduled instances to the instance buffer (reallocating it if it is too small).
    fn prepare_instances(&mut self) {
        let size = (mem::size_of::<Instance>() * self.instances.len()) as u64;
        if self.instance_buf.size() < size {
            self.instance_buf = self.device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.global_bg = None;
        }
        self.queue
            .write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&self.instances));
        self.instances.clear();
    }

    fn global_bind_group(&mut self) -> &BindGroup {
        self.global_bg.get_or_insert_with(|| {
            self.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.global_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Sampler(&self.sampler),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(
                            self.instance_buf.as_entire_buffer_binding(),
                        ),
                    },
                ],
            })
        })
    }
}

struct Win {
    window: Arc<Window>,
    surface: Surface<'static>,
    gpu: Gpu,

    pass_paint: PassData,
    pass_display: PassData,

    canvas: Drawable,
    brush: Drawable,
    cursor_draw: Drawable,
    eraser: Drawable,
    cursor_erase: Drawable,

    tool: Tool,
    cursor_pos: Option<Vec2f>,
    stroke: Stroke,
}

impl Win {
    fn recreate_swapchain(&self) {
        let res = self.window.inner_size();

        let mut config = self
            .surface
            .get_default_config(&self.gpu.adapter, res.width, res.height)
            .expect("adapter does not support surface");
        config.alpha_mode = ALPHA_MODE;

        log::debug!(
            "configuring window surface for {}x{} (format: {:?}, present mode: {:?}, alpha mode: {:?})",
            res.width,
            res.height,
            config.format,
            config.present_mode,
            config.alpha_mode,
        );

        self.surface.configure(&self.gpu.device, &config);
    }

    fn redraw(&mut self) {
        let st = match self.surface.get_current_texture() {
            Ok(st) => st,
            Err(err @ (SurfaceError::Outdated | SurfaceError::Lost)) => {
                log::debug!("surface error: {}", err);
                self.recreate_swapchain();
                self.surface
                    .get_current_texture()
                    .expect("failed to acquire next frame after recreating swapchain")
            }
            Err(e) => {
                panic!("failed to acquire frame: {}", e);
            }
        };

        let size = self.window.inner_size();
        let canvas_pos = vec2(size.width as f32 * 0.5, size.height as f32 * 0.5);

        let mut rec = Vec::new();

        let mut pass = self.pass_paint.start(
            &mut self.gpu,
            &self.canvas.texture,
            None,
            self.tool == Tool::Erase,
        );
        match self.tool {
            Tool::Draw => self.stroke.draw(&mut pass, &mut self.brush),
            Tool::Erase => self.stroke.draw(&mut pass, &mut self.eraser),
            Tool::MoveCanvas => self.stroke.discard(),
        }
        rec.push(pass.finish());

        // Draw the canvas and cursor onto the window surface.
        let mut pass =
            self.pass_display
                .start(&mut self.gpu, &st.texture, Some(Color::TRANSPARENT), false);
        self.canvas.draw_at(&mut pass, canvas_pos);
        if let Some(pos) = self.cursor_pos {
            match self.tool {
                Tool::Draw => self.cursor_draw.draw_at(&mut pass, pos),
                Tool::Erase => self.cursor_erase.draw_at(&mut pass, pos),
                Tool::MoveCanvas => {} // TODO
            }
        }
        rec.push(pass.finish());

        // Write all scheduled instances before submission.
        self.gpu.prepare_instances();

        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        for rec in rec {
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &rec.target,
                    resolve_target: None,
                    ops: Operations {
                        load: if let Some(clear) = rec.clear {
                            LoadOp::Clear(clear)
                        } else {
                            LoadOp::Load
                        },
                        ..Default::default()
                    },
                })],
                ..Default::default()
            });
            if rec.erase {
                pass.set_pipeline(&self.gpu.erase_pipeline);
            } else {
                pass.set_pipeline(&self.gpu.render_pipeline);
            }
            pass.set_bind_group(0, self.gpu.global_bind_group(), &[]);
            pass.set_bind_group(1, &rec.data.pass_bg, &[]);

            for draw in rec.draws {
                pass.set_bind_group(2, &draw.drawable_bg, &[]);
                pass.draw(0..4, draw.instances);
            }
        }

        self.gpu.queue.submit([enc.finish()]);
        self.window.pre_present_notify();
        st.present();
    }
}

impl App {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        Ok(Self {
            config,
            instance: wgpu::Instance::new(InstanceDescriptor {
                backends: Backends::PRIMARY,
                ..Default::default()
            }),
            win: None,
        })
    }

    fn create_win(&self, event_loop: &ActiveEventLoop) -> anyhow::Result<Win> {
        let mut target_monitor = None;
        for monitor in event_loop.available_monitors() {
            let Some(name) = monitor.name() else {
                continue;
            };
            let size = monitor.size();
            let pos = monitor.position();
            log::debug!(
                "monitor '{name}': {}x{} @ ({},{})",
                size.width,
                size.height,
                pos.x,
                pos.y
            );

            if Some(&name) == self.config.monitor.as_ref() {
                target_monitor = Some(monitor);
            }
        }

        if let Some(mon) = &self.config.monitor {
            if target_monitor.is_none() {
                bail!("failed to find configured monitor output '{mon}'");
            }
        }

        if target_monitor.is_none() {
            // Try to always use the primary monitor by default. Note that Wayland does not support
            // this, so we'll end up with the *current* monitor there.
            target_monitor = event_loop.primary_monitor();
        }

        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_window_level(WindowLevel::AlwaysOnTop)
                    .with_transparent(true)
                    .with_fullscreen(Some(Fullscreen::Borderless(target_monitor)))
                    .with_title("Dwawing Pwogwam"),
            )?,
        );
        window.set_cursor_hittest(false)?;

        let surface = self
            .instance
            .create_surface(SurfaceTarget::from(window.clone()))?;
        let res = window.inner_size();
        let mut gpu = Gpu::new(&self.instance, &surface, res.width, res.height)?;

        let size = window.inner_size();
        log::debug!(
            "creating canvas at {}x{}, format={:?}",
            size.width,
            size.height,
            gpu.format
        );
        let canvas = Drawable::empty(&gpu, size.width, size.height);

        let brush = Brush {
            size: 3,
            color: vec3(1.0, 1.0, 1.0),
        }
        .into_drawable(&gpu);
        let cursor_draw = Brush {
            size: 5,
            color: vec3(1.0, 0.5, 0.5),
        }
        .into_drawable(&gpu);
        let eraser = Brush {
            size: 20,
            color: vec3(1.0, 1.0, 1.0),
        }
        .into_drawable(&gpu);
        let cursor_erase = Brush {
            size: 20,
            color: vec3(1.0, 1.0, 1.0),
        }
        .into_drawable(&gpu);
        Ok(Win {
            pass_paint: PassData::new(&mut gpu),
            pass_display: PassData::new(&mut gpu),
            window,
            surface,
            gpu,
            canvas,
            brush,
            cursor_draw,
            eraser,
            cursor_erase,

            tool: Tool::Draw,
            cursor_pos: None,
            stroke: Stroke::new(1.0),
        })
    }
}

impl ApplicationHandler<Cmd> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.win.is_none() {
            let win = match self.create_win(event_loop) {
                Ok(win) => win,
                Err(e) => {
                    eprintln!("could not create window: {e}");
                    process::exit(1);
                }
            };
            self.win = Some(win);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(win) = &mut self.win else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => win.redraw(),
            WindowEvent::Resized(_) => {
                win.recreate_swapchain();
                win.window.request_redraw();
            }
            _ => {}
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: Cmd) {
        let Some(win) = &mut self.win else { return };
        win.window.request_redraw();
        let size = win.window.inner_size();

        match event {
            Cmd::Clear => {
                log::info!("clearing canvas");
                win.canvas = Drawable::empty(&win.gpu, size.width, size.height);
            }
            Cmd::PenMove {
                position,
                aspect_ratio: _a, // FIXME: we don't adjust for this yet
                pressure,
            } => {
                // Position is in range 0..1 with the top left being 0,0
                let size = vec2(size.width as f32, size.height as f32);
                let pos = position * size;
                win.stroke.append_movement(pos, pressure);

                win.cursor_pos = Some(pos);

                // There's no indication when the pen goes out of the detection range, so we use a
                // timeout instead.
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + Duration::from_millis(100),
                ));
            }
            Cmd::SetTool { tool } => {
                win.tool = tool;
            }
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        let Some(win) = &mut self.win else { return };
        match cause {
            StartCause::ResumeTimeReached { .. } => {
                win.cursor_pos = None;
                win.window.request_redraw();
                event_loop.set_control_flow(ControlFlow::default());
            }
            _ => {}
        }
    }
}

struct Stroke {
    /// Spacing between brush impressions, in pixels.
    spacing_px: f32,
    /// Queued impressions that should be drawn for this stroke.
    impressions: Vec<Instance>,
    last_draw: Option<Instance>,
}

impl Stroke {
    fn new(spacing_px: f32) -> Self {
        Self {
            spacing_px,
            impressions: Vec::new(),
            last_draw: None,
        }
    }

    /// Appends a pen movement onto the end of this stroke.
    ///
    /// `position` and `pressure` are the pen position (in canvas pixels) and pressure (in range
    /// 0-1) at the end of the movement.
    ///
    /// A pressure equal to `0.0` ends the stroke.
    fn append_movement(&mut self, position: Vec2f, pressure: f32) {
        if pressure == 0.0 {
            self.last_draw = None;
            return;
        }

        let opacity = pressure; // (this would be a good place to apply a pressure curve)

        // FIXME: we should `max(prev, new)` the opacity when we're too close to the last impression to add a new one
        // issue is, this can happen across multiple frames, and then the old impression is already drawn.
        // sounds like this isn't solveable without putting the in-progress stroke on a temporary layer?
        // Turns out, Krita has (what seems like) the same behavior, so I guess this is fine?

        match self.last_draw {
            Some(last) => {
                // Place impressions exactly `spacing_px` away from the last drawn one, until we
                // reach the end of the new pen movement, while interpolating pen pressure across
                // the movement.
                let total_dist = (position - last.pos).length();
                let n = total_dist / self.spacing_px;
                for i in 0..n.floor() as usize {
                    let t = i as f32 / n;
                    let opacity = lerp(last.opacity..=opacity, t);
                    let pos = lerp(last.pos..=position, t);
                    let impression = Instance::new(pos, opacity);
                    self.last_draw = Some(impression);
                    self.impressions.push(impression);
                }
            }
            None => {
                let impression = Instance::new(position, opacity);
                self.last_draw = Some(impression);
                self.impressions.push(impression);
            }
        }
    }

    /// Discards the brush impressions that *would* be drawn by [`Stroke::draw`].
    fn discard(&mut self) {
        self.impressions.clear();
    }

    /// Schedules the queued impressions to be drawn with `brush`.
    fn draw<'a>(&mut self, pass: &mut Pass<'_, 'a>, brush: &'a mut Drawable) {
        brush.draw_instances(pass, &self.impressions);
        self.discard();
    }
}

#[derive(Clone, Copy, NoUninit)]
#[repr(C)]
struct PassUniforms {
    render_target_size: Vec2u,
}

#[derive(Debug, Clone, Copy, NoUninit)]
#[repr(C)]
struct Instance {
    /// Center position in pixel coordinates.
    pos: Vec2f,
    opacity: f32,
    _padding: f32,
}

impl Instance {
    fn new(position: Vec2f, opacity: f32) -> Self {
        Self {
            pos: position,
            opacity,
            _padding: 0.0,
        }
    }
}

struct PassData {
    uniforms: Buffer,
    pass_bg: BindGroup,
}

impl PassData {
    fn new(gpu: &mut Gpu) -> Self {
        let uniforms = gpu.device.create_buffer(&BufferDescriptor {
            label: None,
            size: mem::size_of::<PassUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pass_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &gpu.pass_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(uniforms.as_entire_buffer_binding()),
            }],
        });

        Self { uniforms, pass_bg }
    }

    fn start<'gpu, 'a>(
        &'a self,
        gpu: &'gpu mut Gpu,
        target: &Texture,
        clear: Option<Color>,
        erase: bool,
    ) -> Pass<'gpu, 'a> {
        let uniforms = PassUniforms {
            render_target_size: vec2(target.width(), target.height()),
        };
        gpu.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));

        Pass {
            gpu,
            clear,
            target: target.create_view(&Default::default()),
            data: self,
            draws: Vec::new(),
            erase,
        }
    }
}

/// A buffered draw operation.
struct Draw<'a> {
    drawable_bg: &'a BindGroup,
    instances: Range<u32>,
}

struct Pass<'gpu, 'a> {
    gpu: &'gpu mut Gpu,
    clear: Option<Color>,
    erase: bool,
    target: TextureView,
    data: &'a PassData,
    draws: Vec<Draw<'a>>,
}

struct RecordedPass<'a> {
    draws: Vec<Draw<'a>>,
    target: TextureView,
    data: &'a PassData,
    clear: Option<Color>,
    erase: bool,
}

impl<'a> Pass<'_, 'a> {
    fn finish(self) -> RecordedPass<'a> {
        RecordedPass {
            draws: self.draws,
            target: self.target,
            data: self.data,
            clear: self.clear,
            erase: self.erase,
        }
    }
}

struct Drawable {
    texture: Texture,
    drawable_bg: BindGroup,
}

impl Drawable {
    fn empty(gpu: &Gpu, width: u32, height: u32) -> Self {
        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: gpu.format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self::from_texture(gpu, texture)
    }

    fn from_texture(gpu: &Gpu, texture: Texture) -> Self {
        let drawable_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &gpu.drawable_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&texture.create_view(&Default::default())),
            }],
        });

        Self {
            texture,
            drawable_bg,
        }
    }

    fn draw_at<'a>(&'a self, p: &mut Pass<'_, 'a>, position: Vec2f) {
        self.draw_instances(p, &[Instance::new(position, 1.0)]);
    }

    fn draw_instances<'a>(&'a self, p: &mut Pass<'_, 'a>, instances: &[Instance]) {
        let range = p.gpu.schedule_instances(instances);
        p.draws.push(Draw {
            drawable_bg: &self.drawable_bg,
            instances: range.clone(),
        });
    }
}

struct Brush {
    size: u32,
    color: Vec3f,
}

impl Brush {
    fn sample(&self, pos: Vec2f) -> Vec4f {
        let strength = 1.0 - pos.length();
        let c = self.color;
        vec4(c[0], c[1], c[2], 1.0) * strength
    }

    fn into_drawable(self, gpu: &Gpu) -> Drawable {
        let mut pixels = vec![Vec4h::ZERO; (self.size * self.size) as usize];
        for y in 0..self.size {
            let yf = (y as f32 + 0.5) / self.size as f32;
            for x in 0..self.size {
                let xf = (x as f32 + 0.5) / self.size as f32;
                let uv = vec2(xf, yf);
                let offset = uv - vec2(0.5, 0.5);

                // NB: not perceptually uniform
                let color = self.sample(offset * 2.0);
                pixels[(y * self.size + x) as usize] = color.map(f16::from_f32);
            }
        }

        Drawable::from_texture(
            gpu,
            gpu.device.create_texture_with_data(
                &gpu.queue,
                &TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: self.size,
                        height: self.size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba16Float, // (f32 is not filterable by default)
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                bytemuck::cast_slice(&pixels),
            ),
        )
    }
}
