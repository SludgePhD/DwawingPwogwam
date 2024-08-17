use std::{
    mem, process,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::bail;
use bytemuck::NoUninit;
use wgpu::{
    util::{DeviceExt, TextureDataOrder},
    Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoder, CompositeAlphaMode, Device, DeviceDescriptor, Extent3d,
    FilterMode, FragmentState, InstanceDescriptor, LoadOp, MemoryHints, MultisampleState,
    Operations, PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState,
    PrimitiveTopology, Queue, RenderPass, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, Surface, SurfaceError,
    SurfaceTarget, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDimension, VertexState,
};
use winit::{
    application::ApplicationHandler,
    event::{StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    window::{Fullscreen, Window, WindowId, WindowLevel},
};

use crate::{
    cmd::Cmd,
    math::{lerp, vec2, Vec2f, Vec2u},
};

const ALPHA_MODE: CompositeAlphaMode = CompositeAlphaMode::PreMultiplied;

pub struct App {
    instance: wgpu::Instance,
    win: Option<Win>,
}

struct Gpu {
    adapter: Adapter,
    device: Device,
    queue: Queue,
    /// Format of the window surface, used as the format of every render target.
    format: TextureFormat,

    render_pipeline: RenderPipeline,
    sampler_bg: BindGroup,

    texture_bgl: BindGroupLayout,
    uniforms_bgl: BindGroupLayout,
    instances_bgl: BindGroupLayout,
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
        let sampler_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("sampler"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                count: None,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
            }],
        });
        let texture_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("texture"),
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
        let uniforms_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("uniforms"),
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
        let instances_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("instances"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                count: None,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }],
        });

        // Pipeline.
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("main_render_pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("main_render_pipeline"),
                bind_group_layouts: &[&sampler_bgl, &texture_bgl, &uniforms_bgl, &instances_bgl],
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
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::all(),
                })],
            }),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let sampler_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("sampler"),
            layout: &sampler_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(&sampler),
            }],
        });

        Ok(Gpu {
            adapter,
            device,
            queue,
            format: config.format,
            render_pipeline,
            sampler_bg,
            texture_bgl,
            uniforms_bgl,
            instances_bgl,
        })
    }
}

struct Win {
    window: Arc<Window>,
    surface: Surface<'static>,
    gpu: Gpu,

    canvas: Drawable,
    brush: Drawable,
    cursor_draw: Drawable,

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

        self.update_cursor();
        let size = self.window.inner_size();
        self.canvas.set_position(
            &self.gpu,
            vec2(size.width as f32 * 0.5, size.height as f32 * 0.5),
        );

        let mut enc = self.gpu.device.create_command_encoder(&Default::default());

        self.stroke.put_impressions(&self.gpu, &mut self.brush);

        let mut pass = Pass::new(&self.gpu, &mut enc, &self.canvas.texture, None);
        self.brush.draw(&mut pass);
        drop(pass);

        // Draw the canvas and cursor onto the window surface.
        let mut pass = Pass::new(&self.gpu, &mut enc, &st.texture, Some(Color::TRANSPARENT));
        self.canvas.draw(&mut pass);
        self.cursor_draw.draw(&mut pass);
        drop(pass);

        self.gpu.queue.submit([enc.finish()]);
        self.window.pre_present_notify();
        st.present();
    }

    fn update_cursor(&mut self) {
        // (later: configure the right type of cursor)
        match self.cursor_pos {
            Some(pos) => self.cursor_draw.set_position(&self.gpu, pos),
            None => self.cursor_draw.clear(),
        }
    }
}

impl App {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: wgpu::Instance::new(InstanceDescriptor {
                backends: Backends::PRIMARY,
                ..Default::default()
            }),
            win: None,
        })
    }

    fn create_win(&self, event_loop: &ActiveEventLoop) -> anyhow::Result<Win> {
        // FIXME: set monitor handle
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_window_level(WindowLevel::AlwaysOnTop)
                    .with_transparent(true)
                    .with_fullscreen(Some(Fullscreen::Borderless(None)))
                    .with_title("Dwawing Pwogwam"),
            )?,
        );
        window.set_cursor_hittest(false)?;

        let surface = self
            .instance
            .create_surface(SurfaceTarget::from(window.clone()))?;
        let res = window.inner_size();
        let gpu = Gpu::new(&self.instance, &surface, res.width, res.height)?;

        let size = window.inner_size();
        log::debug!(
            "creating canvas at {}x{}, format={:?}",
            size.width,
            size.height,
            gpu.format
        );
        let canvas = Drawable::empty(&gpu, size.width, size.height);

        let brush = Drawable::from_texture(
            &gpu,
            gpu.device.create_texture_with_data(
                &gpu.queue,
                &TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: 8,
                        height: 8,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                &[0xff; 8 * 8],
            ),
        );
        let cursor_draw = Drawable::from_texture(
            &gpu,
            gpu.device.create_texture_with_data(
                &gpu.queue,
                &TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: 8,
                        height: 8,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                &[0xff; 8 * 8],
            ),
        );
        Ok(Win {
            window,
            surface,
            gpu,
            canvas,
            brush,
            cursor_draw,
            cursor_pos: None,
            stroke: Stroke::new(4.0),
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
            _ => {
                // TODO
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

    /// Schedules the queued impressions to be drawn with `brush`.
    fn put_impressions(&mut self, gpu: &Gpu, brush: &mut Drawable) {
        brush.set_instances(gpu, &self.impressions);
        self.impressions.clear();
    }
}

#[derive(Clone, Copy, NoUninit)]
#[repr(C)]
struct Uniforms {
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

struct Pass<'a> {
    gpu: &'a Gpu,
    pass: RenderPass<'a>,
    render_target_size: Vec2u,
}

impl<'a> Pass<'a> {
    fn new(
        gpu: &'a Gpu,
        enc: &'a mut CommandEncoder,
        target: &Texture,
        clear: Option<Color>,
    ) -> Self {
        let pass = enc.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target.create_view(&Default::default()),
                resolve_target: None,
                ops: Operations {
                    load: if let Some(clear) = clear {
                        LoadOp::Clear(clear)
                    } else {
                        LoadOp::Load
                    },
                    ..Default::default()
                },
            })],
            ..Default::default()
        });

        Self {
            gpu,
            pass,
            render_target_size: vec2(target.width(), target.height()),
        }
    }
}

struct Drawable {
    texture: Texture,
    uniform_buf: Buffer,
    instance_buf: Buffer,
    texture_bg: BindGroup,
    uniforms_bg: BindGroup,
    instances_bg: BindGroup,
    instance_count: u32,
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
        let uniform_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: None,
            size: mem::size_of::<Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instance_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: None,
            size: mem::size_of::<Instance>() as u64, // 1 instance preallocated
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &gpu.texture_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&texture.create_view(&Default::default())),
            }],
        });
        let uniforms_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &gpu.uniforms_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(uniform_buf.as_entire_buffer_binding()),
            }],
        });
        let instances_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &gpu.instances_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(instance_buf.as_entire_buffer_binding()),
            }],
        });

        Self {
            texture,
            uniform_buf,
            instance_buf,
            texture_bg,
            uniforms_bg,
            instances_bg,
            instance_count: 0,
        }
    }

    fn clear(&mut self) {
        self.instance_count = 0;
    }

    fn set_position(&mut self, gpu: &Gpu, pos: Vec2f) {
        self.set_instances(
            gpu,
            &[Instance {
                pos,
                opacity: 1.0,
                _padding: 0.0,
            }],
        );
    }

    fn set_instances(&mut self, gpu: &Gpu, instances: &[Instance]) {
        let size = (mem::size_of::<Instance>() * instances.len()) as u64;
        if self.instance_buf.size() < size {
            self.instance_buf = gpu.device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &gpu.instances_bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(self.instance_buf.as_entire_buffer_binding()),
                }],
            });
        }
        gpu.queue
            .write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(instances));
        self.instance_count = instances.len() as u32;
    }

    fn draw(&self, p: &mut Pass<'_>) {
        // FIXME: footgun! we're overwriting the uniforms here, but that can only be done once per submission and `Drawable`
        // (on second thought the same thing applies to the instance buffer modified above)
        let uniforms = Uniforms {
            render_target_size: p.render_target_size,
        };
        p.gpu
            .queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        p.pass.set_pipeline(&p.gpu.render_pipeline);
        p.pass.set_bind_group(0, &p.gpu.sampler_bg, &[]);
        p.pass.set_bind_group(1, &self.texture_bg, &[]);
        p.pass.set_bind_group(2, &self.uniforms_bg, &[]);
        p.pass.set_bind_group(3, &self.instances_bg, &[]);
        p.pass.draw(0..4, 0..self.instance_count);
    }
}
