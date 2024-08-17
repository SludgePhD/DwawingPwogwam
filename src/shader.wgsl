// "All-purpose" shader.
//
// Draws instances of a texture onto the render target. Used both for drawing onto the canvas,
// and also for displaying the canvas on the window surface.

struct Uniforms {
    /// Dimensions of the render target (in pixels).
    render_target_size: vec2u,
}

/// Describes a single brush impression (or instance of the bound texture).
struct Instance {
    /// Impression center position, in fractional render target pixels (top-left is 0,0).
    pos: vec2f,
    /// Impression opacity (flow), depends on pen pressure for draw ops.
    opacity: f32,
}

@group(0) @binding(0)
var samp: sampler;

@group(1) @binding(0)
var texture: texture_2d<f32>;

@group(2) @binding(0)
var<uniform> uniforms: Uniforms;

@group(3) @binding(0)
var<storage, read> instances: array<Instance>;

struct VertexOutput {
    @builtin(position)
    position: vec4f,
    @location(0)
    uv: vec2f,
    @location(1)
    opacity: f32,
}

const POSITIONS = array(
    vec2(-0.5,  0.5), // top left
    vec2( 0.5,  0.5), // top right
    vec2(-0.5, -0.5), // bottom left
    vec2( 0.5, -0.5), // bottom right
);

// This is invoked for 4 vertices (triangle strip) per instance/brush impression.
@vertex
fn vertex(
    @builtin(vertex_index) v: u32,
    @builtin(instance_index) i: u32,
) -> VertexOutput {
    let size = vec2f(textureDimensions(texture));

    var positions = POSITIONS;
    let vert_pos = positions[v];
    let pix_pos = instances[i].pos + vert_pos * size;
    let fract_pos = pix_pos / vec2f(uniforms.render_target_size);  // 0..1
    let clip_pos = (fract_pos - vec2(0.5)) * vec2(2.0, -2.0);      // -1..1

    var out: VertexOutput;
    out.position = vec4(clip_pos, 0.0, 1.0);
    out.uv = vert_pos + vec2(0.5);
    out.opacity = instances[i].opacity;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(texture, samp, in.uv) * in.opacity;
}
