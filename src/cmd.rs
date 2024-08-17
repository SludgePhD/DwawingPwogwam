use crate::math::Vec2f;

#[derive(Debug)]
pub enum Cmd {
    Clear,

    SetTool {
        tool: Tool,
    },

    PenMove {
        /// Pen position in the tablet's coordinate system, from 0-1.
        position: Vec2f,
        /// Physical aspect ratio of the tablet.
        aspect_ratio: f32,
        /// Pressure from 0-1.
        pressure: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum Tool {
    Draw,
    Erase,
    /// Moves the canvas across the screen while the pen touches the surface.
    MoveCanvas,
}
