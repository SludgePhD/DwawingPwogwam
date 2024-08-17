use std::{collections::HashMap, fs, path::Path};

use anyhow::bail;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub monitor: Option<String>,
    #[serde(rename = "device")]
    pub devices: Vec<Device>,
}

impl Config {
    pub fn load<A: AsRef<Path>>(path: A) -> anyhow::Result<Self> {
        Self::load_impl(path.as_ref())
    }

    fn load_impl(path: &Path) -> anyhow::Result<Self> {
        let contents = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&contents)?;

        // Validate configuration.
        // - There should be exactly 1 device with `tablet = true`.

        let tablet_devs = config.devices.iter().filter(|dev| dev.tablet).count();
        if tablet_devs != 1 {
            bail!(
                "there must be exactly one [[device]] with `tablet = true` (found {tablet_devs})"
            );
        }

        Ok(config)
    }
}

#[derive(Deserialize)]
pub struct Device {
    pub name: String,
    #[serde(default)]
    pub grab: bool,
    #[serde(default)]
    pub tablet: bool,
    pub ratio: Option<f32>,
    #[serde(default)]
    pub bind: HashMap<evdev::Key, CommandVerb>,
}

#[derive(Debug, Deserialize)]
pub enum CommandVerb {
    #[serde(rename = "TOOL_ERASER")]
    ToolEraser,
    #[serde(rename = "TOOL_MOVE")]
    ToolMove,
    #[serde(rename = "CLEAR")]
    Clear,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_config() {
        Config::load("config.example.toml").unwrap();
    }
}
