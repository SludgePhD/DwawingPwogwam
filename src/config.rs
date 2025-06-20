use std::{collections::HashMap, fs, path::Path};

use anyhow::bail;
use serde::{de::Visitor, Deserialize};

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

#[derive(PartialEq, Eq, Hash)]
pub struct Key(pub(crate) evdevil::event::Key);

impl<'a> Deserialize<'a> for Key {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        struct FromStrVisitor;

        impl<'de> Visitor<'de> for FromStrVisitor {
            type Value = Key;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("evdev key name")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Key(v.parse().map_err(|_| {
                    E::custom(format_args!("invalid evdev key name '{v}'"))
                })?))
            }
        }

        deserializer.deserialize_str(FromStrVisitor)
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
    pub bind: HashMap<Key, CommandVerb>,
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
