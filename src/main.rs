use std::{env, mem};

use anyhow::bail;
use app::App;
use config::Config;

mod app;
mod cmd;
mod config;
mod input;
mod math;

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_module(env!("CARGO_CRATE_NAME"), log::LevelFilter::Debug)
        .parse_default_env()
        .init();

    let mut config = match &*env::args_os().skip(1).collect::<Vec<_>>() {
        [path] => Config::load(path)?,
        _ => {
            bail!("usage: {} <config.toml>", env!("CARGO_PKG_NAME"));
        }
    };

    let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    input::spawn(mem::take(&mut config.devices), move |cmd| {
        drop(proxy.send_event(cmd))
    });

    let mut app = App::new(config)?;
    Ok(event_loop.run_app(&mut app)?)
}
