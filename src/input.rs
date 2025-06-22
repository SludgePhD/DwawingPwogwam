use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    process,
    sync::Arc,
    thread,
};

use anyhow::{bail, Context};
use evdevil::{
    event::{Abs, EventKind, KeyState},
    hotplug, AbsInfo, Evdev,
};

use crate::{
    cmd::{Cmd, Tool},
    config::{self, CommandVerb},
};

pub fn spawn(devices: Vec<config::Device>, send_cmd: impl Fn(Cmd) + Send + Sync + 'static) {
    thread::spawn(move || wrap(move || input_main_loop(devices, Arc::new(send_cmd))));
}

fn wrap(f: impl FnOnce() -> anyhow::Result<()>) {
    let res = catch_unwind(AssertUnwindSafe(f));
    match res {
        Ok(Ok(())) => {
            eprintln!("error: input thread exited unexpectedly");
            process::exit(1);
        }
        Ok(Err(e)) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
        Err(_panic) => {
            // Panic message and backtrace are printed by the default panic hook.
            eprintln!("input thread panicked, exiting");
            process::exit(101);
        }
    }
}

fn input_main_loop(
    mut devices: Vec<config::Device>,
    send_cmd: Arc<dyn Fn(Cmd) + Send + Sync>,
) -> anyhow::Result<()> {
    let mut handles = Vec::new();
    for res in hotplug::enumerate()? {
        let device = match res {
            Ok(dev) => dev,
            Err(e) => {
                log::warn!("failed to open a device: {e}");
                continue;
            }
        };

        let name = match device.name() {
            Ok(name) => name,
            Err(e) => {
                log::warn!(
                    "failed to fetch name of '{}': {e}",
                    device.path().unwrap().display()
                );
                continue;
            }
        };

        let Some(i) = devices.iter().position(|dev| dev.name == name) else {
            continue;
        };
        let dev = devices.swap_remove(i);

        log::info!(
            "found matching input device at `{}`: {name}",
            device.path().unwrap().display()
        );

        if dev.tablet {
            let axes = device.supported_abs_axes()?;
            if !axes.contains(Abs::X) || !axes.contains(Abs::Y) {
                bail!("input device '{name}' is configured as the stylus but does not support X/Y absolute axes")
            }
        }

        let keys = device.supported_keys()?;
        for key in dev.bind.keys() {
            if !keys.contains(*key) {
                bail!("input device '{name}' input {:?} is bounds to an action, but the device does not have that key", key)
            }
        }

        if dev.grab {
            log::info!("grabbing '{name}'");
            device.grab()?;
        }

        let send_cmd = send_cmd.clone();
        handles.push(thread::spawn(move || {
            wrap(move || device_main(device, dev, send_cmd))
        }));
    }

    for handle in handles {
        handle.join().ok();
    }

    Ok(())
}

struct TabletState {
    x_info: AbsInfo,
    x_range: i32,
    y_info: AbsInfo,
    y_range: i32,
    pressure_info: AbsInfo,
    pressure_range: i32,
    aspect_ratio: f32,

    x: f32,
    y: f32,
    pressure: f32,
}

impl TabletState {
    fn new(device: &Evdev, config: &config::Device) -> anyhow::Result<Self> {
        let x_info = device.abs_info(Abs::X)?;
        let y_info = device.abs_info(Abs::Y)?;
        let pressure_info = device.abs_info(Abs::PRESSURE)?;
        let x_range = x_info.maximum() - x_info.minimum();
        let y_range = y_info.maximum() - y_info.minimum();
        let pressure_range = pressure_info.maximum() - pressure_info.minimum();
        let aspect_ratio = x_range as f32 / y_range as f32;
        log::debug!(
            "ranges; x={} y={} pressure={} ratio={} ratio_override={:?}",
            x_range,
            y_range,
            pressure_range,
            aspect_ratio,
            config.ratio,
        );
        let aspect_ratio = config.ratio.unwrap_or(aspect_ratio);

        Ok(Self {
            x_info,
            x_range,
            y_info,
            y_range,
            pressure_info,
            pressure_range,
            aspect_ratio,

            x: 0.0,
            y: 0.0,
            pressure: 0.0,
        })
    }
}

fn device_main(
    device: Evdev,
    config: config::Device,
    send_cmd: Arc<dyn Fn(Cmd) + Send + Sync>,
) -> anyhow::Result<()> {
    let mut tablet_state = if config.tablet {
        Some(TabletState::new(&device, &config).context(
            "couldn't open tablet; ensure it has an ABX_X, ABS_Y, and ABS_PRESSURE axis",
        )?)
    } else {
        None
    };

    let mut reader = device.into_reader()?;
    loop {
        let report = reader.next_report()?;

        // Update state based on all events received in this report.
        for event in report {
            let Some(ev) = event.kind() else { continue };

            match ev {
                EventKind::Abs(ev) => {
                    if let Some(tab) = &mut tablet_state {
                        if ev.abs() == Abs::X {
                            tab.x = (ev.value() - tab.x_info.minimum()) as f32 / tab.x_range as f32;
                        } else if ev.abs() == Abs::Y {
                            tab.y = (ev.value() - tab.y_info.minimum()) as f32 / tab.y_range as f32;
                        } else if ev.abs() == Abs::PRESSURE {
                            tab.pressure = (ev.value() - tab.pressure_info.minimum()) as f32
                                / tab.pressure_range as f32;
                        } else {
                            continue;
                        }
                    }
                }
                EventKind::Key(ev) => {
                    if ev.state() == KeyState::REPEAT {
                        continue;
                    }

                    let pressed = ev.state() == KeyState::PRESSED;
                    // FIXME: Doesn't really handle having multiple tools active at once.
                    // The last tool should override the first, and when released should go back to
                    // the previous tool (if still pressed), so I guess there should be a stack.
                    if let Some(verb) = config.bind.get(&ev.key()) {
                        match verb {
                            CommandVerb::ToolEraser => {
                                if pressed {
                                    send_cmd(Cmd::SetTool { tool: Tool::Erase });
                                } else {
                                    send_cmd(Cmd::SetTool { tool: Tool::Draw });
                                }
                            }
                            CommandVerb::ToolMove => {
                                if pressed {
                                    send_cmd(Cmd::SetTool {
                                        tool: Tool::MoveCanvas,
                                    });
                                } else {
                                    send_cmd(Cmd::SetTool { tool: Tool::Draw });
                                }
                            }
                            CommandVerb::Clear => {
                                if pressed {
                                    send_cmd(Cmd::Clear);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(tab) = &tablet_state {
            // Send new pen coordinates for every incoming report.
            send_cmd(Cmd::PenMove {
                position: [tab.x, tab.y].into(),
                aspect_ratio: tab.aspect_ratio,
                pressure: tab.pressure,
            });
        }
    }
}
