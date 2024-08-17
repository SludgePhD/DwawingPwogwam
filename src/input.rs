use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    process,
    sync::Arc,
    thread,
};

use anyhow::bail;
use evdev::AbsoluteAxisType;

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
    for (path, mut device) in evdev::enumerate() {
        let Some(name) = device.name() else { continue };

        let Some(i) = devices.iter().position(|dev| dev.name == name) else {
            continue;
        };
        let dev = devices.swap_remove(i);

        log::info!(
            "found matching input device at `{}`: {name}",
            path.display()
        );

        if dev.tablet {
            match device.supported_absolute_axes() {
                Some(axes) if axes.contains(AbsoluteAxisType::ABS_X) && axes.contains(AbsoluteAxisType::ABS_Y) => {},
                _ => bail!("input device '{name}' is configured as the stylus but does not support X/Y absolute axes"),
            }
        }

        for key in dev.bind.keys() {
            match device.supported_keys() {
                Some(keys) if keys.contains(*key) => {},
                _ => bail!("input device '{name}' input {key:?} is bounds to an action, but the device does not have that key"),
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

fn device_main(
    mut device: evdev::Device,
    config: config::Device,
    send_cmd: Arc<dyn Fn(Cmd) + Send + Sync>,
) -> anyhow::Result<()> {
    let abs = device.get_abs_state()?;
    let x_info = abs[usize::from(AbsoluteAxisType::ABS_X.0)];
    let y_info = abs[usize::from(AbsoluteAxisType::ABS_Y.0)];
    let pressure_info = abs[usize::from(AbsoluteAxisType::ABS_PRESSURE.0)];
    let x_amt = x_info.maximum - x_info.minimum;
    let y_amt = y_info.maximum - y_info.minimum;
    let pressure_amt = pressure_info.maximum - pressure_info.minimum;
    let aspect_ratio = x_amt as f32 / y_amt as f32;
    if config.tablet {
        log::debug!(
            "ranges; x={} y={} ratio={} ratio_override={:?}",
            x_amt,
            y_amt,
            aspect_ratio,
            config.ratio
        );
    }
    let aspect_ratio = config.ratio.unwrap_or(aspect_ratio);

    let mut x = 0.0;
    let mut y = 0.0;
    let mut pressure = 0.0;
    loop {
        let events = device.fetch_events()?; // (blocks for new events)

        // Update state based on all events received in this report.
        for event in events {
            match event.kind() {
                evdev::InputEventKind::AbsAxis(axis) if config.tablet => {
                    if axis == AbsoluteAxisType::ABS_X {
                        x = (event.value() - x_info.minimum) as f32 / x_amt as f32;
                    } else if axis == AbsoluteAxisType::ABS_Y {
                        y = (event.value() - y_info.minimum) as f32 / y_amt as f32;
                    } else if axis == AbsoluteAxisType::ABS_PRESSURE {
                        pressure =
                            (event.value() - pressure_info.minimum) as f32 / pressure_amt as f32;
                    } else {
                        continue;
                    }
                }
                evdev::InputEventKind::Key(key) => {
                    if event.value() == 2 {
                        // Key repeat.
                        continue;
                    }

                    let pressed = event.value() == 1;
                    // NB: doesn't really handle having multiple tools active at once.
                    // the last tool should override the first, and when released should go back to the previous tool (if still pressed).
                    if let Some(verb) = config.bind.get(&key) {
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

        // Send new pen coordinates for every incoming report.
        send_cmd(Cmd::PenMove {
            position: [x, y].into(),
            aspect_ratio,
            pressure,
        });
    }
}
