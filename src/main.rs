use slsh_engine::main_loop::MainLoop;
use slsh_engine::window::Resolution;

fn main() {
    let mut main_loop = MainLoop::new(&Resolution::Windowed(1024, 768), "slsh");

    main_loop.run();
}
