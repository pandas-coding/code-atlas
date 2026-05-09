struct Point {
    x: f64,
    y: f64,
}

fn new_point(x: f64, y: f64) -> Point {
    Point { x, y }
}

enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
}

trait Drawable {
    fn draw(&self);
}

impl Point {
    fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

const MAX_SIZE: usize = 1024;

type Result<T> = std::result::Result<T, Error>;

mod utils;
