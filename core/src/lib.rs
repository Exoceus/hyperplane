//! Hyperplane numerical core.
//!
//! Every algorithm visualized on the site lives here and is compiled to
//! WebAssembly. The JS boundary deliberately uses flat `f64` buffers
//! (`Float64Array` on the JS side) with documented strides instead of
//! serialized objects: it is faster, allocation-light and keeps the
//! generated bindings tiny.
//!
//! Conventions shared by all modules:
//! - A 2D *classification* dataset is a flat buffer with stride 3:
//!   `[x, y, class, x, y, class, ...]` where `class` is `0, 1, 2, ...`.
//! - A 2D *point* or *regression* dataset has stride 2: `[x, y, ...]`.
//! - A *grid* samples the rectangle `[xmin, xmax] × [ymin, ymax]` at cell
//!   centers, row-major, with row 0 at the **top** (`ymax`) so that it maps
//!   straight onto an image (see [`grid::Grid`]).

pub mod contour;
pub mod datasets;
pub mod grid;
pub mod kmeans;
pub mod knn;
pub mod linreg;
pub mod logreg;
pub mod mlp;
pub mod optim;
pub mod pca;
pub mod perceptron;
pub mod rng;
