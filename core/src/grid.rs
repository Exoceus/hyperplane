//! Rectangular sampling grids used to paint decision regions, loss surfaces
//! and contour plots.

/// Samples `[xmin, xmax] × [ymin, ymax]` at the centers of an `nx × ny`
/// lattice of cells. Index `(i, j)` is column `i` (left → right) and row `j`
/// (top → bottom), stored at `j * nx + i`.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Grid {
    pub fn new(xmin: f64, xmax: f64, ymin: f64, ymax: f64, nx: usize, ny: usize) -> Self {
        Grid { xmin, xmax, ymin, ymax, nx: nx.max(1), ny: ny.max(1) }
    }

    #[inline]
    pub fn x(&self, i: usize) -> f64 {
        self.xmin + (i as f64 + 0.5) * (self.xmax - self.xmin) / self.nx as f64
    }

    #[inline]
    pub fn y(&self, j: usize) -> f64 {
        self.ymax - (j as f64 + 0.5) * (self.ymax - self.ymin) / self.ny as f64
    }

    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Evaluate `f(x, y)` at every sample.
    pub fn map(&self, mut f: impl FnMut(f64, f64) -> f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.len());
        for j in 0..self.ny {
            let y = self.y(j);
            for i in 0..self.nx {
                out.push(f(self.x(i), y));
            }
        }
        out
    }
}
