import {BG_RGB, C, type RGB} from "./theme";

export interface View {
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
}

export interface Stroke {
    color?: string;
    width?: number;
    dash?: number[];
    alpha?: number;
    cap?: CanvasLineCap;
}

export interface TextStyle {
    color?: string;
    size?: number;
    align?: CanvasTextAlign;
    baseline?: CanvasTextBaseline;
    font?: "ui" | "mono" | "serif";
    weight?: number;
    dx?: number;
    dy?: number;
    halo?: boolean;
}

const FONTS = {
    ui: '"Inter", system-ui, sans-serif',
    mono: '"JetBrains Mono", ui-monospace, monospace',
    serif: '"Source Serif 4", Georgia, serif',
};

/**
 * Thin wrapper around a 2D canvas context that speaks *data* coordinates.
 * Every figure draws through one of these, so the visual language (point
 * style, arrows, grids, heatmaps) stays consistent across lessons.
 */
export class Painter {
    readonly ctx: CanvasRenderingContext2D;
    readonly width: number;
    readonly height: number;
    readonly view: View;

    constructor(ctx: CanvasRenderingContext2D, width: number, height: number, view: View) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.view = view;
    }

    /** Pixels per data unit, horizontally / vertically. */
    get kx() {
        return this.width / (this.view.xmax - this.view.xmin);
    }
    get ky() {
        return this.height / (this.view.ymax - this.view.ymin);
    }

    sx(x: number) {
        return (x - this.view.xmin) * this.kx;
    }
    sy(y: number) {
        return (this.view.ymax - y) * this.ky;
    }
    ix(px: number) {
        return this.view.xmin + px / this.kx;
    }
    iy(py: number) {
        return this.view.ymax - py / this.ky;
    }

    clear(color: string = C.canvas) {
        this.ctx.fillStyle = color;
        this.ctx.fillRect(0, 0, this.width, this.height);
    }

    private applyStroke(s: Stroke) {
        const ctx = this.ctx;
        ctx.strokeStyle = s.color ?? C.text;
        ctx.lineWidth = s.width ?? 1.5;
        ctx.setLineDash(s.dash ?? []);
        ctx.globalAlpha = s.alpha ?? 1;
        ctx.lineCap = s.cap ?? "round";
        ctx.lineJoin = "round";
    }

    private reset() {
        this.ctx.globalAlpha = 1;
        this.ctx.setLineDash([]);
    }

    /** Background lattice with lines every `step` units, emphasizing the axes. */
    grid(step = 0.25, opts: {axes?: boolean; majorEvery?: number} = {}) {
        const {ctx, view} = this;
        const majorEvery = opts.majorEvery ?? 4;
        ctx.lineWidth = 1;
        const x0 = Math.ceil(view.xmin / step);
        const x1 = Math.floor(view.xmax / step);
        for (let i = x0; i <= x1; i++) {
            const px = Math.round(this.sx(i * step)) + 0.5;
            ctx.strokeStyle = i % majorEvery === 0 ? C.gridMajor : C.grid;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, this.height);
            ctx.stroke();
        }
        const y0 = Math.ceil(view.ymin / step);
        const y1 = Math.floor(view.ymax / step);
        for (let j = y0; j <= y1; j++) {
            const py = Math.round(this.sy(j * step)) + 0.5;
            ctx.strokeStyle = j % majorEvery === 0 ? C.gridMajor : C.grid;
            ctx.beginPath();
            ctx.moveTo(0, py);
            ctx.lineTo(this.width, py);
            ctx.stroke();
        }
        if (opts.axes !== false) this.axes();
    }

    axes(color: string = C.axis) {
        const {view} = this;
        if (view.xmin <= 0 && view.xmax >= 0) this.line(0, view.ymin, 0, view.ymax, {color, width: 1.2});
        if (view.ymin <= 0 && view.ymax >= 0) this.line(view.xmin, 0, view.xmax, 0, {color, width: 1.2});
    }

    /** Numeric tick labels along the bottom and left edges. */
    ticks(stepX: number, stepY: number = stepX, opts: {format?: (v: number) => string; xLabel?: string; yLabel?: string} = {}) {
        const fmt = opts.format ?? ((v: number) => (Math.abs(v) < 1e-9 ? "0" : String(+v.toFixed(2))));
        const {view} = this;
        const style: TextStyle = {color: C.muted, size: 10.5, font: "mono"};
        for (let i = Math.ceil(view.xmin / stepX); i * stepX <= view.xmax + 1e-9; i++) {
            const v = i * stepX;
            if (Math.abs(this.sx(v)) < 12 || Math.abs(this.sx(v) - this.width) < 12) continue;
            this.textPx(fmt(v), this.sx(v), this.height - 6, {...style, align: "center", baseline: "bottom"});
        }
        for (let j = Math.ceil(view.ymin / stepY); j * stepY <= view.ymax + 1e-9; j++) {
            const v = j * stepY;
            if (Math.abs(this.sy(v)) < 12 || Math.abs(this.sy(v) - this.height) < 12) continue;
            this.textPx(fmt(v), 6, this.sy(v), {...style, align: "left", baseline: "middle"});
        }
        if (opts.xLabel)
            this.textPx(opts.xLabel, this.width - 8, this.height - 22, {...style, color: C.text2, align: "right", font: "serif", size: 14});
        if (opts.yLabel) this.textPx(opts.yLabel, 26, 14, {...style, color: C.text2, align: "left", font: "serif", size: 14});
    }

    line(x1: number, y1: number, x2: number, y2: number, s: Stroke = {}) {
        const ctx = this.ctx;
        this.applyStroke(s);
        ctx.beginPath();
        ctx.moveTo(this.sx(x1), this.sy(y1));
        ctx.lineTo(this.sx(x2), this.sy(y2));
        ctx.stroke();
        this.reset();
    }

    /** Flat `[x, y, x, y, …]` polyline. */
    polyline(pts: ArrayLike<number>, s: Stroke & {close?: boolean; fill?: string; fillAlpha?: number} = {}) {
        if (pts.length < 4) return;
        const ctx = this.ctx;
        ctx.beginPath();
        ctx.moveTo(this.sx(pts[0]), this.sy(pts[1]));
        for (let i = 2; i < pts.length; i += 2) ctx.lineTo(this.sx(pts[i]), this.sy(pts[i + 1]));
        if (s.close) ctx.closePath();
        if (s.fill) {
            ctx.globalAlpha = s.fillAlpha ?? 1;
            ctx.fillStyle = s.fill;
            ctx.fill();
            ctx.globalAlpha = 1;
        }
        if (s.color !== "none") {
            this.applyStroke(s);
            ctx.stroke();
        }
        this.reset();
    }

    /** Circle with radius in *pixels*. */
    circle(x: number, y: number, rPx: number, o: {fill?: string; stroke?: string; width?: number; alpha?: number; dash?: number[]} = {}) {
        const ctx = this.ctx;
        ctx.globalAlpha = o.alpha ?? 1;
        ctx.beginPath();
        ctx.arc(this.sx(x), this.sy(y), Math.max(0, rPx), 0, Math.PI * 2);
        if (o.fill) {
            ctx.fillStyle = o.fill;
            ctx.fill();
        }
        if (o.stroke) {
            ctx.strokeStyle = o.stroke;
            ctx.lineWidth = o.width ?? 1.5;
            ctx.setLineDash(o.dash ?? []);
            ctx.stroke();
        }
        this.reset();
    }

    /** The standard data-point glyph: a colored disc with a dark rim. */
    point(x: number, y: number, color: string, o: {r?: number; alpha?: number; ring?: string; ringWidth?: number} = {}) {
        const r = o.r ?? 4.5;
        this.circle(x, y, r + 1.5, {fill: C.canvas, alpha: o.alpha});
        this.circle(x, y, r, {fill: color, alpha: o.alpha});
        if (o.ring) this.circle(x, y, r + 4, {stroke: o.ring, width: o.ringWidth ?? 2, alpha: o.alpha});
    }

    /** Arrow from (x1, y1) to (x2, y2); head size in pixels. */
    arrow(x1: number, y1: number, x2: number, y2: number, s: Stroke & {head?: number} = {}) {
        const ctx = this.ctx;
        const [ax, ay, bx, by] = [this.sx(x1), this.sy(y1), this.sx(x2), this.sy(y2)];
        const len = Math.hypot(bx - ax, by - ay);
        if (len < 0.5) return;
        const head = Math.min(s.head ?? 11, len * 0.6);
        const ux = (bx - ax) / len;
        const uy = (by - ay) / len;
        this.applyStroke(s);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx - ux * head * 0.8, by - uy * head * 0.8);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = s.color ?? C.text;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - ux * head - uy * head * 0.45, by - uy * head + ux * head * 0.45);
        ctx.lineTo(bx - ux * head + uy * head * 0.45, by - uy * head - ux * head * 0.45);
        ctx.closePath();
        ctx.fill();
        this.reset();
    }

    /** Clip the line `a·x + b·y + c = 0` to the view. Returns two endpoints or null. */
    clipLine(a: number, b: number, c: number): [number, number, number, number] | null {
        const {xmin, xmax, ymin, ymax} = this.view;
        const pts: [number, number][] = [];
        if (Math.abs(b) > 1e-12) {
            for (const x of [xmin, xmax]) {
                const y = -(a * x + c) / b;
                if (y >= ymin - 1e-9 && y <= ymax + 1e-9) pts.push([x, y]);
            }
        }
        if (Math.abs(a) > 1e-12) {
            for (const y of [ymin, ymax]) {
                const x = -(b * y + c) / a;
                if (x >= xmin - 1e-9 && x <= xmax + 1e-9) pts.push([x, y]);
            }
        }
        if (pts.length < 2) return null;
        // Pick the two most distant points (handles corner duplicates).
        let best: [number, number, number, number] = [...pts[0], ...pts[1]];
        let bestD = -1;
        for (let i = 0; i < pts.length; i++)
            for (let j = i + 1; j < pts.length; j++) {
                const d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2;
                if (d > bestD) {
                    bestD = d;
                    best = [...pts[i], ...pts[j]];
                }
            }
        return best;
    }

    /** Draw the infinite line `a·x + b·y + c = 0`. */
    infiniteLine(a: number, b: number, c: number, s: Stroke = {}) {
        const seg = this.clipLine(a, b, c);
        if (seg) this.line(seg[0], seg[1], seg[2], seg[3], s);
    }

    /** Fill the half-plane `a·x + b·y + c > 0` (clipped to the view). */
    halfPlane(a: number, b: number, c: number, fill: string, alpha = 1) {
        const {xmin, xmax, ymin, ymax} = this.view;
        const corners: [number, number][] = [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ];
        // Sutherland–Hodgman against a single half-plane.
        const out: number[] = [];
        const f = (p: [number, number]) => a * p[0] + b * p[1] + c;
        for (let i = 0; i < 4; i++) {
            const p = corners[i];
            const q = corners[(i + 1) % 4];
            const fp = f(p);
            const fq = f(q);
            if (fp >= 0) out.push(p[0], p[1]);
            if (fp >= 0 !== fq >= 0) {
                const t = fp / (fp - fq);
                out.push(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t);
            }
        }
        if (out.length >= 6) this.polyline(out, {close: true, fill, fillAlpha: alpha, color: "none"});
    }

    text(str: string, x: number, y: number, st: TextStyle = {}) {
        this.textPx(str, this.sx(x), this.sy(y), st);
    }

    textPx(str: string, px: number, py: number, st: TextStyle = {}) {
        const ctx = this.ctx;
        ctx.font = `${st.weight ?? 500} ${st.size ?? 12}px ${FONTS[st.font ?? "ui"]}`;
        ctx.textAlign = st.align ?? "left";
        ctx.textBaseline = st.baseline ?? "alphabetic";
        const x = px + (st.dx ?? 0);
        const y = py + (st.dy ?? 0);
        if (st.halo !== false) {
            ctx.strokeStyle = C.canvas;
            ctx.lineWidth = 4;
            ctx.globalAlpha = 0.85;
            ctx.lineJoin = "round";
            ctx.strokeText(str, x, y);
            ctx.globalAlpha = 1;
        }
        ctx.fillStyle = st.color ?? C.text;
        ctx.fillText(str, x, y);
    }

    /** Draw a pre-rendered heat image stretched over a rectangle in data coordinates. */
    image(img: CanvasImageSource, rect: View = this.view, smooth = true, alpha = 1) {
        const ctx = this.ctx;
        ctx.imageSmoothingEnabled = smooth;
        ctx.imageSmoothingQuality = "high";
        ctx.globalAlpha = alpha;
        const x = this.sx(rect.xmin);
        const y = this.sy(rect.ymax);
        ctx.drawImage(img, x, y, this.sx(rect.xmax) - x, this.sy(rect.ymin) - y);
        ctx.globalAlpha = 1;
    }

    /** Contour segments (stride 5 from the WASM `contour_segments`). */
    contours(segs: ArrayLike<number>, style: (level: number) => Stroke) {
        const ctx = this.ctx;
        let current = -1;
        for (let i = 0; i < segs.length; i += 5) {
            const level = segs[i + 4];
            if (level !== current) {
                if (current >= 0) ctx.stroke();
                this.applyStroke(style(level));
                ctx.beginPath();
                current = level;
            }
            ctx.moveTo(this.sx(segs[i]), this.sy(segs[i + 1]));
            ctx.lineTo(this.sx(segs[i + 2]), this.sy(segs[i + 3]));
        }
        if (current >= 0) ctx.stroke();
        this.reset();
    }
}

/**
 * Render a grid of values into a small offscreen canvas (one pixel per grid
 * cell) using `color(value, index) → [r, g, b]` (or `[r, g, b, a]`). The
 * result is drawn scaled-up with smoothing by {@link Painter.image}.
 */
export function heatImage(
    values: ArrayLike<number>,
    nx: number,
    ny: number,
    color: (v: number, i: number) => RGB | [number, number, number, number],
    stride = 1
): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext("2d")!;
    const img = ctx.createImageData(nx, ny);
    const n = nx * ny;
    for (let i = 0; i < n; i++) {
        const c = color(values[i * stride], i);
        img.data[4 * i] = c[0];
        img.data[4 * i + 1] = c[1];
        img.data[4 * i + 2] = c[2];
        img.data[4 * i + 3] = c.length > 3 ? (c as number[])[3] : 255;
    }
    ctx.putImageData(img, 0, 0);
    return canvas;
}

/** Blend a class color into the canvas background with strength `t ∈ [0, 1]`. */
export function tint(rgb: RGB, t: number): RGB {
    return [BG_RGB[0] + (rgb[0] - BG_RGB[0]) * t, BG_RGB[1] + (rgb[1] - BG_RGB[1]) * t, BG_RGB[2] + (rgb[2] - BG_RGB[2]) * t];
}
