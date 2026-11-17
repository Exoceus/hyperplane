import type {Painter} from "./painter";
import {C, hexToRgb, type RGB} from "./theme";

export interface Camera {
    /** Rotation around the vertical axis (radians). */
    yaw: number;
    /** Elevation: 0 = side view, π/2 = top view. */
    pitch: number;
    /** Zoom factor (1 = default). */
    zoom: number;
}

export type Vec3 = [number, number, number];

interface Prim {
    depth: number;
    draw: () => void;
}

const LIGHT: Vec3 = (() => {
    const v: Vec3 = [-0.45, -0.35, 0.82];
    const n = Math.hypot(...v);
    return [v[0] / n, v[1] / n, v[2] / n];
})();

const PERSPECTIVE = 4.5;

/** World → [u, v, depth] for a camera (larger depth = farther). */
export function projector(cam: Camera): (x: number, y: number, z: number) => Vec3 {
    const cy = Math.cos(cam.yaw);
    const sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch);
    const sp = Math.sin(cam.pitch);
    return (x, y, z) => {
        const x1 = cy * x - sy * y;
        const y1 = sy * x + cy * y;
        const v = z * cp + y1 * sp;
        const d = y1 * cp - z * sp;
        const f = (PERSPECTIVE / (PERSPECTIVE + d)) * cam.zoom;
        return [x1 * f, v * f - 0.05, d];
    };
}

/**
 * A tiny 3D renderer on top of {@link Painter}: orbit camera, mild
 * perspective, and a painter's-algorithm queue so surfaces, lines and points
 * occlude each other correctly. World space is z-up, roughly within [-1, 1]³.
 */
export class Scene3D {
    readonly p: Painter;
    readonly cam: Camera;
    private queue: Prim[] = [];
    readonly project: (x: number, y: number, z: number) => Vec3;
    private persp = PERSPECTIVE;

    constructor(p: Painter, cam: Camera) {
        this.p = p;
        this.cam = cam;
        this.project = projector(cam);
    }

    /** Lambert shading of a base color for a surface normal. */
    shade(rgb: RGB, normal: Vec3, ambient = 0.55): string {
        const n = Math.hypot(...normal) || 1;
        const lambert = Math.abs((normal[0] * LIGHT[0] + normal[1] * LIGHT[1] + normal[2] * LIGHT[2]) / n);
        const k = ambient + (1 - ambient) * lambert;
        return `rgb(${Math.min(255, rgb[0] * k) | 0},${Math.min(255, rgb[1] * k) | 0},${Math.min(255, rgb[2] * k) | 0})`;
    }

    quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, fill: string, opts: {alpha?: number; stroke?: string; width?: number} = {}) {
        const pa = this.project(...a);
        const pb = this.project(...b);
        const pc = this.project(...c);
        const pd = this.project(...d);
        const depth = (pa[2] + pb[2] + pc[2] + pd[2]) / 4;
        this.queue.push({
            depth,
            draw: () => {
                this.p.polyline([pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], pd[0], pd[1]], {
                    close: true,
                    fill,
                    fillAlpha: opts.alpha ?? 1,
                    color: opts.stroke ?? fill,
                    width: opts.width ?? 0.6,
                    alpha: opts.alpha ?? 1,
                });
            },
        });
    }

    line(a: Vec3, b: Vec3, color: string, width = 1.5, opts: {dash?: number[]; alpha?: number; bias?: number} = {}) {
        const pa = this.project(...a);
        const pb = this.project(...b);
        this.queue.push({
            depth: (pa[2] + pb[2]) / 2 - (opts.bias ?? 0),
            draw: () => this.p.line(pa[0], pa[1], pb[0], pb[1], {color, width, dash: opts.dash, alpha: opts.alpha}),
        });
    }

    point(at: Vec3, color: string, r = 4.5, opts: {ring?: string; bias?: number; alpha?: number} = {}) {
        const pp = this.project(...at);
        this.queue.push({
            depth: pp[2] - (opts.bias ?? 0.02),
            draw: () => this.p.point(pp[0], pp[1], color, {r: r * (this.persp / (this.persp + pp[2])), ring: opts.ring, alpha: opts.alpha}),
        });
    }

    label(at: Vec3, text: string, color: string = C.text2, size = 12) {
        const pp = this.project(...at);
        this.queue.push({depth: -100, draw: () => this.p.text(text, pp[0], pp[1], {color, size, align: "center", baseline: "middle"})});
    }

    /** Draw an arrow (shaft + head) in 3D. */
    arrow(a: Vec3, b: Vec3, color: string, width = 3) {
        const pa = this.project(...a);
        const pb = this.project(...b);
        this.queue.push({depth: Math.min(pa[2], pb[2]) - 0.05, draw: () => this.p.arrow(pa[0], pa[1], pb[0], pb[1], {color, width, head: 11})});
    }

    /** Floor grid on z = `z` spanning [-extent, extent]². */
    floor(z = 0, extent = 1, step = 0.25) {
        for (let t = -extent; t <= extent + 1e-9; t += step) {
            const major = Math.abs(t) < 1e-9;
            const col = major ? "rgba(148,163,184,0.35)" : "rgba(148,163,184,0.12)";
            this.line([t, -extent, z], [t, extent, z], col, 1, {bias: -5});
            this.line([-extent, t, z], [extent, t, z], col, 1, {bias: -5});
        }
    }

    /** Render everything queued, far to near. */
    flush() {
        this.queue.sort((a, b) => b.depth - a.depth);
        for (const prim of this.queue) prim.draw();
        this.queue = [];
    }
}

/**
 * Shaded height-field mesh. `heights` is `n × n` (row-major, row 0 at
 * y = +1), mapped onto [-1, 1]² in x/y; `color(h)` gives the base color.
 */
export function drawSurface(s: Scene3D, heights: ArrayLike<number>, n: number, color: (h: number) => RGB, opts: {alpha?: number; wire?: number} = {}) {
    const at = (i: number, j: number): Vec3 => [-1 + (2 * i) / (n - 1), 1 - (2 * j) / (n - 1), heights[j * n + i]];
    for (let j = 0; j < n - 1; j++)
        for (let i = 0; i < n - 1; i++) {
            const a = at(i, j);
            const b = at(i + 1, j);
            const c = at(i + 1, j + 1);
            const d = at(i, j + 1);
            const u: Vec3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            const v: Vec3 = [b[0] - d[0], b[1] - d[1], b[2] - d[2]];
            const normal: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            const h = (a[2] + b[2] + c[2] + d[2]) / 4;
            const fill = s.shade(color(h), normal);
            const wire = opts.wire && (i % opts.wire === 0 || j % opts.wire === 0);
            s.quad(a, b, c, d, fill, {alpha: opts.alpha, stroke: wire ? "rgba(255,255,255,0.10)" : fill, width: wire ? 0.8 : 0.6});
        }
}

/** Index of the mesh vertex whose projection is closest to (u, v), or -1. */
export function pickVertex(cam: Camera, heights: ArrayLike<number>, n: number, u: number, v: number): number {
    const project = projector(cam);
    let best = -1;
    let bestD = Infinity;
    for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
            const [pu, pv, depth] = project(-1 + (2 * i) / (n - 1), 1 - (2 * j) / (n - 1), heights[j * n + i]);
            // Prefer nearer vertices when several project to the same spot.
            const d = (pu - u) ** 2 + (pv - v) ** 2 + depth * 1e-4;
            if (d < bestD) {
                bestD = d;
                best = j * n + i;
            }
        }
    return bestD < 0.01 ? best : -1;
}

export const rgbOf = hexToRgb;
