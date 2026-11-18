import type {Painter} from "../lib/painter";
import {C, classColor, rgba} from "../lib/theme";
import {Plot} from "./Plot";

/** Tiny deterministic PRNG so thumbnails look the same on every visit. */
function prng(seed: number) {
    let s = seed >>> 0;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

function gauss(r: () => number) {
    return Math.sqrt(-2 * Math.log(Math.max(1e-9, r()))) * Math.cos(2 * Math.PI * r());
}

const view = {xmin: -1.2, xmax: 1.2, ymin: -0.75, ymax: 0.75};

/** Ellipse outline as a closed polyline: center, radii and rotation (radians). */
function ellipse(p: Painter, cx: number, cy: number, rx: number, ry: number, rot: number, o: {color: string; width?: number; fill?: string; fillAlpha?: number; dash?: number[]}) {
    const pts: number[] = [];
    for (let i = 0; i <= 72; i++) {
        const t = (i / 72) * Math.PI * 2;
        const x = rx * Math.cos(t);
        const y = ry * Math.sin(t);
        pts.push(cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot));
    }
    p.polyline(pts, {close: true, color: o.color, width: o.width ?? 1.5, fill: o.fill, fillAlpha: o.fillAlpha, dash: o.dash});
}

/** Points scattered around a center with a covariance given by radii and rotation. */
function blob(r: () => number, n: number, cx: number, cy: number, sx: number, sy: number, rot = 0): [number, number][] {
    return Array.from({length: n}, () => {
        const u = gauss(r) * sx;
        const v = gauss(r) * sy;
        return [cx + u * Math.cos(rot) - v * Math.sin(rot), cy + u * Math.sin(rot) + v * Math.cos(rot)];
    });
}

const DRAW: Record<string, (p: Painter) => void> = {
    "linear-regression": (p) => {
        const r = prng(3);
        p.infiniteLine(-0.45, 1, 0.05, {color: C.yellow, width: 2.5});
        for (let i = 0; i < 9; i++) {
            const x = -1 + i * 0.25;
            const y = 0.45 * x - 0.05 + (r() - 0.5) * 0.5;
            const yh = 0.45 * x - 0.05;
            const s = Math.abs(y - yh);
            p.polyline([x, yh, x, y, x + s * Math.sign(0.5 - (i % 2)), y, x + s * Math.sign(0.5 - (i % 2)), yh], {
                close: true,
                fill: C.yellow,
                fillAlpha: 0.12,
                color: rgba(C.yellow, 0.45),
                width: 1,
            });
            p.point(x, y, C.white, {r: 3.5});
        }
    },
    "gradient-descent": (p) => {
        for (let k = 1; k <= 7; k++) {
            const pts: number[] = [];
            for (let a = 0; a <= 64; a++) {
                const t = (a / 64) * Math.PI * 2;
                const rr = k * 0.11;
                pts.push(0.25 + rr * 1.6 * Math.cos(t) * 0.9 - rr * 0.3 * Math.sin(t), -0.05 + rr * 0.6 * Math.sin(t));
            }
            p.polyline(pts, {color: rgba(C.purple, 0.15 + 0.06 * (8 - k)), width: 1.2, close: true});
        }
        const path = [-0.9, 0.45, -0.55, -0.15, -0.3, 0.2, -0.05, -0.12, 0.12, 0.05, 0.22, -0.06, 0.25, -0.05];
        p.polyline(path, {color: C.yellow, width: 2});
        for (let i = 0; i < path.length; i += 2) p.circle(path[i], path[i + 1], 3, {fill: C.yellow});
    },
    perceptron: (p) => {
        const r = prng(9);
        p.halfPlane(0.6, 1, -0.05, rgba(C.blue, 0.1));
        p.halfPlane(-0.6, -1, 0.05, rgba(C.orange, 0.08));
        p.infiniteLine(0.6, 1, -0.05, {color: C.white, width: 2});
        for (let i = 0; i < 26; i++) {
            const x = (r() * 2 - 1) * 1.1;
            const y = (r() * 2 - 1) * 0.7;
            const s = 0.6 * x + y - 0.05;
            if (Math.abs(s) < 0.12) continue;
            p.point(x, y, s > 0 ? C.blue : C.orange, {r: 3.5});
        }
        p.arrow(0.02, 0.04, 0.32, 0.54, {color: C.yellow, width: 2.5, head: 9});
    },
    "logistic-regression": (p) => {
        // One smooth gradient instead of stripes, so there are no seams.
        const ctx = p.ctx;
        const g = ctx.createLinearGradient(0, 0, p.width, 0);
        for (let i = 0; i <= 20; i++) {
            const x = view.xmin + ((view.xmax - view.xmin) * i) / 20;
            const s = 1 / (1 + Math.exp(-(x + 0.03) * 4.5));
            g.addColorStop(i / 20, rgba(s > 0.5 ? C.blue : C.orange, Math.abs(s - 0.5) * 0.4));
        }
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, p.width, p.height);
        const pts: number[] = [];
        for (let i = 0; i <= 80; i++) {
            const x = -1.2 + (2.4 * i) / 80;
            pts.push(x, -0.55 + 1.1 / (1 + Math.exp(-x * 4.5)));
        }
        p.polyline(pts, {color: C.yellow, width: 2.5});
        const r = prng(4);
        for (let i = 0; i < 14; i++) {
            const x = (r() * 2 - 1) * 1.1;
            const c = x + (r() - 0.5) * 0.6 > 0;
            p.point(x, c ? 0.55 : -0.55, c ? C.blue : C.orange, {r: 3.5});
        }
    },
    "k-nearest-neighbors": (p) => {
        const r = prng(21);
        const pts: [number, number, number][] = [];
        const centers = [
            [-0.6, 0.25],
            [0.55, 0.3],
            [0.0, -0.4],
        ];
        for (let i = 0; i < 30; i++) {
            const c = i % 3;
            pts.push([centers[c][0] + gauss(r) * 0.22, centers[c][1] + gauss(r) * 0.16, c]);
        }
        const q = [0.05, 0.08];
        const d = pts.map((pt) => Math.hypot(pt[0] - q[0], pt[1] - q[1]));
        const order = d.map((_, i) => i).sort((a, b) => d[a] - d[b]).slice(0, 5);
        p.circle(q[0], q[1], d[order[4]] * p.kx, {stroke: rgba(C.white, 0.5), dash: [4, 4], width: 1.2});
        for (const i of order) p.line(q[0], q[1], pts[i][0], pts[i][1], {color: rgba(C.white, 0.35), width: 1});
        const cols = [C.orange, C.blue, C.green];
        for (const pt of pts) p.point(pt[0], pt[1], cols[pt[2]], {r: 3.5});
        p.circle(q[0], q[1], 4, {fill: C.white});
    },
    "neural-networks": (p) => {
        const layers = [2, 4, 4, 1];
        const xs = [-0.9, -0.3, 0.3, 0.9];
        const pos = layers.map((n, l) => Array.from({length: n}, (_, i) => [xs[l], (i - (n - 1) / 2) * 0.32]));
        const r = prng(5);
        for (let l = 0; l < layers.length - 1; l++)
            for (const a of pos[l])
                for (const b of pos[l + 1]) {
                    const w = r() * 2 - 1;
                    p.line(a[0], a[1], b[0], b[1], {color: rgba(w > 0 ? C.blue : C.orange, 0.25 + Math.abs(w) * 0.5), width: 0.6 + Math.abs(w) * 2});
                }
        pos.flat().forEach(([x, y], i) => {
            p.circle(x, y, 9, {fill: C.panel, stroke: i % 3 === 0 ? C.blue : i % 3 === 1 ? C.orange : C.purple, width: 2});
        });
    },
    "k-means": (p) => {
        const r = prng(12);
        const centers = [
            [-0.6, 0.3],
            [0.55, 0.35],
            [0.05, -0.38],
        ];
        const cols = [C.blue, C.green, C.pink];
        for (let i = 0; i < 45; i++) {
            const c = i % 3;
            p.point(centers[c][0] + gauss(r) * 0.18, centers[c][1] + gauss(r) * 0.14, cols[c], {r: 3.2, alpha: 0.9});
        }
        centers.forEach(([x, y], c) => {
            p.circle(x, y, 8, {fill: C.canvas, stroke: cols[c], width: 2.5});
            p.line(x - 4 / p.kx, y, x + 4 / p.kx, y, {color: cols[c], width: 2});
            p.line(x, y - 4 / p.ky, x, y + 4 / p.ky, {color: cols[c], width: 2});
        });
    },
    pca: (p) => {
        const r = prng(8);
        const a = 0.5;
        for (let i = 0; i < 50; i++) {
            const u = gauss(r) * 0.42;
            const v = gauss(r) * 0.1;
            p.point(u * Math.cos(a) - v * Math.sin(a), u * Math.sin(a) + v * Math.cos(a), C.teal, {r: 3, alpha: 0.85});
        }
        p.arrow(0, 0, 0.8 * Math.cos(a), 0.8 * Math.sin(a), {color: C.yellow, width: 2.5});
        p.arrow(0, 0, -0.25 * Math.sin(a), 0.25 * Math.cos(a), {color: C.pink, width: 2.5});
    },
    svm: (p) => {
        // The widest street: boundary, two margins, ringed support vectors.
        const [a, b, c] = [0.55, 1, 0.02];
        const n = Math.hypot(a, b);
        const m = 0.2 * n;
        p.halfPlane(a, b, c, rgba(C.blue, 0.1));
        p.halfPlane(-a, -b, -c, rgba(C.orange, 0.08));
        p.polyline(
            (() => {
                const s1 = p.clipLine(a, b, c - m);
                const s2 = p.clipLine(a, b, c + m);
                return s1 && s2 ? [s1[0], s1[1], s1[2], s1[3], s2[2], s2[3], s2[0], s2[1]] : [];
            })(),
            {close: true, fill: C.white, fillAlpha: 0.05, color: "none"}
        );
        p.infiniteLine(a, b, c - m, {color: rgba(C.white, 0.45), width: 1.2, dash: [5, 5]});
        p.infiniteLine(a, b, c + m, {color: rgba(C.white, 0.45), width: 1.2, dash: [5, 5]});
        p.infiniteLine(a, b, c, {color: C.white, width: 2.2});
        const r = prng(31);
        for (let i = 0; i < 30; i++) {
            const x = (r() * 2 - 1) * 1.1;
            const y = (r() * 2 - 1) * 0.7;
            const d = (a * x + b * y + c) / n;
            if (Math.abs(d) < 0.27) continue;
            p.point(x, y, d > 0 ? C.blue : C.orange, {r: 3.5});
        }
        for (const [x, side] of [
            [-0.55, 1],
            [0.35, 1],
            [-0.1, -1],
            [0.75, -1],
        ]) {
            const y = (-(c - side * m) - a * x) / b;
            p.point(x, y, side > 0 ? C.blue : C.orange, {r: 3.8, ring: C.white, ringWidth: 1.5});
        }
    },
    "decision-trees": (p) => {
        // Axis-aligned boxes: the staircase a tree carves.
        const boxes: [number, number, number, number, number][] = [
            [-1.2, -0.75, -0.35, 0.75, 0],
            [-0.35, 0.15, 0.4, 0.75, 1],
            [-0.35, -0.75, 0.4, 0.15, 0],
            [0.4, -0.75, 1.2, -0.2, 1],
            [0.4, -0.2, 1.2, 0.75, 2],
        ];
        const cols = [C.orange, C.blue, C.green];
        const r = prng(17);
        for (const [x0, y0, x1, y1, c] of boxes) {
            p.polyline([x0, y0, x1, y0, x1, y1, x0, y1], {close: true, fill: cols[c], fillAlpha: 0.13, color: "none"});
            const area = (x1 - x0) * (y1 - y0);
            for (let i = 0; i < Math.round(area * 9); i++) p.point(x0 + 0.08 + r() * (x1 - x0 - 0.16), y0 + 0.08 + r() * (y1 - y0 - 0.16), cols[c], {r: 3.2});
        }
        p.line(-0.35, -0.75, -0.35, 0.75, {color: C.white, width: 2});
        p.line(0.4, -0.75, 0.4, 0.75, {color: C.white, width: 2});
        p.line(-0.35, 0.15, 0.4, 0.15, {color: rgba(C.white, 0.75), width: 1.6});
        p.line(0.4, -0.2, 1.2, -0.2, {color: rgba(C.white, 0.75), width: 1.6});
    },
    "naive-bayes": (p) => {
        // One axis-aligned Gaussian per class; the boundary is where they tie.
        const cls = [
            {c: C.orange, x: -0.45, y: 0.05, sx: 0.32, sy: 0.18},
            {c: C.blue, x: 0.5, y: -0.05, sx: 0.16, sy: 0.32},
        ];
        const r = prng(23);
        for (const k of cls) {
            for (const s of [1, 2]) ellipse(p, k.x, k.y, k.sx * s, k.sy * s, 0, {color: rgba(k.c, s === 1 ? 0.8 : 0.4), width: s === 1 ? 1.8 : 1.2, dash: s === 2 ? [4, 4] : undefined, fill: k.c, fillAlpha: s === 1 ? 0.08 : 0});
            for (const [x, y] of blob(r, 16, k.x, k.y, k.sx * 0.8, k.sy * 0.8)) p.point(x, y, k.c, {r: 3.1});
        }
        const bd: number[] = [];
        for (let i = 0; i <= 40; i++) {
            const y = -0.75 + (1.5 * i) / 40;
            bd.push(0.18 - 0.55 * y * y, y);
        }
        p.polyline(bd, {color: C.white, width: 2});
    },
    overfitting: (p) => {
        const truth = (x: number) => 0.32 * Math.sin(2.4 * x);
        const r = prng(6);
        const xs = Array.from({length: 9}, (_, i) => -1 + i * 0.25);
        const ys = xs.map((x) => truth(x) + (r() - 0.5) * 0.3);
        // Lagrange interpolation through every point: the degree-8 polynomial that overfits.
        const lag = (x: number) =>
            ys.reduce((sum, y, i) => {
                let t = y;
                for (let j = 0; j < xs.length; j++) if (j !== i) t *= (x - xs[j]) / (xs[i] - xs[j]);
                return sum + t;
            }, 0);
        const wild: number[] = [];
        const smooth: number[] = [];
        for (let i = 0; i <= 160; i++) {
            const x = -1.1 + (2.2 * i) / 160;
            wild.push(x, Math.max(-0.74, Math.min(0.74, lag(x))));
            smooth.push(x, truth(x));
        }
        p.polyline(smooth, {color: rgba(C.green, 0.85), width: 2, dash: [6, 5]});
        p.polyline(wild, {color: C.pink, width: 2.4});
        xs.forEach((x, i) => p.point(x, ys[i], C.blue, {r: 3.6}));
        for (const x of [-0.88, -0.38, 0.12, 0.62]) p.circle(x, truth(x) + 0.04, 3.4, {stroke: C.text2, width: 1.4});
    },
    "gaussian-processes": (p) => {
        const f = (x: number) => 0.3 * Math.sin(2.2 * x) + 0.1 * x;
        const obs = [-0.85, -0.3, 0.25, 0.8];
        const sd = (x: number) => 0.3 * Math.sqrt(1 - Math.exp(-(Math.min(...obs.map((o) => (x - o) ** 2)) / 0.05))) + 0.015;
        const xs = Array.from({length: 121}, (_, i) => -1.2 + (2.4 * i) / 120);
        for (const k of [2, 1]) {
            const up: number[] = [];
            const down: number[] = [];
            for (const x of xs) {
                up.push(x, f(x) + k * sd(x));
                down.unshift(x, f(x) - k * sd(x));
            }
            p.polyline([...up, ...down], {close: true, fill: C.purple, fillAlpha: k === 1 ? 0.2 : 0.1, color: "none"});
        }
        for (const [amp, ph] of [
            [0.9, 0.3],
            [-0.8, 1.7],
            [0.7, 3.1],
        ]) {
            const pts: number[] = [];
            for (const x of xs) pts.push(x, f(x) + amp * sd(x) * Math.sin(5 * x + ph));
            p.polyline(pts, {color: rgba(C.purple, 0.55), width: 1});
        }
        const mean: number[] = [];
        for (const x of xs) mean.push(x, f(x));
        p.polyline(mean, {color: C.yellow, width: 2.4});
        for (const x of obs) p.point(x, f(x), C.white, {r: 3.8});
    },
    "gaussian-mixtures": (p) => {
        const comps = [
            {c: C.blue, x: -0.6, y: 0.15, rx: 0.36, ry: 0.11, rot: 0.6},
            {c: C.green, x: 0.15, y: -0.3, rx: 0.4, ry: 0.13, rot: -0.15},
            {c: C.pink, x: 0.65, y: 0.32, rx: 0.2, ry: 0.17, rot: 0.2},
        ];
        const r = prng(41);
        for (const k of comps) {
            for (const [x, y] of blob(r, 18, k.x, k.y, k.rx * 0.75, k.ry * 0.75, k.rot)) p.point(x, y, k.c, {r: 3, alpha: 0.9});
            ellipse(p, k.x, k.y, k.rx, k.ry, k.rot, {color: k.c, width: 2, fill: k.c, fillAlpha: 0.1});
            ellipse(p, k.x, k.y, k.rx * 1.8, k.ry * 1.8, k.rot, {color: rgba(k.c, 0.45), width: 1.2, dash: [4, 4]});
        }
    },
    dbscan: (p) => {
        // Two interleaved moons found by density, with stray noise left out.
        const r = prng(53);
        const moon = (cx: number, cy: number, flip: number, color: string) => {
            for (let i = 0; i < 26; i++) {
                const t = (i / 25) * Math.PI;
                const x = cx + Math.cos(t) * 0.5 + (r() - 0.5) * 0.08;
                const y = cy + flip * Math.sin(t) * 0.42 + (r() - 0.5) * 0.08;
                const border = i === 0 || i === 25 || i % 9 === 4;
                if (border) p.circle(x, y, 3.4, {stroke: color, width: 1.6});
                else p.point(x, y, color, {r: 3.4});
                if (i === 8) p.circle(x, y, 0.17 * p.kx, {stroke: rgba(color, 0.4), width: 1, dash: [3, 3]});
            }
        };
        moon(-0.3, -0.15, 1, C.blue);
        moon(0.25, 0.15, -1, C.green);
        for (const [x, y] of [
            [-1.0, 0.55],
            [0.95, -0.58],
            [1.0, 0.5],
            [-0.95, -0.55],
            [0.05, 0.62],
        ]) {
            const d = 0.03;
            p.line(x - d, y - d, x + d, y + d, {color: C.muted, width: 1.6});
            p.line(x - d, y + d, x + d, y - d, {color: C.muted, width: 1.6});
        }
    },
    "q-learning": (p) => {
        // A gridworld: value spreading back from the goal, arrows for the greedy policy.
        const W = 8;
        const H = 5;
        const s = 0.27;
        const x0 = -(W * s) / 2;
        const y0 = -(H * s) / 2;
        const walls = new Set(["2,1", "2,2", "2,3", "5,3", "5,2", "5,0"]);
        const goal = [7, 4];
        const pit = [6, 1];
        const cell = (i: number, j: number) => [x0 + i * s, y0 + j * s];
        for (let i = 0; i < W; i++)
            for (let j = 0; j < H; j++) {
                const [x, y] = cell(i, j);
                const key = `${i},${j}`;
                let fill: string | null = null;
                let alpha = 0;
                if (walls.has(key)) [fill, alpha] = [C.muted, 0.28];
                else if (i === goal[0] && j === goal[1]) [fill, alpha] = [C.green, 0.65];
                else if (i === pit[0] && j === pit[1]) [fill, alpha] = [C.red, 0.55];
                else {
                    const d = Math.abs(i - goal[0]) + Math.abs(j - goal[1]);
                    [fill, alpha] = [C.green, Math.max(0, 0.42 - d * 0.045)];
                }
                if (fill) p.polyline([x, y, x + s, y, x + s, y + s, x, y + s], {close: true, fill, fillAlpha: alpha, color: "none"});
                p.polyline([x, y, x + s, y, x + s, y + s, x, y + s], {close: true, color: rgba(C.white, 0.08), width: 1});
            }
        // Greedy route from the start (bottom left) to the goal.
        const route: [number, number][] = [
            [0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4],
        ];
        for (let k = 0; k < route.length - 1; k++) {
            const [ax, ay] = cell(route[k][0] + 0.5, route[k][1] + 0.5);
            const [bx, by] = cell(route[k + 1][0] + 0.5, route[k + 1][1] + 0.5);
            p.arrow(ax, ay, ax + (bx - ax) * 0.55, ay + (by - ay) * 0.55, {color: rgba(C.white, 0.85), width: 1.4, head: 6});
        }
        const [gx, gy] = cell(goal[0] + 0.5, goal[1] + 0.5);
        p.text("+1", gx, gy, {color: C.white, size: 11, align: "center", baseline: "middle", weight: 700});
        const [ax, ay] = cell(3.5, 4.5);
        p.circle(ax, ay, 0.085 * p.kx, {fill: C.yellow, stroke: C.canvas, width: 2});
    },
    bandits: (p) => {
        // Thompson sampling's beliefs: one Beta posterior per arm, the best one sharpest.
        const arms = [
            [3, 6],
            [9, 8],
            [26, 9],
            [2, 3],
        ];
        const lg = (z: number): number => {
            // Stirling-series log-gamma, accurate enough for a picture.
            return (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI) + 1 / (12 * z);
        };
        arms.forEach(([a, b], k) => {
            const cx = -0.9 + k * 0.6;
            const ys: number[] = [];
            for (let i = 0; i <= 80; i++) {
                const t = Math.min(0.995, Math.max(0.005, i / 80));
                ys.push(Math.exp((a - 1) * Math.log(t) + (b - 1) * Math.log(1 - t) + lg(a + b) - lg(a) - lg(b)));
            }
            const peak = Math.max(...ys);
            const best = k === 2;
            const color = best ? C.yellow : C.pink;
            const pts: number[] = [];
            ys.forEach((d, i) => pts.push(cx + (0.24 * d) / peak, -0.62 + (1.24 * i) / 80));
            for (let i = ys.length - 1; i >= 0; i--) pts.push(cx - (0.24 * ys[i]) / peak, -0.62 + (1.24 * i) / 80);
            p.polyline(pts, {close: true, fill: color, fillAlpha: best ? 0.35 : 0.22, color, width: 1.5});
            const mean = a / (a + b);
            p.line(cx - 0.18, -0.62 + 1.24 * mean, cx + 0.18, -0.62 + 1.24 * mean, {color: C.white, width: 1.5, dash: [3, 3]});
        });
    },
    "t-sne": (p) => {
        // Tight islands: each high-dimensional cluster pulled into its own patch.
        const r = prng(61);
        const islands = [
            [-0.8, 0.35],
            [-0.25, 0.48],
            [0.45, 0.38],
            [0.85, -0.25],
            [0.1, -0.4],
            [-0.65, -0.35],
        ];
        islands.forEach(([x, y], c) => {
            for (const [px, py] of blob(r, 16, x, y, 0.07, 0.06, c)) p.point(px, py, classColor(c + 1), {r: 3});
        });
    },
};

const fallback = (p: Painter) => {
    p.infiniteLine(1, -1.5, 0, {color: rgba(C.muted, 0.6), width: 1.5, dash: [5, 6]});
};

export function Thumb({slug}: {slug: string}) {
    return (
        <Plot
            label=""
            view={view}
            draw={(p) => {
                p.clear(C.canvas);
                p.grid(0.25, {axes: false});
                (DRAW[slug] ?? fallback)(p);
            }}
        />
    );
}
