import type {Painter} from "../lib/painter";
import {C, rgba} from "../lib/theme";
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
        const steps = 40;
        for (let i = 0; i < steps; i++) {
            const x0 = view.xmin + ((view.xmax - view.xmin) * i) / steps;
            const x1 = x0 + (view.xmax - view.xmin) / steps + 0.01;
            const s = 1 / (1 + Math.exp(-(x0 + 0.03) * 4.5));
            const col = s > 0.5 ? C.blue : C.orange;
            p.polyline([x0, view.ymin, x1, view.ymin, x1, view.ymax, x0, view.ymax], {close: true, fill: col, fillAlpha: Math.abs(s - 0.5) * 0.4, color: "none"});
        }
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
            p.circle(x, y, 9, {fill: "#18202e", stroke: i % 3 === 0 ? C.blue : i % 3 === 1 ? C.orange : C.purple, width: 2});
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
