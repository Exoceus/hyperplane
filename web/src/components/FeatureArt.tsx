import {useState} from "react";
import {useAnimationFrame, useInView} from "../lib/hooks";
import type {Painter} from "../lib/painter";
import {C, rgba} from "../lib/theme";
import {Plot} from "./Plot";

const VIEW = {xmin: -1.6, xmax: 1.6, ymin: -0.9, ymax: 0.9};

// ---------------------------------------------------------------------------
// Gradient descent: three optimizers racing down a narrow, tilted valley.
// ---------------------------------------------------------------------------

const ROT = 0.45;
const [CA, SA] = [Math.cos(ROT), Math.sin(ROT)];
/** f = (u² + 14 v²) / 2 in a rotated frame: a long, narrow valley. */
function grad(x: number, y: number): [number, number] {
    const u = CA * x + SA * y;
    const v = -SA * x + CA * y;
    const gu = u;
    const gv = 14 * v;
    return [CA * gu - SA * gv, SA * gu + CA * gv];
}

type Path = number[];
function race(): {color: string; path: Path}[] {
    // Far along the valley, a little up one wall: (u, v) = (−1.35, 0.22).
    const start: [number, number] = [CA * -1.35 - SA * 0.22, SA * -1.35 + CA * 0.22];
    const steps = 90;
    const gd: Path = [...start];
    const mom: Path = [...start];
    const adam: Path = [...start];
    let [x, y] = start;
    for (let i = 0; i < steps; i++) {
        const [gx, gy] = grad(x, y);
        x -= 0.125 * gx;
        y -= 0.125 * gy;
        gd.push(x, y);
    }
    [x, y] = start;
    let [vx, vy] = [0, 0];
    for (let i = 0; i < steps; i++) {
        const [gx, gy] = grad(x, y);
        vx = 0.85 * vx - 0.025 * gx;
        vy = 0.85 * vy - 0.025 * gy;
        x += vx;
        y += vy;
        mom.push(x, y);
    }
    [x, y] = start;
    let [m1, m2, s1, s2] = [0, 0, 0, 0];
    for (let i = 1; i <= steps; i++) {
        const [gx, gy] = grad(x, y);
        m1 = 0.9 * m1 + 0.1 * gx;
        m2 = 0.9 * m2 + 0.1 * gy;
        s1 = 0.999 * s1 + 0.001 * gx * gx;
        s2 = 0.999 * s2 + 0.001 * gy * gy;
        const c1 = 1 - 0.9 ** i;
        const c2 = 1 - 0.999 ** i;
        x -= (0.06 * (m1 / c1)) / (Math.sqrt(s1 / c2) + 1e-8);
        y -= (0.06 * (m2 / c1)) / (Math.sqrt(s2 / c2) + 1e-8);
        adam.push(x, y);
    }
    return [
        {color: C.white, path: gd},
        {color: C.blue, path: mom},
        {color: C.pink, path: adam},
    ];
}
const RACE = race();

function drawDescent(p: Painter, t: number) {
    // Contours of the valley.
    for (let k = 1; k <= 9; k++) {
        const level = 0.018 * k * k;
        const pts: number[] = [];
        for (let a = 0; a <= 96; a++) {
            const th = (a / 96) * Math.PI * 2;
            const u = Math.sqrt(2 * level) * Math.cos(th);
            const v = Math.sqrt((2 * level) / 14) * Math.sin(th);
            pts.push(CA * u - SA * v, SA * u + CA * v);
        }
        p.polyline(pts, {close: true, color: rgba(C.purple, 0.1 + 0.05 * (10 - k)), width: 1.2, fill: C.purple, fillAlpha: 0.025});
    }
    const n = RACE[0].path.length / 2;
    const k = Math.min(n - 1, t);
    for (const r of RACE) {
        const i = Math.floor(k);
        const f = k - i;
        const pts = r.path.slice(0, 2 * (i + 1));
        const j = Math.min(n - 1, i + 1);
        const cx = r.path[2 * i] + (r.path[2 * j] - r.path[2 * i]) * f;
        const cy = r.path[2 * i + 1] + (r.path[2 * j + 1] - r.path[2 * i + 1]) * f;
        pts.push(cx, cy);
        p.polyline(pts, {color: r.color, width: 2, alpha: 0.85});
        p.circle(cx, cy, 6, {fill: r.color, stroke: C.canvas, width: 2});
    }
    p.circle(RACE[0].path[0], RACE[0].path[1], 6, {stroke: C.white, width: 1.5, dash: [3, 3]});
}

// ---------------------------------------------------------------------------
// Q-learning: an agent walking its learned route while value spreads back.
// ---------------------------------------------------------------------------

const W = 10;
const H = 6;
const S = 0.27;
const X0 = -(W * S) / 2;
const Y0 = -(H * S) / 2;
const WALLS = new Set(["3,0", "3,1", "3,2", "3,3", "6,5", "6,4", "6,3", "6,2", "8,1"]);
const GOAL = [9, 0];
const PIT = [8, 0];
/** The greedy route the agent has learned: around both walls, past the pit, to the goal. */
const ROUTE_FULL: [number, number][] = [
    [0, 1], [1, 1], [2, 1], [2, 2], [2, 3], [2, 4], [3, 4], [4, 4], [5, 4], [5, 3],
    [5, 2], [5, 1], [6, 1], [7, 1], [7, 2], [8, 2], [9, 2], [9, 1], [9, 0],
];

function cell(i: number, j: number): [number, number] {
    return [X0 + i * S, Y0 + j * S];
}

function drawGrid(p: Painter, t: number) {
    const steps = ROUTE_FULL.length - 1;
    const cycle = t % (steps + 4);
    const reach = Math.min(1, cycle / steps);
    for (let i = 0; i < W; i++)
        for (let j = 0; j < H; j++) {
            const [x, y] = cell(i, j);
            const key = `${i},${j}`;
            const box = [x, y, x + S, y, x + S, y + S, x, y + S];
            if (WALLS.has(key)) {
                p.polyline(box, {close: true, fill: "#302e29", color: "#403d36", width: 1});
                continue;
            }
            if (i === GOAL[0] && j === GOAL[1]) p.polyline(box, {close: true, fill: C.green, fillAlpha: 0.7, color: "none"});
            else if (i === PIT[0] && j === PIT[1]) p.polyline(box, {close: true, fill: C.red, fillAlpha: 0.55, color: "none"});
            else {
                const d = Math.abs(i - GOAL[0]) + Math.abs(j - GOAL[1]);
                p.polyline(box, {close: true, fill: C.green, fillAlpha: Math.max(0, 0.4 - d * 0.03) * (0.4 + 0.6 * reach), color: "none"});
            }
            p.polyline(box, {close: true, color: rgba(C.white, 0.07), width: 1});
        }
    for (let k = 0; k < ROUTE_FULL.length - 1; k++) {
        const [ax, ay] = cell(ROUTE_FULL[k][0] + 0.5, ROUTE_FULL[k][1] + 0.5);
        const [bx, by] = cell(ROUTE_FULL[k + 1][0] + 0.5, ROUTE_FULL[k + 1][1] + 0.5);
        p.arrow(ax, ay, ax + (bx - ax) * 0.55, ay + (by - ay) * 0.55, {color: rgba(C.white, 0.5), width: 1.3, head: 6});
    }
    const [gx, gy] = cell(GOAL[0] + 0.5, GOAL[1] + 0.5);
    p.text("+1", gx, gy, {color: C.white, size: 12, align: "center", baseline: "middle", weight: 700});
    const [px, py] = cell(PIT[0] + 0.5, PIT[1] + 0.5);
    p.text("−1", px, py, {color: C.white, size: 12, align: "center", baseline: "middle", weight: 700});
    // The agent, gliding cell to cell.
    const pos = Math.min(steps, cycle);
    const i = Math.floor(pos);
    const f = pos - i;
    const e = f * f * (3 - 2 * f);
    const a = ROUTE_FULL[i];
    const b = ROUTE_FULL[Math.min(steps, i + 1)];
    const [x, y] = cell(a[0] + (b[0] - a[0]) * e + 0.5, a[1] + (b[1] - a[1]) * e + 0.5);
    p.circle(x, y, 0.09 * p.kx, {fill: C.yellow, stroke: C.canvas, width: 2.5});
}

const ART: Record<string, {draw: (p: Painter, t: number) => void; rate: number; loop: number}> = {
    "gradient-descent": {draw: drawDescent, rate: 14, loop: RACE[0].path.length / 2 + 30},
    "q-learning": {draw: drawGrid, rate: 2.6, loop: Infinity},
};

/** A small looping animation for a featured playground (paused while offscreen). */
export function FeatureArt({slug, label}: {slug: string; label: string}) {
    const [ref, visible] = useInView<HTMLDivElement>("0px");
    const [t, setT] = useState(0);
    const art = ART[slug];
    useAnimationFrame((dt) => setT((v) => (v + dt * art.rate) % art.loop), visible && !!art);
    if (!art) return null;
    return (
        <div ref={ref}>
            <Plot
                label={label}
                view={VIEW}
                draw={(p) => {
                    p.clear();
                    p.grid(0.2, {axes: false});
                    art.draw(p, t);
                }}
            />
        </div>
    );
}
