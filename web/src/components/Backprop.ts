import type {Painter} from "../lib/painter";
import {C, hexToRgb, mix, type RGB} from "../lib/theme";

/**
 * One training point traced through a small tanh network (sigmoid output,
 * cross-entropy loss): every activation on the way forward, every error
 * signal and weight gradient on the way back. Computed in JS from a snapshot
 * of the live network's parameters, mirroring the Rust core exactly.
 */
export interface Trace {
    x: number;
    y: number;
    label: number;
    p: number;
    loss: number;
    /** Activations per layer, input layer first. */
    acts: number[][];
    /** dLoss/dz for the outputs of each weight layer. */
    deltas: number[][];
    /** Weights per layer, row-major [out][in] (the snapshot). */
    W: number[][];
    /** dLoss/dW, same layout as W. */
    grads: number[][];
}

/** Unpack `Mlp.params()`: per layer, weights row-major [out][in], then biases. */
export function unpack(params: ArrayLike<number>, sizes: number[]): {W: number[][]; B: number[][]} {
    const W: number[][] = [];
    const B: number[][] = [];
    let off = 0;
    for (let l = 0; l < sizes.length - 1; l++) {
        const n = sizes[l] * sizes[l + 1];
        W.push(Array.from({length: n}, (_, k) => params[off + k]));
        off += n;
        B.push(Array.from({length: sizes[l + 1]}, (_, k) => params[off + k]));
        off += sizes[l + 1];
    }
    return {W, B};
}

export function traceSample(params: ArrayLike<number>, sizes: number[], x: number, y: number, label: number): Trace {
    const {W, B} = unpack(params, sizes);
    const L = sizes.length - 1;
    const acts: number[][] = [[x, y]];
    for (let l = 0; l < L; l++) {
        const nin = sizes[l];
        const out: number[] = [];
        for (let o = 0; o < sizes[l + 1]; o++) {
            let z = B[l][o];
            for (let i = 0; i < nin; i++) z += W[l][o * nin + i] * acts[l][i];
            out.push(l === L - 1 ? 1 / (1 + Math.exp(-z)) : Math.tanh(z));
        }
        acts.push(out);
    }
    const p = acts[L][0];
    const eps = 1e-9;
    const loss = -(label * Math.log(p + eps) + (1 - label) * Math.log(1 - p + eps));
    // Backward: sigmoid + cross-entropy gives dL/dz = p − y at the output.
    const deltas: number[][] = new Array(L);
    deltas[L - 1] = [p - label];
    for (let l = L - 1; l >= 1; l--) {
        const nin = sizes[l];
        deltas[l - 1] = Array.from({length: nin}, (_, i) => {
            let s = 0;
            for (let o = 0; o < sizes[l + 1]; o++) s += W[l][o * nin + i] * deltas[l][o];
            return s * (1 - acts[l][i] ** 2);
        });
    }
    const grads = W.map((w, l) => w.map((_, k) => deltas[l][Math.floor(k / sizes[l])] * acts[l][k % sizes[l]]));
    return {x, y, label, p, loss, acts, deltas, W, grads};
}

// ---------------------------------------------------------------------------
// Timing of one cycle: pick a point, forward, loss, backward, update.
// ---------------------------------------------------------------------------

export const CYCLE = 4.2;
const T = {forward: [0.35, 1.65], loss: [1.65, 2.25], backward: [2.25, 3.55], update: [3.55, 4.2]} as const;

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const ease = (v: number) => v * v * (3 - 2 * v);
const span = (t: number, [a, b]: readonly [number, number]) => clamp01((t - a) / (b - a));

export function phaseName(t: number): string {
    if (t < T.forward[0]) return "pick a training point";
    if (t < T.loss[0]) return "forward pass";
    if (t < T.backward[0]) return "measure the loss";
    if (t < T.update[0]) return "backpropagate";
    return "update the weights";
}

// ---------------------------------------------------------------------------
// The diagram.
// ---------------------------------------------------------------------------

export const NET_VIEW = {xmin: 0, xmax: 100, ymin: 0, ymax: 32};
const COLS = [6, 35, 64, 92];
const PINK = C.pink;
const BLUE = hexToRgb(C.blue);
const ORANGE = hexToRgb(C.orange);
const CANVAS = hexToRgb(C.canvas);

function nodeY(i: number, n: number): number {
    const gap = n > 2 ? 2.45 : 8;
    return 17 - (i - (n - 1) / 2) * gap;
}

const rgb = (c: RGB) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
/** Signed value in [−1, 1] → orange (negative) … canvas … blue (positive). */
const signed = (v: number) => rgb(mix(CANVAS, v >= 0 ? BLUE : ORANGE, Math.min(1, Math.abs(v)) * 0.9));

const maxAbs = (xs: number[]) => xs.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-12);

/** Draw the network and the traced point's pulses at cycle time `t`. */
export function drawNetwork(p: Painter, tr: Trace, sizes: number[], t: number) {
    p.clear();
    const f = ease(span(t, T.forward)) * 3; // forward front: 0 (input) → 3 (output)
    const g = 3 - ease(span(t, T.backward)) * 3; // backward front: 3 → 0
    const u = ease(span(t, T.update));
    const backStarted = t >= T.backward[0];
    // Node radius in pixels: as large as the spacing of a 10-neuron column allows.
    const rPx = Math.min(5.2, 2.45 * p.ky * 0.4);
    const r = rPx / p.kx;
    const W = tr.W;
    const wMax = Math.max(...W.map(maxAbs));

    for (let l = 0; l < sizes.length - 1; l++) {
        const nin = sizes[l];
        const nout = sizes[l + 1];
        const signal = W[l].map((w, k) => w * tr.acts[l][k % nin]);
        const sMax = maxAbs(signal);
        const gMax = maxAbs(tr.grads[l]);
        const fs = f - l; // progress of the forward front along this layer
        const bs = l + 1 - g; // progress of the backward front along this layer
        for (let o = 0; o < nout; o++)
            for (let i = 0; i < nin; i++) {
                const k = o * nin + i;
                const x1 = COLS[l] + r;
                const y1 = nodeY(i, nin);
                const x2 = COLS[l + 1] - r;
                const y2 = nodeY(o, nout);
                const w = W[l][k];
                const wn = Math.abs(w) / wMax;
                p.line(x1, y1, x2, y2, {color: w >= 0 ? C.blue : C.orange, alpha: 0.1 + 0.32 * wn, width: 0.5 + 1.6 * wn});
                // Forward: the weighted signal travels along each edge.
                const sn = Math.abs(signal[k]) / sMax;
                if (fs > 0 && fs < 1) {
                    p.line(x1, y1, x2, y2, {color: C.white, alpha: 0.35 * sn, width: 1});
                    const e = fs;
                    p.circle(x1 + (x2 - x1) * e, y1 + (y2 - y1) * e, 1.2 + 2 * sn, {fill: C.white, alpha: 0.25 + 0.75 * sn});
                }
                // Backward: the gradient travels back, then stays lit until the
                // update, when it fades as the weights take their step.
                const gn = Math.abs(tr.grads[l][k]) / gMax;
                if (backStarted && bs > 0) {
                    if (bs < 1) {
                        const e = 1 - bs;
                        p.circle(x1 + (x2 - x1) * e, y1 + (y2 - y1) * e, 1.2 + 2.2 * gn, {fill: PINK, alpha: 0.25 + 0.75 * gn});
                    }
                    const glow = Math.min(1, bs) * (1 - u);
                    if (glow > 0) p.line(x1, y1, x2, y2, {color: PINK, alpha: 0.75 * gn * glow, width: 0.6 + 2.2 * gn});
                }
            }
    }

    // Nodes: filled with their activation once the forward pass reaches them,
    // ringed with their error signal once backprop does.
    for (let l = 0; l < sizes.length; l++) {
        const n = sizes[l];
        const reached = l === 0 ? clamp01(t / 0.3) : clamp01((f - (l - 1)) * 1.5 - 0.5);
        const dMax = l > 0 ? maxAbs(tr.deltas[l - 1]) : 1;
        for (let i = 0; i < n; i++) {
            const x = COLS[l];
            const y = nodeY(i, n);
            const a = tr.acts[l][i];
            const v = l === 0 ? a / 1.3 : l === sizes.length - 1 ? (a - 0.5) * 2 : a;
            p.circle(x, y, rPx, {fill: C.canvas, stroke: C.axis, width: 1});
            if (reached > 0) p.circle(x, y, rPx, {fill: signed(v), alpha: reached});
            if (l > 0 && backStarted && g <= l) {
                const d = Math.abs(tr.deltas[l - 1][i]) / dMax;
                p.circle(x, y, rPx + 1.8, {stroke: PINK, width: 0.8 + 2.4 * d, alpha: (0.35 + 0.65 * d) * (1 - u * 0.7)});
            }
        }
    }

    // Labels.
    const muted = {color: C.muted, size: 10, font: "mono" as const, align: "center" as const};
    p.text("input", COLS[0], 1.2, muted);
    p.text("hidden", COLS[1], 1.2, muted);
    p.text("hidden", COLS[2], 1.2, muted);
    p.text("output", COLS[3], 1.2, muted);
    const step = t < T.forward[0] ? 0 : t < T.loss[0] ? 1 : t < T.backward[0] ? 2 : t < T.update[0] ? 3 : 4;
    p.textPx(`${["·", "①", "②", "③", "④"][step]} ${phaseName(t)}`, 10, 16, {color: step === 3 ? PINK : C.text2, size: 11.5, font: "mono"});
    if (t >= T.loss[0]) {
        const target = tr.label > 0.5 ? "blue" : "orange";
        const right = p.width - 10;
        const narrow = p.width < 480;
        p.textPx(`p(blue) = ${tr.p.toFixed(2)}`, right, 16, {color: C.text2, size: 11, font: "mono", align: "right"});
        p.textPx(narrow ? `loss ${tr.loss.toFixed(3)}` : `truth: ${target} · loss ${tr.loss.toFixed(3)}`, right, 31, {color: tr.loss > 0.3 ? PINK : C.muted, size: 11, font: "mono", align: "right"});
    }
}
