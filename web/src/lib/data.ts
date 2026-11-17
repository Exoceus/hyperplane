import type {Painter} from "./painter";
import {C, classColor} from "./theme";

/** Index of the data point within `radiusPx` of (x, y), or -1. */
export function hitPoint(data: ArrayLike<number>, stride: number, x: number, y: number, kx: number, ky: number, radiusPx = 9): number {
    let best = -1;
    let bestD = radiusPx * radiusPx;
    for (let i = 0, k = 0; i + 1 < data.length; i += stride, k++) {
        const d = ((data[i] - x) * kx) ** 2 + ((data[i + 1] - y) * ky) ** 2;
        if (d <= bestD) {
            bestD = d;
            best = k;
        }
    }
    return best;
}

export function appendRow(data: Float64Array, row: number[]): Float64Array {
    const out = new Float64Array(data.length + row.length);
    out.set(data);
    out.set(row, data.length);
    return out;
}

export function removeRow(data: Float64Array, stride: number, index: number): Float64Array {
    const out = new Float64Array(data.length - stride);
    out.set(data.subarray(0, index * stride));
    out.set(data.subarray((index + 1) * stride), index * stride);
    return out;
}

export function setRow(data: Float64Array, stride: number, index: number, row: number[]): Float64Array {
    const out = data.slice();
    out.set(row, index * stride);
    return out;
}

/** Drop the label column from stride-3 data. */
export function coordsOnly(data: Float64Array): Float64Array {
    const n = data.length / 3;
    const out = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
        out[2 * i] = data[3 * i];
        out[2 * i + 1] = data[3 * i + 1];
    }
    return out;
}

/** Draw stride-3 labelled points with the standard glyph. */
export function drawLabelled(p: Painter, data: ArrayLike<number>, opts: {r?: number; alpha?: number; ring?: (i: number) => string | undefined} = {}) {
    for (let i = 0, k = 0; i < data.length; i += 3, k++) {
        p.point(data[i], data[i + 1], classColor(data[i + 2]), {r: opts.r ?? 4.5, alpha: opts.alpha, ring: opts.ring?.(k)});
    }
}

export function drawPlain(p: Painter, data: ArrayLike<number>, color: string = C.white, r = 4) {
    for (let i = 0; i < data.length; i += 2) p.point(data[i], data[i + 1], color, {r});
}
