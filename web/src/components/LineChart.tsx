import {C} from "../lib/theme";
import {Plot} from "./Plot";

export interface Series {
    values: ArrayLike<number>;
    color: string;
    /** x value of index 0 and the spacing between samples. */
    x0?: number;
    dx?: number;
    width?: number;
    dash?: number[];
    /** Draw only the first `upto` (fractional) samples. */
    upto?: number;
    dots?: boolean;
}

interface LineChartProps {
    series: Series[];
    label: string;
    height?: number;
    xMax?: number;
    xMin?: number;
    yMin?: number;
    yMax?: number;
    logY?: boolean;
    /** Vertical marker at this x. */
    marker?: number;
    xLabel?: string;
    yLabel?: string;
    yFormat?: (v: number) => string;
    onPick?: (x: number) => void;
}

function niceStep(range: number, target = 4): number {
    const raw = range / target;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

/** Small auto-scaled line chart (loss curves, accuracy vs k, …). */
export function LineChart({series, label, height = 150, xMin, xMax, yMin, yMax, logY, marker, xLabel, yLabel, yFormat, onPick}: LineChartProps) {
    const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-12)) : v);
    let lo = Infinity;
    let hi = -Infinity;
    let maxX = 1;
    let minX = Infinity;
    for (const s of series) {
        const n = s.upto !== undefined ? Math.min(s.values.length, Math.ceil(s.upto) + 1) : s.values.length;
        for (let i = 0; i < n; i++) {
            const v = s.values[i];
            if (!Number.isFinite(v)) continue;
            lo = Math.min(lo, tf(v));
            hi = Math.max(hi, tf(v));
        }
        const x0 = s.x0 ?? 0;
        const dx = s.dx ?? 1;
        maxX = Math.max(maxX, x0 + dx * (s.values.length - 1));
        minX = Math.min(minX, x0);
    }
    if (!Number.isFinite(lo)) {
        lo = 0;
        hi = 1;
    }
    if (yMin !== undefined) lo = tf(yMin);
    if (yMax !== undefined) hi = tf(yMax);
    if (hi - lo < 1e-9) {
        hi += 0.5;
        lo -= 0.5;
    }
    const pad = (hi - lo) * 0.12;
    const x0 = xMin ?? (Number.isFinite(minX) ? minX : 0);
    const x1 = xMax ?? maxX;
    const view = {xmin: x0, xmax: x1 + (x1 - x0) * 0.02, ymin: lo - pad, ymax: hi + pad};

    return (
        <Plot
            label={label}
            view={view}
            height={height}
            className="chart"
            draw={(p) => {
                p.clear(C.canvas);
                const step = niceStep(hi - lo);
                p.ctx.lineWidth = 1;
                for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
                    p.line(view.xmin, v, view.xmax, v, {color: C.grid, width: 1});
                    const shown = logY ? 10 ** v : v;
                    p.textPx(yFormat ? yFormat(shown) : String(+shown.toPrecision(3)), 4, p.sy(v) - 3, {color: C.muted, size: 9.5, font: "mono"});
                }
                if (marker !== undefined) p.line(marker, view.ymin, marker, view.ymax, {color: C.muted, width: 1, dash: [3, 4]});
                for (const s of series) {
                    const sx0 = s.x0 ?? 0;
                    const dx = s.dx ?? 1;
                    const n = s.upto !== undefined ? Math.min(s.values.length - 1, s.upto) : s.values.length - 1;
                    const pts: number[] = [];
                    for (let i = 0; i <= Math.floor(n); i++) {
                        const v = s.values[i];
                        if (Number.isFinite(v)) pts.push(sx0 + i * dx, tf(v));
                    }
                    const frac = n - Math.floor(n);
                    if (frac > 0 && Math.floor(n) + 1 < s.values.length) {
                        const a = tf(s.values[Math.floor(n)]);
                        const b = tf(s.values[Math.floor(n) + 1]);
                        if (Number.isFinite(a) && Number.isFinite(b)) pts.push(sx0 + (Math.floor(n) + frac) * dx, a + (b - a) * frac);
                    }
                    p.polyline(pts, {color: s.color, width: s.width ?? 2, dash: s.dash});
                    if (s.dots) for (let i = 0; i < pts.length; i += 2) p.circle(pts[i], pts[i + 1], 2.5, {fill: s.color});
                    if (pts.length >= 2) p.circle(pts[pts.length - 2], pts[pts.length - 1], 3.5, {fill: s.color});
                }
                if (xLabel) p.textPx(xLabel, p.width - 6, p.height - 6, {color: C.muted, size: 10.5, align: "right"});
                if (yLabel) p.textPx(yLabel, p.width - 6, 14, {color: C.muted, size: 10.5, align: "right"});
            }}
            onPointer={onPick ? (e) => (e.dragging || e.type === "down" ? onPick(e.x) : undefined) : undefined}
        />
    );
}
