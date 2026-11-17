import {useMemo, useState} from "react";
import {Link} from "react-router";
import {Button, Legend, Meter, Player, Segmented, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {Callout, Figure, FigureLayout, Panel} from "../../components/Figure";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {appendRow, drawLabelled, hitPoint, removeRow} from "../../lib/data";
import {ease, fmt, lerp, useTimeline} from "../../lib/hooks";
import type {Painter, View} from "../../lib/painter";
import {C, classColor, rgba} from "../../lib/theme";
import {make_linear, perceptron_train, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.25, xmax: 1.25, ymin: -1.25, ymax: 1.25};
const STRIDE = 5; // perceptron history stride: [w1, w2, b, index, epoch]

/** A random "true" separating line with unit normal. */
function randomLine(seed: number, throughOrigin: boolean) {
    const r = Math.sin(seed * 12.9898) * 43758.5453;
    const frac = r - Math.floor(r);
    const angle = frac * Math.PI * 2;
    const b = throughOrigin ? 0 : (Math.cos(seed) * 0.5) * 0.6;
    return {w1: Math.cos(angle), w2: Math.sin(angle), b};
}

function makeData(seed: number, n: number, margin: number, flip: number, throughOrigin: boolean) {
    const t = randomLine(seed, throughOrigin);
    return {data: make_linear(n, t.w1, t.w2, t.b, margin, flip, seed), truth: t};
}

/** Shade both half-planes of `w·x + b` and draw the boundary. */
function drawClassifier(p: Painter, w1: number, w2: number, b: number, opts: {alpha?: number; lineAlpha?: number} = {}) {
    if (Math.hypot(w1, w2) < 1e-9) return;
    p.halfPlane(w1, w2, b, C.blue, opts.alpha ?? 0.1);
    p.halfPlane(-w1, -w2, -b, C.orange, opts.alpha ?? 0.08);
    p.infiniteLine(w1, w2, b, {color: C.white, width: 2.5, alpha: opts.lineAlpha ?? 1});
}

function isMistake(w1: number, w2: number, b: number, x: number, y: number, c: number) {
    const s = w1 * x + w2 * y + b;
    return (c > 0.5 ? 1 : -1) * s <= 0;
}

// ===========================================================================
// Figure 1 — anatomy of a linear classifier
// ===========================================================================

function AnatomyFigure() {
    const [seed, setSeed] = useState(7);
    const {data} = useMemo(() => makeData(seed, 44, 0.04, 0, false), [seed]);
    const [w, setW] = useState({w1: 0.55, w2: 0.75});
    const [b, setB] = useState(-0.15);
    const [drag, setDrag] = useState<"w" | "line" | null>(null);
    const [hover, setHover] = useState<{x: number; y: number} | null>(null);

    const norm = Math.hypot(w.w1, w.w2);
    let mistakes = 0;
    for (let i = 0; i < data.length; i += 3) if (isMistake(w.w1, w.w2, b, data[i], data[i + 1], data[i + 2])) mistakes++;

    const onPointer = (e: PlotPointer) => {
        const nearTip = Math.hypot((e.x - w.w1) * e.kx, (e.y - w.w2) * e.ky) < 16;
        const lineDist = (Math.abs(w.w1 * e.x + w.w2 * e.y + b) / Math.max(norm, 1e-9)) * e.kx;
        if (e.type === "down") setDrag(nearTip ? "w" : lineDist < 10 ? "line" : null);
        if (e.type === "up") setDrag(null);
        if (e.type === "leave") setHover(null);
        if (e.type === "move" || e.type === "down") {
            if (drag === "w" || (e.type === "down" && nearTip)) {
                const x = Math.max(-1.15, Math.min(1.15, e.x));
                const y = Math.max(-1.15, Math.min(1.15, e.y));
                if (Math.hypot(x, y) > 0.08) setW({w1: x, w2: y});
            } else if (drag === "line" || (e.type === "down" && lineDist < 10)) {
                setB(Math.max(-1.5, Math.min(1.5, -(w.w1 * e.x + w.w2 * e.y))));
            }
            setHover(drag ? null : {x: e.x, y: e.y});
        }
    };

    const hoverTip = hover && Math.hypot((hover.x - w.w1) * 100, (hover.y - w.w2) * 100) < 16;

    return (
        <Figure n={1} title="Anatomy of a linear classifier" hint="Drag the yellow arrow or the white line">
            <FigureLayout
                main={
                    <Plot
                        label="Linear classifier with a draggable weight vector"
                        view={VIEW}
                        maxHeight={560}
                        cursor={drag ? "grabbing" : hoverTip ? "grab" : "crosshair"}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            drawClassifier(p, w.w1, w.w2, b);
                            p.grid(0.25);
                            // Offset of the line from the origin, along w.
                            const d = -b / norm;
                            const fx = (d * w.w1) / norm;
                            const fy = (d * w.w2) / norm;
                            p.line(0, 0, fx, fy, {color: C.muted, dash: [4, 4], width: 1.5});
                            if (Math.abs(d) > 0.08)
                                p.text(`−b/‖w‖ = ${d.toFixed(2)}`, fx / 2 - (0.12 * w.w2) / norm, fy / 2 + (0.12 * w.w1) / norm, {color: C.muted, size: 11.5, align: "center", font: "mono"});
                            drawLabelled(p, data, {
                                ring: (k) => (isMistake(w.w1, w.w2, b, data[3 * k], data[3 * k + 1], data[3 * k + 2]) ? C.red : undefined),
                            });
                            // Hover probe: perpendicular to the boundary and the score.
                            if (hover && !drag) {
                                const s = w.w1 * hover.x + w.w2 * hover.y + b;
                                const t = s / (norm * norm);
                                const px = hover.x - t * w.w1;
                                const py = hover.y - t * w.w2;
                                p.line(hover.x, hover.y, px, py, {color: s > 0 ? C.blue : C.orange, width: 1.5, dash: [3, 3]});
                                p.circle(hover.x, hover.y, 3, {fill: C.white});
                                p.text(`s = ${s.toFixed(2)}`, hover.x, hover.y, {color: s > 0 ? C.blue : C.orange, dx: 10, dy: -8, font: "mono", size: 12});
                            }
                            p.arrow(0, 0, w.w1, w.w2, {color: C.yellow, width: 3.5, head: 14});
                            p.circle(w.w1, w.w2, 7, {stroke: C.yellow, width: 2, alpha: 0.5});
                            p.text("w", w.w1, w.w2, {color: C.yellow, size: 17, font: "serif", weight: 600, dx: 10, dy: 4});
                        }}
                    />
                }
                side={
                    <div className="controls">
                        <Slider label={<M>{"\\yel{w_1}"}</M>} value={w.w1} min={-1.15} max={1.15} onChange={(v) => setW({...w, w1: v})} color={C.yellow} />
                        <Slider label={<M>{"\\yel{w_2}"}</M>} value={w.w2} min={-1.15} max={1.15} onChange={(v) => setW({...w, w2: v})} color={C.yellow} />
                        <Slider label={<M>{"b"}</M>} value={b} min={-1.5} max={1.5} onChange={setB} color={C.text2} />
                        <Stats>
                            <Stat label="Mistakes" value={`${mistakes} / ${data.length / 3}`} color={mistakes ? C.red : C.green} />
                            <Stat label={<M>{"\\lVert\\vw\\rVert"}</M>} value={norm.toFixed(2)} color={C.yellow} />
                        </Stats>
                        <Legend
                            items={[
                                {color: C.blue, label: "class +1"},
                                {color: C.orange, label: "class −1"},
                                {color: C.red, label: "misclassified", shape: "dash"},
                            ]}
                        />
                        <p className="note">
                            Hover anywhere to read the score <M>{"s(\\vx)"}</M>. Its sign is the prediction; its size grows with the distance to the line.
                        </p>
                        <div className="control-row">
                            <Button onClick={() => setSeed(randomSeed())}>New data</Button>
                        </div>
                    </div>
                }
            />
        </Figure>
    );
}

// ===========================================================================
// Figure 2 — learning from mistakes, in data space and weight space
// ===========================================================================

function wrapAngle(a: number) {
    return Math.atan2(Math.sin(a), Math.cos(a));
}

function UpdatesFigure() {
    const [seed, setSeed] = useState(3);
    const {data, truth} = useMemo(() => makeData(seed, 16, 0.12, 0, true), [seed]);
    const [w0Override, setW0] = useState<{w1: number; w2: number} | null>(null);
    const [draggingW0, setDraggingW0] = useState(false);
    // Default start: pointing well away from a correct answer, so there is something to learn.
    const w0 = useMemo(() => {
        if (w0Override) return w0Override;
        const a = Math.atan2(truth.w2, truth.w1) + 2.4;
        return {w1: 0.95 * Math.cos(a), w2: 0.95 * Math.sin(a)};
    }, [w0Override, truth]);

    const run = useMemo(() => {
        const r = perceptron_train(data, w0.w1, w0.w2, 0, false, 100, 300);
        const out = {history: r.history, converged: r.converged, epochs: r.epochs};
        r.free();
        return out;
    }, [data, w0]);
    const K = run.history.length / STRIDE - 1;
    const tl = useTimeline(K, 1);

    const H = run.history;
    const wAt = (k: number) => ({w1: H[k * STRIDE], w2: H[k * STRIDE + 1]});
    const u = Math.min(K, Math.max(0, Math.ceil(tl.t - 1e-9)));
    const f = u === 0 ? 0 : 1 - (u - tl.t);
    const phaseA = ease(f / 0.35);
    const phaseB = ease((f - 0.35) / 0.65);
    const from = wAt(Math.max(0, u - 1));
    const to = wAt(u);
    const cur = u === 0 ? from : {w1: lerp(from.w1, to.w1, phaseB), w2: lerp(from.w2, to.w2, phaseB)};
    const active = u > 0 ? H[u * STRIDE + 3] : -1;
    const ax = active >= 0 ? data[3 * active] : 0;
    const ay = active >= 0 ? data[3 * active + 1] : 0;
    const yv = active >= 0 ? (data[3 * active + 2] > 0.5 ? 1 : -1) : 1;

    // Stable weight-space view that contains the whole trajectory.
    const wView = useMemo<View>(() => {
        let m = 1.0;
        for (let k = 0; k <= K; k++) m = Math.max(m, Math.abs(H[k * STRIDE]), Math.abs(H[k * STRIDE + 1]));
        for (let i = 0; i < data.length; i += 3) m = Math.max(m, Math.abs(data[i]), Math.abs(data[i + 1]));
        m *= 1.15;
        return {xmin: -m, xmax: m, ymin: -m, ymax: m};
    }, [H, K, data]);

    // Feasible wedge: all directions within 90° of every y·x.
    const wedge = useMemo(() => {
        const ref = Math.atan2(truth.w2, truth.w1);
        let lo = -Infinity;
        let hi = Infinity;
        for (let i = 0; i < data.length; i += 3) {
            const y = data[i + 2] > 0.5 ? 1 : -1;
            const d = wrapAngle(Math.atan2(y * data[i + 1], y * data[i]) - ref);
            lo = Math.max(lo, d - Math.PI / 2);
            hi = Math.min(hi, d + Math.PI / 2);
        }
        return {lo: ref + lo, hi: ref + hi};
    }, [data, truth]);

    const mistakesNow = (() => {
        let m = 0;
        for (let i = 0; i < data.length; i += 3) if (isMistake(cur.w1, cur.w2, 0, data[i], data[i + 1], data[i + 2])) m++;
        return m;
    })();

    const curNorm = Math.hypot(cur.w1, cur.w2);
    const status =
        u === 0 ? (
            <>
                Starting from <M>{`\\vw_0 = (${from.w1.toFixed(2)}, ${from.w2.toFixed(2)})`}</M>, which misclassifies {mistakesNow} of {data.length / 3} points. Press
                play or step.
            </>
        ) : (
            <>
                <b>Update {u}</b> of {K}: point #{active + 1} ({yv > 0 ? <span className="c-blue">y = +1</span> : <span className="c-orange">y = −1</span>}) had{" "}
                <M>{`\\vw\\cdot\\vx = ${(from.w1 * ax + from.w2 * ay).toFixed(2)}`}</M>, the wrong sign, so <M>{yv > 0 ? "\\vw \\leftarrow \\vw + \\vx" : "\\vw \\leftarrow \\vw - \\vx"}</M>.
                {u === K && run.converged && <span className="c-green"> Every point is now correct, so training stops.</span>}
            </>
        );

    const onWeightPointer = (e: PlotPointer) => {
        const nearW0 = Math.hypot((e.x - w0.w1) * e.kx, (e.y - w0.w2) * e.ky) < 14;
        if (e.type === "down" && nearW0 && tl.t === 0) setDraggingW0(true);
        if (e.type === "up") setDraggingW0(false);
        if (draggingW0 && e.type === "move") {
            setW0({w1: e.x, w2: e.y});
            tl.reset();
        }
    };

    const pointRing = (k: number) => (k === active && f > 0 ? C.white : isMistake(cur.w1, cur.w2, 0, data[3 * k], data[3 * k + 1], data[3 * k + 2]) ? C.red : undefined);

    return (
        <Figure n={2} title="Learning from mistakes: two views of the same update" hint="Drag the starting 𝐰₀ (at step 0) in weight space">
            <div className="duo">
                <Panel label="Data space" right={`${mistakesNow} mistakes`}>
                    <Plot
                        label="Data space: points and the perceptron boundary"
                        view={VIEW}
                        draw={(p) => {
                            p.clear();
                            drawClassifier(p, cur.w1, cur.w2, 0);
                            p.grid(0.25);
                            drawLabelled(p, data, {ring: pointRing});
                            if (active >= 0 && f > 0) {
                                p.arrow(0, 0, yv * ax, yv * ay, {color: classColor(yv > 0 ? 1 : 0), width: 2.5, alpha: phaseA, dash: yv < 0 ? [6, 5] : undefined});
                                p.text(yv > 0 ? "+x" : "−x", yv * ax, yv * ay, {color: classColor(yv > 0 ? 1 : 0), dx: 8, dy: -8, size: 14, font: "serif"});
                            }
                            if (curNorm > 1e-9) {
                                const s = 0.75 / curNorm;
                                p.arrow(0, 0, cur.w1 * s, cur.w2 * s, {color: C.yellow, width: 3.5, head: 13});
                                p.text("direction of w", cur.w1 * s, cur.w2 * s, {color: C.yellow, size: 12, dx: 8, dy: 14});
                            }
                        }}
                    />
                </Panel>
                <Panel label="Weight space" right={`w = (${cur.w1.toFixed(2)}, ${cur.w2.toFixed(2)})`}>
                    <Plot
                        label="Weight space: the perceptron's weight vector hopping into the feasible region"
                        view={wView}
                        cursor={draggingW0 ? "grabbing" : tl.t === 0 ? "grab" : "default"}
                        onPointer={onWeightPointer}
                        draw={(p) => {
                            p.clear();
                            // Feasible wedge.
                            const R = (wView.xmax - wView.xmin) * 2;
                            if (wedge.hi > wedge.lo) {
                                const pts = [0, 0];
                                for (let i = 0; i <= 40; i++) {
                                    const a = wedge.lo + ((wedge.hi - wedge.lo) * i) / 40;
                                    pts.push(R * Math.cos(a), R * Math.sin(a));
                                }
                                p.polyline(pts, {close: true, fill: C.green, fillAlpha: 0.13, color: "none"});
                            }
                            p.grid(wView.xmax > 3 ? 1 : 0.5, {majorEvery: 2});
                            // One constraint line per point: w·x = 0.
                            for (let i = 0, k = 0; i < data.length; i += 3, k++) {
                                const on = k === active && f > 0;
                                p.infiniteLine(data[i], data[i + 1], 0, {color: on ? C.white : rgba(C.muted, 0.35), width: on ? 1.8 : 1, dash: on ? undefined : [2, 4]});
                            }
                            // Trajectory so far.
                            const trail: number[] = [];
                            for (let k = 0; k < u; k++) trail.push(H[k * STRIDE], H[k * STRIDE + 1]);
                            trail.push(cur.w1, cur.w2);
                            p.polyline(trail, {color: rgba(C.yellow, 0.45), width: 1.5});
                            for (let k = 0; k < u; k++) p.circle(H[k * STRIDE], H[k * STRIDE + 1], 3, {fill: rgba(C.yellow, 0.6)});
                            // Tip-to-tail: w_old + y·x.
                            if (active >= 0 && f > 0) {
                                p.arrow(from.w1, from.w2, from.w1 + yv * ax * phaseA, from.w2 + yv * ay * phaseA, {color: classColor(yv > 0 ? 1 : 0), width: 2.5});
                            }
                            p.arrow(0, 0, cur.w1, cur.w2, {color: C.yellow, width: 3.5, head: 13});
                            p.circle(H[0], H[1], 6, {stroke: C.yellow, width: 1.5, dash: [3, 3]});
                            p.text("𝐰₀", H[0], H[1], {color: C.yellow, dx: 9, dy: 14, size: 13});
                        }}
                    />
                </Panel>
            </div>
            <div style={{marginTop: 14, display: "grid", gap: 10}}>
                <Player timeline={tl} speeds={[0.5, 1, 2, 4]} />
                <div className="status-line">{status}</div>
                <div className="control-row">
                    <Button onClick={() => (setSeed(randomSeed()), setW0(null), tl.reset())}>New data</Button>
                    <Button
                        onClick={() => {
                            const a = Math.random() * Math.PI * 2;
                            setW0({w1: 1.1 * Math.cos(a), w2: 1.1 * Math.sin(a)});
                            tl.reset();
                        }}
                    >
                        Random start
                    </Button>
                    <Legend
                        items={[
                            {color: C.yellow, label: "weight vector 𝐰", shape: "line"},
                            {color: C.green, label: "feasible weights", shape: "square"},
                            {color: C.muted, label: "𝐰·𝐱ᵢ = 0 for each point", shape: "dash"},
                        ]}
                    />
                </div>
            </div>
        </Figure>
    );
}

// ===========================================================================
// Figure 3 — with a bias, and the convergence guarantee
// ===========================================================================

function ConvergenceFigure() {
    const [seed, setSeed] = useState(11);
    const [n, setN] = useState(100);
    const [margin, setMargin] = useState(0.03);
    const [noise, setNoise] = useState(0);
    const [extra, setExtra] = useState<Float64Array>(new Float64Array());
    const [brush, setBrush] = useState<0 | 1>(1);
    const [showTruth, setShowTruth] = useState(true);

    const base = useMemo(() => makeData(seed, n, margin, noise, false), [seed, n, margin, noise]);
    const data = useMemo(() => {
        const out = new Float64Array(base.data.length + extra.length);
        out.set(base.data);
        out.set(extra, base.data.length);
        return out;
    }, [base, extra]);

    const run = useMemo(() => {
        const r = perceptron_train(data, 0, 0, 0, true, 300, 3000);
        const out = {history: r.history, converged: r.converged, epochs: r.epochs};
        r.free();
        return out;
    }, [data]);
    const K = run.history.length / STRIDE - 1;
    const tl = useTimeline(K, 6);
    const H = run.history;

    const u = Math.min(K, Math.max(0, Math.ceil(tl.t - 1e-9)));
    const f = u === 0 ? 0 : 1 - (u - tl.t);
    const k0 = Math.max(0, u - 1);
    const e = ease(f);
    const cur = {
        w1: lerp(H[k0 * STRIDE], H[u * STRIDE], e),
        w2: lerp(H[k0 * STRIDE + 1], H[u * STRIDE + 1], e),
        b: lerp(H[k0 * STRIDE + 2], H[u * STRIDE + 2], e),
    };
    const active = u > 0 ? H[u * STRIDE + 3] : -1;
    const epoch = u > 0 ? H[u * STRIDE + 4] : 0;

    // Novikoff bound using the generating line, if it separates everything.
    const bound = useMemo(() => {
        const {w1, w2, b} = base.truth;
        const norm = Math.hypot(w1, w2, b);
        let gamma = Infinity;
        let R = 0;
        for (let i = 0; i < data.length; i += 3) {
            const y = data[i + 2] > 0.5 ? 1 : -1;
            gamma = Math.min(gamma, (y * (w1 * data[i] + w2 * data[i + 1] + b)) / norm);
            R = Math.max(R, Math.hypot(data[i], data[i + 1], 1));
        }
        return gamma > 0 ? {gamma, R, k: Math.ceil((R / gamma) ** 2)} : null;
    }, [base, data]);

    let mistakesNow = 0;
    for (let i = 0; i < data.length; i += 3) if (isMistake(cur.w1, cur.w2, cur.b, data[i], data[i + 1], data[i + 2])) mistakesNow++;

    const onPointer = (ev: PlotPointer) => {
        if (ev.type !== "down") return;
        const nBase = base.data.length / 3;
        const hit = hitPoint(data, 3, ev.x, ev.y, ev.kx, ev.ky);
        if (hit >= nBase) setExtra(removeRow(extra, 3, hit - nBase));
        else if (hit < 0) setExtra(appendRow(extra, [ev.x, ev.y, ev.shift || ev.button === 2 ? 1 - brush : brush]));
        tl.reset();
    };

    const done = tl.atEnd && K > 0;

    return (
        <Figure n={3} title="With a bias: does it always stop?" hint="Click to add points · click an added point to remove it">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Perceptron with bias training on generated data"
                            view={VIEW}
                            maxHeight={560}
                            onPointer={onPointer}
                            draw={(p) => {
                                p.clear();
                                drawClassifier(p, cur.w1, cur.w2, cur.b);
                                p.grid(0.25);
                                if (showTruth && noise === 0) {
                                    const {w1, w2, b} = base.truth;
                                    p.infiniteLine(w1, w2, b, {color: C.green, width: 1.5, dash: [6, 6], alpha: 0.8});
                                    p.infiniteLine(w1, w2, b - margin, {color: C.green, width: 1, dash: [2, 5], alpha: 0.5});
                                    p.infiniteLine(w1, w2, b + margin, {color: C.green, width: 1, dash: [2, 5], alpha: 0.5});
                                }
                                drawLabelled(p, data, {
                                    ring: (k) =>
                                        u === 0
                                            ? undefined
                                            : k === active && f > 0 && f < 1
                                              ? C.white
                                              : isMistake(cur.w1, cur.w2, cur.b, data[3 * k], data[3 * k + 1], data[3 * k + 2])
                                                ? C.red
                                                : undefined,
                                });
                                if (u === 0) p.textPx("𝐰 = 0: no boundary yet. Press play.", p.width / 2, 26, {align: "center", color: C.text2, size: 13});
                            }}
                        />
                        <Player
                            timeline={tl}
                            speeds={[2, 6, 20, 60]}
                            status={
                                done ? (
                                    run.converged ? (
                                        <span className="c-green">
                                            Converged: {K} updates over {run.epochs} {run.epochs === 1 ? "pass" : "passes"} through the data.
                                        </span>
                                    ) : (
                                        <span className="c-red">
                                            Never settled. After {run.epochs} passes there was still a mistake in every pass, because no line separates these points.
                                        </span>
                                    )
                                ) : u === 0 ? (
                                    <>Starting from 𝐰 = 0, b = 0. The first point visited is always a “mistake”.</>
                                ) : (
                                    <>
                                        Update {u} / {K} · pass {epoch} · {mistakesNow} points currently misclassified
                                    </>
                                )
                            }
                        />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label="Points" value={n} min={10} max={200} step={1} onChange={(v) => (setN(v), tl.reset())} format={(v) => v.toFixed(0)} />
                        <Slider label="Margin (gap around the true line)" value={margin} min={0.005} max={0.3} onChange={(v) => (setMargin(v), tl.reset())} color={C.green} />
                        <Slider
                            label="Label noise"
                            value={noise}
                            min={0}
                            max={0.2}
                            onChange={(v) => (setNoise(v), tl.reset())}
                            format={(v) => `${Math.round(v * 100)}%`}
                            color={C.red}
                        />
                        <Segmented
                            label="Clicking adds"
                            value={brush}
                            onChange={setBrush}
                            options={[
                                {value: 1, label: <span className="c-blue">● class +1</span>},
                                {value: 0, label: <span className="c-orange">● class −1</span>},
                            ]}
                        />
                        <Toggle label="Show the generating line & margin" checked={showTruth} onChange={setShowTruth} />
                        {bound ? (
                            <Meter
                                label="Mistakes vs. guarantee"
                                right={`${u} ≤ ${bound.k}`}
                                value={u}
                                max={Math.max(bound.k, 1)}
                                color={C.yellow}
                            />
                        ) : (
                            <p className="note note-muted">The data isn’t separable by the generating line, so the guarantee doesn’t apply.</p>
                        )}
                        <Stats>
                            <Stat label="Updates" value={u} color={C.yellow} />
                            <Stat label={<M>R</M>} value={bound ? bound.R.toFixed(2) : "—"} />
                            <Stat label={<M>{"\\gamma"}</M>} value={bound ? fmt(bound.gamma, 3) : "—"} color={C.green} />
                        </Stats>
                        <div className="control-row">
                            <Button onClick={() => (setSeed(randomSeed()), setExtra(new Float64Array()), tl.reset())}>New data</Button>
                            {extra.length > 0 && <Button onClick={() => (setExtra(new Float64Array()), tl.reset())}>Clear added</Button>}
                        </div>
                    </div>
                }
            />
        </Figure>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

export default function Perceptron() {
    return (
        <>
            <p>
                In 1958 Frank Rosenblatt unveiled a machine that taught itself to tell shapes apart. The algorithm behind it, the <strong>perceptron</strong>,
                fits in three lines of code. It still contains every ingredient of modern machine learning: a model with adjustable parameters, a way to
                notice mistakes, and a rule for improving after each one.
            </p>
            <p>
                It also explains this site's name. The perceptron's decision boundary is a <em>hyperplane</em>: a line in 2D, a plane in 3D, and a
                flat <M>{"(d-1)"}</M>-dimensional sheet in <M>{"d"}</M> dimensions.
            </p>

            <h2 id="linear-classifier">A line that makes decisions</h2>
            <p>
                Each data point is a vector <M>{"\\vx = (x_1, x_2)"}</M> with a label <M>{"y"}</M> that is either <span className="c-blue">+1</span> or{" "}
                <span className="c-orange">−1</span>. A linear classifier gives every point a <strong>score</strong>, then predicts using only its sign:
            </p>
            <MathBlock>{"s(\\vx) = \\vw \\cdot \\vx + b = \\yel{w_1} x_1 + \\yel{w_2} x_2 + b, \\qquad \\hat y = \\operatorname{sign}\\big(s(\\vx)\\big)"}</MathBlock>
            <p>
                The boundary is the set of points that score exactly zero, <M>{"\\vw\\cdot\\vx + b = 0"}</M>. Two facts about it are worth seeing for yourself
                before reading the algebra:
            </p>
            <ul>
                <li>
                    The weight vector <M>{"\\vw"}</M> is <strong>perpendicular</strong> to the boundary, and it points toward the{" "}
                    <span className="c-blue">positive</span> side.
                </li>
                <li>
                    The bias <M>{"b"}</M> <strong>slides</strong> the boundary without turning it. The line sits a signed distance{" "}
                    <M>{"-b/\\lVert\\vw\\rVert"}</M> from the origin.
                </li>
            </ul>

            <AnatomyFigure />

            <p>
                Why perpendicular? If two points <M>{"\\vx"}</M> and <M>{"\\vx'"}</M> both lie on the boundary, subtracting their equations gives{" "}
                <M>{"\\vw\\cdot(\\vx - \\vx') = 0"}</M>. Every direction <em>along</em> the line is orthogonal to <M>{"\\vw"}</M>. The score also has a
                geometric meaning: <M>{"s(\\vx)/\\lVert\\vw\\rVert"}</M> is the signed distance from <M>{"\\vx"}</M> to the line. Points far from the
                boundary get confident scores and points near it get scores close to zero.
            </p>

            <Callout kind="try">
                Try to drive the mistake count to zero by hand. You will find yourself rotating the arrow toward the blue points and sliding the line into
                the gap. Next, we will make that process automatic.
            </Callout>

            <h2 id="mistakes">Learning from mistakes</h2>
            <p>
                The perceptron starts from any weights and visits the points one at a time. When a point is classified correctly, nothing happens. When a
                point is misclassified, which means <M>{"y\\,(\\vw\\cdot\\vx + b) \\le 0"}</M>, the weights are nudged toward that point's label:
            </p>
            <MathBlock label="the perceptron update">{"\\vw \\leftarrow \\vw + y\\,\\vx, \\qquad b \\leftarrow b + y"}</MathBlock>
            <p>
                Why should this help? Look at what the update does to the score of the point that triggered it. Ignoring the bias for a moment:
            </p>
            <MathBlock>{"y\\,(\\vw + y\\vx)\\cdot\\vx \\;=\\; y\\,\\vw\\cdot\\vx \\;+\\; \\underbrace{y^2}_{=1}\\lVert\\vx\\rVert^2 \\;=\\; y\\,\\vw\\cdot\\vx + \\lVert\\vx\\rVert^2"}</MathBlock>
            <p>
                The quantity <M>{"y\\,\\vw\\cdot\\vx"}</M> is positive exactly when the point is classified correctly, and each update raises it by{" "}
                <M>{"\\lVert\\vx\\rVert^2 > 0"}</M>. The fix may not be complete in one step, and it can disturb other points. But it always pushes in the
                right direction for the point that complained.
            </p>
            <p>
                Geometry makes this vivid. If we temporarily drop the bias so the boundary passes through the origin, then weights and data live in the same
                plane. Adding <M>{"\\vx"}</M> to <M>{"\\vw"}</M> tilts <M>{"\\vw"}</M> <em>toward</em> a misclassified positive point, and subtracting it
                tilts <M>{"\\vw"}</M> <em>away</em> from a misclassified negative one.
            </p>
            <p>
                The right panel below is <strong>weight space</strong>. Every possible classifier is a single point there. Each data point{" "}
                <M>{"\\vx_i"}</M> draws a line <M>{"\\vw\\cdot\\vx_i = 0"}</M> through weight space, and only weights on the correct side of that line
                classify the point correctly. Intersect all of those half-planes and you get the <span className="c-green">green wedge</span> of perfect
                classifiers. Training is a walk in weight space, one tip-to-tail vector addition at a time, until the walk ends inside the wedge.
            </p>

            <UpdatesFigure />

            <Callout>
                In data space, a mistake is a point on the wrong side of a line. In weight space, the same mistake is a weight vector on the wrong side of a
                line. The update <M>{"\\vw \\leftarrow \\vw + y\\vx"}</M> is a step perpendicular to that line, toward the side where the point is
                classified correctly. Seeing one picture through the other, called <em>duality</em>, comes up all over machine learning.
            </Callout>

            <h2 id="convergence">Adding the bias back, and a guarantee</h2>
            <p>
                The bias is handled with a classic trick. Append a constant <M>{"1"}</M> to every input and fold <M>{"b"}</M> into the weights:
            </p>
            <MathBlock>{"\\tilde{\\vx} = (x_1, x_2, 1), \\qquad \\tilde{\\vw} = (\\yel{w_1}, \\yel{w_2}, b) \\quad\\Longrightarrow\\quad \\tilde\\vw \\cdot \\tilde\\vx = \\vw\\cdot\\vx + b"}</MathBlock>
            <p>
                Now <M>{"\\tilde\\vw \\leftarrow \\tilde\\vw + y\\tilde\\vx"}</M> is exactly the update above, bias included. Geometrically, we lifted the
                2D data onto the plane <M>{"z = 1"}</M> in 3D and are searching for a plane through the origin. Where that plane slices <M>{"z=1"}</M>, it
                draws our line.
            </p>
            <p>
                Does the walk always end? In 1962 Albert Novikoff proved that it does whenever a perfect line exists. Suppose some unit vector{" "}
                <M>{"\\vw^\\ast"}</M> separates the data with a <span className="c-green">margin</span> <M>{"\\gamma"}</M>, meaning{" "}
                <M>{"y_i\\,\\vw^\\ast\\!\\cdot\\tilde\\vx_i \\ge \\gamma"}</M> for every point, and every point satisfies{" "}
                <M>{"\\lVert\\tilde\\vx_i\\rVert \\le R"}</M>. Then, starting from <M>{"\\vw = 0"}</M>, the perceptron makes at most
            </p>
            <MathBlock label="Novikoff, 1962">{"k \\;\\le\\; \\left(\\frac{R}{\\gamma}\\right)^{2} \\quad\\text{mistakes.}"}</MathBlock>
            <p>The proof squeezes the weight vector between two bounds after <M>{"k"}</M> mistakes:</p>
            <ul>
                <li>
                    <strong>It aligns with the answer quickly.</strong> Each update adds <M>{"y\\,\\tilde\\vx\\cdot\\vw^\\ast \\ge \\gamma"}</M> to{" "}
                    <M>{"\\vw\\cdot\\vw^\\ast"}</M>, so <M>{"\\vw\\cdot\\vw^\\ast \\ge k\\gamma"}</M>.
                </li>
                <li>
                    <strong>It grows slowly.</strong> On a mistake <M>{"\\lVert\\vw + y\\tilde\\vx\\rVert^2 = \\lVert\\vw\\rVert^2 + 2y\\,\\vw\\cdot\\tilde\\vx + \\lVert\\tilde\\vx\\rVert^2"}</M>,
                    and the middle term is <M>{"\\le 0"}</M> because the point was misclassified. So <M>{"\\lVert\\vw\\rVert^2 \\le kR^2"}</M>.
                </li>
            </ul>
            <p>
                Cauchy–Schwarz says <M>{"\\vw\\cdot\\vw^\\ast \\le \\lVert\\vw\\rVert"}</M>, so <M>{"k\\gamma \\le \\sqrt{k}\\,R"}</M>, which rearranges to{" "}
                <M>{"k \\le (R/\\gamma)^2"}</M>. Notice what does <em>not</em> appear in the bound: the number of points and the dimension. Only the
                geometry of the gap matters.
            </p>

            <ConvergenceFigure />

            <Callout kind="try">
                <ul>
                    <li>Shrink the margin and watch the number of updates climb. A thin gap is hard to find.</li>
                    <li>
                        Add a little label noise, or click an orange point into the blue cloud. Once no line can separate the data, the perceptron thrashes
                        forever. That flaw motivated <Link to="/learn/logistic-regression">logistic regression</Link> and, later, the max-margin idea behind
                        support vector machines.
                    </li>
                </ul>
            </Callout>

            <section className="takeaways">
                <h2>Takeaways</h2>
                <ul>
                    <li>
                        A linear classifier is a score <M>{"\\vw\\cdot\\vx + b"}</M>. The weights set the boundary's orientation (perpendicular to{" "}
                        <M>{"\\vw"}</M>), and the bias sets its offset.
                    </li>
                    <li>
                        The perceptron only learns from mistakes: <M>{"\\vw \\leftarrow \\vw + y\\vx"}</M>. Each update tilts the boundary toward getting
                        the offending point right.
                    </li>
                    <li>In weight space, training is a walk toward the region of perfect classifiers.</li>
                    <li>
                        If the data is separable with margin <M>{"\\gamma"}</M>, the walk ends after at most <M>{"(R/\\gamma)^2"}</M> mistakes. If it
                        isn't separable, the walk never ends.
                    </li>
                </ul>
            </section>
        </>
    );
}
