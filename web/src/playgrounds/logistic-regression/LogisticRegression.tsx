import {useMemo, useState} from "react";
import {Button, Legend, Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {FigureLayout} from "../../components/Figure";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {Plot3D} from "../../components/Plot3D";
import {drawSurface} from "../../lib/scene3d";
import {probabilityColor} from "../../lib/colors";
import {appendRow, drawLabelled, hitPoint, removeRow} from "../../lib/data";
import {lerp, useTimeline} from "../../lib/hooks";
import {heatImage, type View} from "../../lib/painter";
import {C, CLASS_RGB, classColor, rgba} from "../../lib/theme";
import {logreg_grid, logreg_loss, logreg_train, make_two_blobs, randomSeed} from "../../lib/wasm";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// ===========================================================================
// Figure 1 — a sigmoid on a line
// ===========================================================================

const VIEW_1D: View = {xmin: -2, xmax: 2, ymin: -0.18, ymax: 1.18};
const LOSS_VIEW: View = {xmin: -2, xmax: 2, ymin: 0, ymax: 4.2};

function make1d(seed: number): Float64Array {
    const blobs = make_two_blobs(26, 1.6, 0.42, seed);
    const out = new Float64Array(26 * 2);
    for (let i = 0; i < 26; i++) {
        out[2 * i] = Math.max(-1.9, Math.min(1.9, blobs[3 * i]));
        out[2 * i + 1] = blobs[3 * i + 2];
    }
    return out;
}

function SigmoidFigure() {
    const [seed, setSeed] = useState(5);
    const data = useMemo(() => make1d(seed), [seed]);
    const [params, setParams] = useState({w: 1.2, b: 0.6});
    const [fit, setFit] = useState<Float64Array | null>(null);
    const K = fit ? fit.length / 4 - 1 : 0;
    const tl = useTimeline(K, 30);

    let w = params.w;
    let b = params.b;
    if (fit) {
        const k = Math.min(K, Math.floor(tl.t));
        const k1 = Math.min(K, k + 1);
        const s = tl.t - k;
        w = lerp(fit[4 * k], fit[4 * k1], s);
        b = lerp(fit[4 * k + 1], fit[4 * k1 + 1], s);
    }
    const [loss, acc] = logreg_loss(data, new Float64Array([w, b]), 0);

    const set = (next: {w: number; b: number}) => {
        setFit(null);
        setParams(next);
    };

    const runFit = () => {
        const traj = logreg_train(data, new Float64Array([w, b]), 1.5, 0, 300, 3);
        setParams({w: traj[traj.length - 4], b: traj[traj.length - 3]});
        setFit(traj);
        tl.reset();
        tl.play();
    };

    const jitter = (i: number) => (((i * 37) % 11) / 11 - 0.5) * 0.07;
    const threshold = Math.abs(w) > 1e-6 ? -b / w : NaN;

    return (
        <Instrument hint="Move the sliders, then let gradient descent fit them">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Sigmoid curve fit to one-dimensional labelled data"
                            view={VIEW_1D}
                            aspect={0.5}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25, {axes: false});
                                p.line(VIEW_1D.xmin, 0, VIEW_1D.xmax, 0, {color: C.axis, width: 1});
                                p.line(VIEW_1D.xmin, 1, VIEW_1D.xmax, 1, {color: C.axis, width: 1});
                                p.line(VIEW_1D.xmin, 0.5, VIEW_1D.xmax, 0.5, {color: rgba(C.muted, 0.4), width: 1, dash: [2, 5]});
                                if (Number.isFinite(threshold)) p.line(threshold, VIEW_1D.ymin, threshold, VIEW_1D.ymax, {color: C.white, width: 1.5, dash: [5, 5], alpha: 0.7});
                                const pts: number[] = [];
                                for (let i = 0; i <= 200; i++) {
                                    const x = VIEW_1D.xmin + ((VIEW_1D.xmax - VIEW_1D.xmin) * i) / 200;
                                    pts.push(x, sigmoid(w * x + b));
                                }
                                for (let i = 0; i < data.length; i += 2) {
                                    const x = data[i];
                                    const y = data[i + 1];
                                    p.line(x, y + jitter(i), x, sigmoid(w * x + b), {color: rgba(C.red, 0.45), width: 1.2});
                                }
                                p.polyline(pts, {color: C.yellow, width: 3});
                                for (let i = 0; i < data.length; i += 2) p.point(data[i], data[i + 1] + jitter(i), classColor(data[i + 1]), {r: 5});
                                p.textPx("p = 1", 8, p.sy(1) - 6, {color: C.muted, size: 11, font: "mono"});
                                p.textPx("p = 0", 8, p.sy(0) + 14, {color: C.muted, size: 11, font: "mono"});
                            }}
                        />
                        <div className="panel-label" style={{marginTop: 4}}>
                            <span>Each point's penalty −log p(correct label)</span>
                        </div>
                        <Plot
                            label="Per-point cross-entropy penalties"
                            view={LOSS_VIEW}
                            height={90}
                            draw={(p) => {
                                p.clear();
                                for (let i = 0; i < data.length; i += 2) {
                                    const x = data[i];
                                    const q = sigmoid(w * x + b);
                                    const pc = data[i + 1] > 0.5 ? q : 1 - q;
                                    const l = Math.min(4.1, -Math.log(Math.max(pc, 1e-9)));
                                    const half = 0.025;
                                    p.polyline([x - half, 0, x + half, 0, x + half, l, x - half, l], {close: true, fill: classColor(data[i + 1]), fillAlpha: 0.85, color: "none"});
                                }
                            }}
                        />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label={<>weight <M>{"\\yel{w}"}</M> (steepness)</>} value={w} min={-6} max={14} onChange={(v) => set({w: v, b})} color={C.yellow} />
                        <Slider label={<>bias <M>{"b"}</M> (shift)</>} value={b} min={-8} max={8} onChange={(v) => set({w, b: v})} color={C.text2} />
                        <Stats>
                            <Stat label="mean loss" value={loss.toFixed(3)} color={C.red} />
                            <Stat label="accuracy" value={`${Math.round(acc * 100)}%`} color={C.green} />
                        </Stats>
                        <div className="control-row">
                            <Button variant="primary" onClick={runFit}>
                                Fit by gradient descent
                            </Button>
                            <Button onClick={() => (setSeed(randomSeed()), setFit(null))}>New data</Button>
                        </div>
                        <Legend
                            items={[
                                {color: C.yellow, label: "σ(wx + b)", shape: "line"},
                                {color: C.white, label: "threshold p = ½", shape: "dash"},
                            ]}
                        />
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Figure 2 — logistic regression in 2D
// ===========================================================================

const VIEW: View = {xmin: -1.4, xmax: 1.4, ymin: -1.4, ymax: 1.4};
const G = 120;
const RECORD = 2;
const STEPS = 1600;

function PlaneFigure(props: {sep?: number; l2?: number}) {
    const [seed, setSeed] = useState(2);
    const [sep, setSep] = useState(props.sep ?? 1.1);
    const [l2, setL2] = useState(props.l2 ?? 0);
    const [extra, setExtra] = useState<Float64Array>(new Float64Array());
    const [brush, setBrush] = useState<0 | 1>(0);
    const base = useMemo(() => make_two_blobs(120, sep, 0.24, seed), [sep, seed]);
    const data = useMemo(() => {
        const out = new Float64Array(base.length + extra.length);
        out.set(base);
        out.set(extra, base.length);
        return out;
    }, [base, extra]);

    const traj = useMemo(() => logreg_train(data, new Float64Array([0, 0, 0]), 2, l2, STEPS, RECORD), [data, l2]);
    const K = traj.length / 5 - 1;
    const tl = useTimeline(K, 60, {autoplay: true, runKey: traj});
    const k = Math.min(K, Math.floor(tl.t));
    const k1 = Math.min(K, k + 1);
    const s = tl.t - k;
    const w1 = lerp(traj[5 * k], traj[5 * k1], s);
    const w2 = lerp(traj[5 * k + 1], traj[5 * k1 + 1], s);
    const b = lerp(traj[5 * k + 2], traj[5 * k1 + 2], s);
    const loss = traj[5 * k + 3];
    const acc = traj[5 * k + 4];
    const norm = Math.hypot(w1, w2);

    const img = useMemo(() => heatImage(logreg_grid(w1, w2, b, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G), G, G, (p) => probabilityColor(p, 0.3)), [w1, w2, b]);
    const losses = useMemo(() => {
        const out = new Float64Array(K + 1);
        for (let i = 0; i <= K; i++) out[i] = traj[5 * i + 3];
        return out;
    }, [traj, K]);
    const norms = useMemo(() => {
        const out = new Float64Array(K + 1);
        for (let i = 0; i <= K; i++) out[i] = Math.hypot(traj[5 * i], traj[5 * i + 1]);
        return out;
    }, [traj, K]);

    const onPointer = (e: PlotPointer) => {
        if (e.type !== "down") return;
        const nBase = base.length / 3;
        const hit = hitPoint(data, 3, e.x, e.y, e.kx, e.ky);
        if (hit >= nBase) setExtra(removeRow(extra, 3, hit - nBase));
        else if (hit < 0) setExtra(appendRow(extra, [e.x, e.y, e.shift || e.button === 2 ? 1 - brush : brush]));
    };

    const levels = [0.1, 0.25, 0.5, 0.75, 0.9];

    return (
        <Instrument hint="Click to add points · click an added point to remove it">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Logistic regression probability map"
                            view={VIEW}
                            maxHeight={580}
                            onPointer={onPointer}
                            draw={(p) => {
                                p.clear();
                                p.image(img, VIEW);
                                p.grid(0.25);
                                if (norm > 1e-6) {
                                    // Only label the level lines when they are far enough apart to read.
                                    const labelable = (Math.log(3) / norm) * p.kx > 30;
                                    for (const q of levels) {
                                        const z = Math.log(q / (1 - q));
                                        const isHalf = q === 0.5;
                                        p.infiniteLine(w1, w2, b - z, {color: isHalf ? C.white : rgba(C.white, 0.35), width: isHalf ? 2.5 : 1.2, dash: isHalf ? undefined : [4, 5]});
                                        // Label each level line near the right edge of the view.
                                        const seg = p.clipLine(w1, w2, b - z);
                                        if (seg && !isHalf && labelable) {
                                            const [x1, y1, x2, y2] = seg;
                                            const t = 0.85;
                                            p.text(q.toFixed(2), x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, {color: C.text2, size: 10.5, font: "mono", align: "center", baseline: "middle"});
                                        }
                                    }
                                    // w, drawn from the point of the boundary nearest the origin.
                                    const fx = (-b * w1) / (norm * norm);
                                    const fy = (-b * w2) / (norm * norm);
                                    const L = Math.min(0.9, 0.18 + norm * 0.06);
                                    p.arrow(fx, fy, fx + (w1 / norm) * L, fy + (w2 / norm) * L, {color: C.yellow, width: 3, head: 12});
                                }
                                drawLabelled(p, data, {r: 4.2});
                            }}
                        />
                        <Player timeline={tl} speeds={[20, 60, 200]} status={`gradient descent step ${k * RECORD} of ${STEPS}`} />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label="Class separation" value={sep} min={0} max={2.4} onChange={(v) => (setSep(v), tl.reset())} />
                        <Slider label={<>L2 penalty <M>{"\\lambda"}</M></>} value={l2} min={0} max={0.05} step={0.001} onChange={(v) => (setL2(v), tl.reset())} color={C.purple} format={(v) => v.toFixed(3)} />
                        <Stats>
                            <Stat label="loss" value={loss.toFixed(3)} color={C.red} />
                            <Stat label="accuracy" value={`${Math.round(acc * 100)}%`} color={C.green} />
                            <Stat label={<M>{"\\lVert\\vw\\rVert"}</M>} value={norm.toFixed(2)} color={C.yellow} />
                        </Stats>
                        <div className="panel-label">
                            <span>Loss</span>
                        </div>
                        <LineChart label="Cross-entropy loss over training" series={[{values: losses, color: C.red, upto: tl.t, dx: RECORD}]} height={80} yMin={0} />
                        <div className="panel-label">
                            <span>Weight norm</span>
                        </div>
                        <LineChart label="Weight norm over training" series={[{values: norms, color: C.yellow, upto: tl.t, dx: RECORD}]} height={80} yMin={0} />
                        <Segmented
                            label="Clicking adds"
                            value={brush}
                            onChange={setBrush}
                            options={[
                                {value: 1, label: <span className="c-blue">● class 1</span>},
                                {value: 0, label: <span className="c-orange">● class 0</span>},
                            ]}
                        />
                        <div className="control-row">
                            <Button onClick={() => (setSeed(randomSeed()), setExtra(new Float64Array()), tl.reset())}>New data</Button>
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// The sigmoid as a 3D sheet
// ===========================================================================

const NS = 34;
const SPAN = 1.3;

function Sigmoid3D(props: {sep?: number}) {
    const [sep, setSep] = useState(props.sep ?? 1.2);
    const [l2, setL2] = useState(0);
    const data = useMemo(() => make_two_blobs(90, sep, 0.24, 6), [sep]);
    const traj = useMemo(() => logreg_train(data, new Float64Array([0, 0, 0]), 2, l2, 900, 6), [data, l2]);
    const K = traj.length / 5 - 1;
    const tl = useTimeline(K, 1, {autoplay: true, runKey: traj, duration: 6});
    const k = Math.min(K, Math.floor(tl.t));
    const k1 = Math.min(K, k + 1);
    const s = tl.t - k;
    const w1 = lerp(traj[5 * k], traj[5 * k1], s);
    const w2 = lerp(traj[5 * k + 1], traj[5 * k1 + 1], s);
    const b = lerp(traj[5 * k + 2], traj[5 * k1 + 2], s);
    const heights = useMemo(() => {
        const out = new Float64Array(NS * NS);
        for (let j = 0; j < NS; j++)
            for (let i = 0; i < NS; i++) {
                const x = (-1 + (2 * i) / (NS - 1)) * SPAN;
                const y = (1 - (2 * j) / (NS - 1)) * SPAN;
                out[j * NS + i] = 0.9 / (1 + Math.exp(-(w1 * x + w2 * y + b))) - 0.45;
            }
        return out;
    }, [w1, w2, b]);
    const blend = (z: number): [number, number, number] => {
        const t = (z + 0.45) / 0.9;
        const a = CLASS_RGB[0];
        const c = CLASS_RGB[1];
        return [a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t, a[2] + (c[2] - a[2]) * t];
    };

    return (
        <Instrument hint="Height = predicted probability of blue · drag to orbit">
            <FigureLayout
                main={
                    <>
                        <Plot3D
                            label="Logistic regression's sigmoid surface in 3D"
                            initial={{pitch: 0.4, yaw: 0.9}}
                            draw={(sc) => {
                                drawSurface(sc, heights, NS, blend, {alpha: 0.85, wire: 4});
                                for (let i = 0; i < data.length; i += 3) sc.point([data[i] / SPAN, data[i + 1] / SPAN, data[i + 2] > 0.5 ? 0.47 : -0.47], classColor(data[i + 2]), 3.8, {bias: 0.01});
                            }}
                        />
                        <Player timeline={tl} speeds={[0.5, 1, 3]} status={`Step ${k * 6}: ‖w‖ = ${Math.hypot(w1, w2).toFixed(2)}, so the ramp gets steeper as the model grows more confident`} />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label="Class separation" value={sep} min={0.2} max={2.4} onChange={setSep} />
                        <Slider label={<>L2 penalty <M>{"\\lambda"}</M></>} value={l2} min={0} max={0.05} step={0.001} onChange={setL2} color={C.purple} format={(v) => v.toFixed(3)} />
                        <p className="note">
                            Logistic regression is a single smooth step, <M>{"\\sigma(\\vw\\cdot\\vx + b)"}</M>, stretched over the plane. Training tilts it
                            to face the right way, then steepens it. On separable data it keeps steepening toward a cliff unless L2 holds it back.
                        </p>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Playground
// ===========================================================================

const SCENARIOS: Scenario[] = [
    {
        id: "overlap",
        title: "Overlapping classes",
        prompt: <>The boundary settles where the classes meet, and the dashed probability bands show how confident the model is on either side.</>,
        render: () => <PlaneFigure />,
    },
    {
        id: "separable",
        title: "Perfectly separable",
        prompt: <>Nothing overlaps. Play it and watch ‖w‖: it never stops growing, because more confidence always lowers the loss. The bands squeeze into a hard step.</>,
        render: () => <PlaneFigure sep={2.4} />,
    },
    {
        id: "l2",
        title: "Rein it in with L2",
        prompt: <>Same separable data with an L2 penalty on the weights. Now ‖w‖ levels off and the probabilities stay soft. Slide λ to trade confidence for caution.</>,
        render: () => <PlaneFigure sep={2.4} l2={0.01} />,
    },
    {
        id: "3d",
        title: "The sigmoid in 3D",
        prompt: <>The model's probability as a sheet over the plane. Watch gradient descent tilt the S-shaped ramp to face the right way, then steepen it as confidence grows.</>,
        render: () => <Sigmoid3D />,
    },
    {
        id: "sigmoid",
        title: "One dimension",
        prompt: <>Shape the sigmoid with w (steepness) and b (shift). The bars below are each point's penalty. Make the curve steep and slightly misplaced to see the confident mistakes explode.</>,
        render: () => <SigmoidFigure />,
    },
];

export default function LogisticRegression() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>Keep the perceptron's linear score, but squash it into a probability:</li>
                    </ul>
                    <MathBlock>{"P(y = 1 \\mid \\vx) = \\sigma(\\vw\\cdot\\vx + b), \\qquad \\sigma(z) = \\frac{1}{1 + e^{-z}}"}</MathBlock>
                    <ul>
                        <li>
                            Train by minimizing cross-entropy, <M>{"-\\frac1n\\sum_i \\big[y_i \\log p_i + (1-y_i)\\log(1-p_i)\\big]"}</M>, which charges{" "}
                            <M>{"-\\log"}</M> of the probability given to the true label.
                        </li>
                        <li>
                            The gradient is just <M>{"\\frac1n\\sum_i (p_i - y_i)\\,\\vx_i"}</M>, a soft, always-on version of the perceptron update. The
                            loss is convex, so there are no local minima.
                        </li>
                        <li>
                            The score is the log-odds. <M>{"\\lVert\\vw\\rVert"}</M> sets how sharp the transition is. On separable data it grows without
                            bound unless you add <M>{"\\tfrac\\lambda2\\lVert\\vw\\rVert^2"}</M>.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
