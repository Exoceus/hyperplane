import {useEffect, useMemo, useRef, useState} from "react";
import {Button, LiveControls, Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {FigureLayout, Panel} from "../../components/Figure";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot} from "../../components/Plot";
import {Plot3D} from "../../components/Plot3D";
import {drawSurface} from "../../lib/scene3d";
import {probabilityColor} from "../../lib/colors";
import {drawLabelled} from "../../lib/data";
import {ease, lerp, useAnimationFrame, useInView, useTimeline} from "../../lib/hooks";
import {heatImage, type View} from "../../lib/painter";
import {getSpeed, takeSteps} from "../../lib/speed";
import {C, CLASS_RGB, classColor, rgba, sequential, type RGB} from "../../lib/theme";
import {contour_segments, make_shape, Mlp, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};

const ACTIVATIONS = [
    {id: 0, name: "tanh", f: Math.tanh, range: [-1, 1] as const},
    {id: 1, name: "ReLU", f: (z: number) => Math.max(0, z), range: [0, 3] as const},
    {id: 2, name: "sigmoid", f: (z: number) => 1 / (1 + Math.exp(-z)), range: [0, 1] as const},
    {id: 3, name: "linear", f: (z: number) => z, range: [-3, 3] as const},
];

/** Map an activation value to a color, given the activation's natural range. */
function activationColor(a: number, act: number, scale = 1): RGB {
    if (act === 0) return probabilityColor((a / scale + 1) / 2, 0.55);
    if (act === 2) return probabilityColor(a, 0.55);
    if (act === 1) return sequential(Math.min(1, a / scale) * 0.85).map((c) => c * 0.9) as RGB;
    return probabilityColor(Math.max(0, Math.min(1, (a / scale + 1) / 2)), 0.55);
}

// ===========================================================================
// Figure 1 — a single neuron
// ===========================================================================

const N1 = 90;

function NeuronFigure() {
    const [w1, setW1] = useState(2.2);
    const [w2, setW2] = useState(1.4);
    const [b, setB] = useState(-0.3);
    const [act, setAct] = useState(0);
    const [probe, setProbe] = useState<{x: number; y: number} | null>(null);
    const A = ACTIVATIONS[act];

    const img = useMemo(() => {
        const vals = new Float64Array(N1 * N1);
        for (let j = 0; j < N1; j++)
            for (let i = 0; i < N1; i++) {
                const x = VIEW.xmin + ((i + 0.5) * (VIEW.xmax - VIEW.xmin)) / N1;
                const y = VIEW.ymax - ((j + 0.5) * (VIEW.ymax - VIEW.ymin)) / N1;
                vals[j * N1 + i] = A.f(w1 * x + w2 * y + b);
            }
        return heatImage(vals, N1, N1, (a) => activationColor(a, act, act === 1 || act === 3 ? 3 : 1));
    }, [w1, w2, b, act, A]);

    const z = probe ? w1 * probe.x + w2 * probe.y + b : null;
    const curveView: View = {xmin: -4, xmax: 4, ymin: A.range[0] - 0.3, ymax: A.range[1] + 0.3};

    return (
        <Instrument hint="Hover the plane to trace a point through the neuron">
            <FigureLayout
                main={
                    <Plot
                        label="Output of a single neuron over the input plane"
                        view={VIEW}
                        maxHeight={520}
                        onPointer={(e) => setProbe(e.type === "leave" ? null : {x: e.x, y: e.y})}
                        draw={(p) => {
                            p.clear();
                            p.image(img, VIEW);
                            p.grid(0.25);
                            p.infiniteLine(w1, w2, b, {color: C.white, width: 1.5, dash: [5, 5], alpha: 0.7});
                            const n = Math.hypot(w1, w2);
                            if (n > 1e-6) p.arrow(0, 0, (w1 / n) * 0.6, (w2 / n) * 0.6, {color: C.yellow, width: 3});
                            if (probe) {
                                p.circle(probe.x, probe.y, 5, {fill: C.white, stroke: C.canvas, width: 2});
                            }
                        }}
                    />
                }
                side={
                    <div className="controls">
                        <Segmented value={act} onChange={setAct} options={ACTIVATIONS.map((a) => ({value: a.id, label: a.name}))} />
                        <Slider label={<M>{"\\yel{w_1}"}</M>} value={w1} min={-5} max={5} onChange={setW1} color={C.yellow} />
                        <Slider label={<M>{"\\yel{w_2}"}</M>} value={w2} min={-5} max={5} onChange={setW2} color={C.yellow} />
                        <Slider label={<M>{"b"}</M>} value={b} min={-3} max={3} onChange={setB} color={C.text2} />
                        <div className="panel-label">
                            <span>activation f(z)</span>
                            <span className="panel-label-right">{z !== null ? `z = ${z.toFixed(2)} → ${A.f(z).toFixed(2)}` : "hover the plane"}</span>
                        </div>
                        <Plot
                            label="The activation function"
                            view={curveView}
                            height={120}
                            draw={(p) => {
                                p.clear();
                                p.grid(1, {majorEvery: 100});
                                const pts: number[] = [];
                                for (let i = 0; i <= 120; i++) {
                                    const x = -4 + (8 * i) / 120;
                                    pts.push(x, A.f(x));
                                }
                                p.polyline(pts, {color: C.white, width: 2});
                                if (z !== null) {
                                    const zc = Math.max(-4, Math.min(4, z));
                                    p.line(zc, curveView.ymin, zc, A.f(zc), {color: C.yellow, width: 1.5, dash: [3, 3]});
                                    p.circle(zc, Math.max(curveView.ymin, Math.min(curveView.ymax, A.f(zc))), 5, {fill: C.yellow});
                                }
                            }}
                        />
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Figure 2 — the playground
// ===========================================================================

type Shape = "circles" | "xor" | "moons" | "spiral" | "blobs";
const SHAPES: {value: Shape; label: string}[] = [
    {value: "circles", label: "Circles"},
    {value: "xor", label: "XOR"},
    {value: "moons", label: "Moons"},
    {value: "spiral", label: "Spiral"},
    {value: "blobs", label: "Blobs"},
];
const GRID_OUT = 100;
const GRID_NEURON = 30;

interface Frame {
    out: HTMLCanvasElement;
    neurons: HTMLCanvasElement[];
    params: Float64Array;
    loss: number;
    acc: number;
    steps: number;
}

function inputImage(axis: 0 | 1): HTMLCanvasElement {
    const vals = new Float64Array(GRID_NEURON * GRID_NEURON);
    for (let j = 0; j < GRID_NEURON; j++)
        for (let i = 0; i < GRID_NEURON; i++) {
            const t = axis === 0 ? (i + 0.5) / GRID_NEURON : 1 - (j + 0.5) / GRID_NEURON;
            vals[j * GRID_NEURON + i] = t;
        }
    return heatImage(vals, GRID_NEURON, GRID_NEURON, (t) => probabilityColor(t, 0.55));
}

function PlaygroundFigure(props: {shape?: Shape; layers?: number[]; act?: number; noise?: number}) {
    const [shape, setShape] = useState<Shape>(props.shape ?? "spiral");
    const [noise, setNoise] = useState(props.noise ?? 0.1);
    const [dataSeed, setDataSeed] = useState(3);
    const [layers, setLayers] = useState<number[]>(props.layers ?? [8, 8]);
    const [act, setAct] = useState(props.act ?? 0);
    const [lr, setLr] = useState(0.02);
    const [running, setRunning] = useState(true);
    const [netSeed, setNetSeed] = useState(1);
    const [frame, setFrame] = useState<Frame | null>(null);
    const [lossHistory, setLossHistory] = useState<number[]>([]);
    const [ref, visible] = useInView<HTMLDivElement>();

    const data = useMemo(() => make_shape(shape, 300, noise, dataSeed), [shape, noise, dataSeed]);
    const sizes = useMemo(() => [2, ...layers, 1], [layers]);
    const key = `${sizes.join("-")}|${act}|${netSeed}|${shape}|${noise}|${dataSeed}`;
    const sim = useRef<{net: Mlp; key: string; dirty: boolean; frameNo: number; history: number[]} | null>(null);
    const acc = useRef(0);
    const inputs = useMemo(() => [inputImage(0), inputImage(1)], []);

    useEffect(
        () => () => {
            sim.current?.net.free();
            sim.current = null;
        },
        []
    );

    useAnimationFrame(() => {
        let s = sim.current;
        if (!s || s.key !== key) {
            s?.net.free();
            s = {net: new Mlp(new Uint32Array(sizes), act, netSeed), key, dirty: true, frameNo: 0, history: []};
            sim.current = s;
        }
        // Four Adam steps per frame at 1×, scaled by the viewer's speed.
        const n = running ? takeSteps(acc, 4 * getSpeed(), 40) : 0;
        if (n > 0) {
            s.history.push(s.net.train(data, n, lr, 0, 1, 0));
            s.dirty = true;
        }
        s.frameNo++;
        if (s.dirty && (s.frameNo % 2 === 0 || !running)) {
            s.dirty = false;
            const net = s.net;
            const out = heatImage(net.predict_grid(VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, GRID_OUT, GRID_OUT), GRID_OUT, GRID_OUT, (p) => probabilityColor(p, 0.32));
            const all = net.neuron_grids(VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, GRID_NEURON, GRID_NEURON);
            const cells = GRID_NEURON * GRID_NEURON;
            const neurons: HTMLCanvasElement[] = [];
            for (let n = 0; n < all.length / cells; n++) {
                const block = all.subarray(n * cells, (n + 1) * cells);
                const isOutput = n === all.length / cells - 1;
                let scale = 1;
                if (!isOutput && (act === 1 || act === 3)) {
                    scale = 1e-6;
                    for (const v of block) scale = Math.max(scale, Math.abs(v));
                }
                neurons.push(heatImage(block, GRID_NEURON, GRID_NEURON, (a) => (isOutput ? probabilityColor(a, 0.55) : activationColor(a, act, scale))));
            }
            setFrame({out, neurons, params: net.params(), loss: net.loss(data), acc: net.accuracy(data), steps: net.steps});
            setLossHistory(s.history.slice());
        }
    }, visible);

    const setLayer = (i: number, n: number) => setLayers(layers.map((v, j) => (j === i ? Math.max(1, Math.min(8, n)) : v)));
    const reset = () => {
        setNetSeed(randomSeed());
    };
    const stepOnce = () => {
        const s = sim.current;
        if (!s || s.key !== key) return;
        s.history.push(s.net.train(data, 1, lr, 0, 1, 0));
        s.dirty = true;
    };

    // Network diagram layout in its own coordinate system.
    const diagram = useMemo(() => {
        const W = 100;
        const H = 62;
        const cols = sizes.length;
        const maxN = Math.max(...sizes);
        const cell = Math.min(7.2, (H - 4) / (maxN * 1.22));
        const xs = sizes.map((_, l) => 6 + ((W - 12) * l) / (cols - 1));
        const pos = sizes.map((n, l) => Array.from({length: n}, (_, i) => ({x: xs[l], y: H / 2 - (i - (n - 1) / 2) * cell * 1.22})));
        return {W, H, cell, pos};
    }, [sizes]);

    const status = frame ? `step ${frame.steps} · loss ${frame.loss.toFixed(3)} · accuracy ${Math.round(frame.acc * 100)}%` : "";

    return (
        <Instrument hint="Training live · change the data, architecture or activation and watch it relearn">
            <div ref={ref}>
                <FigureLayout
                    sideWidth={300}
                    main={
                        <>
                            <div className="duo">
                                <Panel label="Output" right={frame ? `${Math.round(frame.acc * 100)}% correct` : ""}>
                                    <Plot
                                        label="Network output over the plane"
                                        view={VIEW}
                                        draw={(p) => {
                                            p.clear();
                                            if (frame) p.image(frame.out, VIEW);
                                            p.grid(0.25, {axes: false});
                                            drawLabelled(p, data, {r: 3.6});
                                        }}
                                    />
                                </Panel>
                                <Panel label="Inside the network" right={`${sizes.slice(1, -1).reduce((a, b) => a + b, 0)} hidden neurons`}>
                                    <Plot
                                        label="Network diagram: each neuron shows its activation over the input plane"
                                        view={{xmin: 0, xmax: diagram.W, ymin: 0, ymax: diagram.H}}
                                        draw={(p) => {
                                            p.clear();
                                            const {pos, cell} = diagram;
                                            // The frame can lag one tick behind an architecture change.
                                            let expected = 0;
                                            for (let l = 0; l < sizes.length - 1; l++) expected += sizes[l] * sizes[l + 1] + sizes[l + 1];
                                            const fresh = frame && frame.params.length === expected;
                                            if (frame && fresh) {
                                                let offset = 0;
                                                for (let l = 0; l < sizes.length - 1; l++) {
                                                    const nin = sizes[l];
                                                    const nout = sizes[l + 1];
                                                    for (let o = 0; o < nout; o++)
                                                        for (let i = 0; i < nin; i++) {
                                                            const w = frame.params[offset + o * nin + i];
                                                            const a = pos[l][i];
                                                            const bpos = pos[l + 1][o];
                                                            p.line(a.x + cell / 2, a.y, bpos.x - cell / 2, bpos.y, {
                                                                color: w >= 0 ? C.blue : C.orange,
                                                                width: Math.min(4, 0.3 + Math.abs(w) * 0.9),
                                                                alpha: Math.min(0.85, 0.15 + Math.abs(w) * 0.3),
                                                                cap: "butt",
                                                            });
                                                        }
                                                    offset += nin * nout + nout;
                                                }
                                            }
                                            let neuron = 0;
                                            pos.forEach((col, l) =>
                                                col.forEach(({x, y}, i) => {
                                                    const img = l === 0 ? inputs[i] : fresh ? frame?.neurons[neuron++] : undefined;
                                                    const r = {xmin: x - cell / 2, xmax: x + cell / 2, ymin: y - cell / 2, ymax: y + cell / 2};
                                                    if (img) p.image(img, r);
                                                    p.polyline([r.xmin, r.ymin, r.xmax, r.ymin, r.xmax, r.ymax, r.xmin, r.ymax], {close: true, color: l === sizes.length - 1 ? C.white : C.axis, width: 1});
                                                })
                                            );
                                            p.text("x₁", pos[0][0].x, pos[0][0].y - cell / 2, {color: C.muted, size: 10, align: "center", dy: 12});
                                            p.text("x₂", pos[0][1].x, pos[0][1].y - cell / 2, {color: C.muted, size: 10, align: "center", dy: 12});
                                        }}
                                    />
                                </Panel>
                            </div>
                            <LiveControls
                                running={running}
                                onToggle={() => setRunning(!running)}
                                onStep={stepOnce}
                                stepTitle="One training step"
                                onReset={reset}
                                resetTitle="Start again from new random weights"
                                status={<span className="mono">{status}</span>}
                            />
                            <LineChart label="Training loss" series={[{values: lossHistory, color: C.red}]} height={90} yMin={0} xLabel="training →" />
                        </>
                    }
                    side={
                        <div className="controls">
                            <Segmented value={shape} onChange={setShape} options={SHAPES} />
                            <Slider label="noise" value={noise} min={0} max={0.5} onChange={setNoise} />
                            <div className="control-row">
                                <Button onClick={() => setDataSeed(randomSeed())}>New sample</Button>
                            </div>
                            <div className="side-block">
                                <span className="control-label">Hidden layers</span>
                                {layers.map((n, i) => (
                                    <div key={i} className="control-row" style={{justifyContent: "space-between"}}>
                                        <span className="note">Layer {i + 1}</span>
                                        <span className="control-row" style={{gap: 4}}>
                                            <Button variant="subtle" onClick={() => setLayer(i, n - 1)} disabled={n <= 1} title="Remove a neuron">
                                                −
                                            </Button>
                                            <span className="mono" style={{width: 92, textAlign: "center", whiteSpace: "nowrap"}}>
                                                {n} neuron{n === 1 ? "" : "s"}
                                            </span>
                                            <Button variant="subtle" onClick={() => setLayer(i, n + 1)} disabled={n >= 8} title="Add a neuron">
                                                +
                                            </Button>
                                        </span>
                                    </div>
                                ))}
                                <div className="control-row">
                                    <Button onClick={() => setLayers([...layers, 4])} disabled={layers.length >= 4}>
                                        + layer
                                    </Button>
                                    <Button onClick={() => setLayers(layers.slice(0, -1))} disabled={layers.length <= 1}>
                                        − layer
                                    </Button>
                                </div>
                            </div>
                            <Segmented label="activation" value={act} onChange={setAct} options={ACTIVATIONS.map((a) => ({value: a.id, label: a.name}))} />
                            <Slider label="learning rate (Adam)" value={lr} min={0.001} max={0.2} log onChange={setLr} color={C.yellow} format={(v) => v.toFixed(3)} />
                            {frame && (
                                <Stats>
                                    <Stat label="parameters" value={frame.params.length} />
                                    <Stat label="loss" value={frame.loss.toFixed(3)} color={C.red} />
                                </Stats>
                            )}
                        </div>
                    }
                />
            </div>
        </Instrument>
    );
}

// ===========================================================================
// Figure 3 — networks bend space
// ===========================================================================

const WARP_VIEW: View = {xmin: -1.35, xmax: 1.35, ymin: -1.35, ymax: 1.35};
const WARP_SIZES = [2, 2, 2, 2, 1];
const HIDDEN = WARP_SIZES.length - 2; // 3 hidden layers of width 2
const LINES = 17;
const SAMPLES = 60;
const POPULATION = 12;

interface Candidate {
    net: Mlp;
    loss: number;
    acc: number;
    /** How two-dimensional the thinnest hidden representation is (0 = flat line, 1 = round). */
    flatness: number;
    finished: boolean;
}

/** sqrt(λ₂/λ₁) of the covariance of stride-2 points. */
function aspectRatio(pts: Float64Array): number {
    const n = pts.length / 2;
    let mx = 0;
    let my = 0;
    for (let i = 0; i < pts.length; i += 2) {
        mx += pts[i];
        my += pts[i + 1];
    }
    mx /= n;
    my /= n;
    let a = 0;
    let b = 0;
    let c = 0;
    for (let i = 0; i < pts.length; i += 2) {
        const dx = pts[i] - mx;
        const dy = pts[i + 1] - my;
        a += dx * dx;
        b += dx * dy;
        c += dy * dy;
    }
    const tr = a + c;
    const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - (a * c - b * b)));
    return Math.sqrt(Math.max(0, tr / 2 - disc) / Math.max(1e-300, tr / 2 + disc));
}

function gridLines(): Float64Array {
    const out = new Float64Array(LINES * 2 * SAMPLES * 2);
    let k = 0;
    for (let dir = 0; dir < 2; dir++)
        for (let i = 0; i < LINES; i++) {
            const c = -1.2 + (2.4 * i) / (LINES - 1);
            for (let s = 0; s < SAMPLES; s++) {
                const t = -1.2 + (2.4 * s) / (SAMPLES - 1);
                out[k++] = dir === 0 ? t : c;
                out[k++] = dir === 0 ? c : t;
            }
        }
    return out;
}

interface Warp {
    lines: Float64Array[];
    points: Float64Array[];
    boundary: Float64Array[];
    final: [number, number, number];
    loss: number;
    acc: number;
    steps: number;
    region: HTMLCanvasElement;
}

/** Trains a population of narrow networks and keeps the most legible one. */
class WarpSim {
    readonly key: string;
    cands: Candidate[];
    frameNo = 0;
    done = false;

    constructor(key: string, seed: number) {
        this.key = key;
        this.cands = Array.from({length: POPULATION}, (_, i) => ({
            net: new Mlp(new Uint32Array(WARP_SIZES), 0, seed + i * 7919),
            loss: 1,
            acc: 0,
            flatness: 0,
            finished: false,
        }));
    }

    /** Advance training one frame. Returns the candidate to draw, or null if nothing changed. */
    step(data: Float64Array, probe: Float64Array): Candidate | null {
        if (this.done) return null;
        this.frameNo++;
        for (const c of this.cands) {
            if (c.finished) continue;
            c.loss = c.net.train(data, 12, 0.03, 0, 1, 0);
            if (this.frameNo % 5 === 0) {
                c.acc = c.net.accuracy(data);
                if (c.acc >= 0.995 || c.net.steps >= 2400) {
                    c.finished = true;
                    c.flatness = Math.min(...Array.from({length: HIDDEN}, (_, l) => aspectRatio(c.net.layer_outputs(probe, l + 1))));
                }
            }
        }
        const score = (c: Candidate) => (c.finished && c.acc >= 0.97 ? 10 + c.flatness : c.acc - c.loss);
        let best = this.cands[0];
        for (const c of this.cands) if (score(c) > score(best)) best = c;
        if (this.cands.every((c) => c.finished)) {
            this.cands.forEach((c) => c !== best && c.net.free());
            this.cands = [best];
            this.done = true;
        }
        return this.frameNo % 3 === 0 || this.done ? best : null;
    }

    free() {
        this.cands.forEach((c) => c.net.free());
        this.cands = [];
    }
}

function WarpFigure(props: {shape?: "moons" | "xor" | "circles"}) {
    const [shape, setShape] = useState<"moons" | "xor" | "circles">(props.shape ?? "moons");
    const [seed, setSeed] = useState(4);
    const [warp, setWarp] = useState<Warp | null>(null);
    const [ref, visible] = useInView<HTMLDivElement>();
    const data = useMemo(() => make_shape(shape, 200, 0.08, 11), [shape]);
    const coords = useMemo(() => {
        const out = new Float64Array((data.length / 3) * 2);
        for (let i = 0; i < data.length / 3; i++) {
            out[2 * i] = data[3 * i];
            out[2 * i + 1] = data[3 * i + 1];
        }
        return out;
    }, [data]);
    const lines = useMemo(() => gridLines(), []);
    const key = `${shape}|${seed}`;
    // Narrow networks are sensitive to their initialization, and many of the
    // solutions they find squash the plane flat onto a line, which classifies
    // fine but shows nothing. So we train a small population side by side,
    // stop each one as soon as it classifies the data, and keep the one whose
    // layers stay the most two-dimensional.
    const sim = useRef<WarpSim | null>(null);
    const tl = useTimeline(HIDDEN, 0.7);
    const probe = useMemo(() => {
        const out = new Float64Array(21 * 21 * 2);
        for (let i = 0, k = 0; i < 21; i++)
            for (let j = 0; j < 21; j++) {
                out[k++] = -1.2 + 0.12 * i;
                out[k++] = -1.2 + 0.12 * j;
            }
        return out;
    }, []);

    useEffect(
        () => () => {
            sim.current?.free();
            sim.current = null;
        },
        []
    );

    useAnimationFrame(() => {
        let w = sim.current;
        if (!w || w.key !== key) {
            w?.free();
            w = new WarpSim(key, seed);
            sim.current = w;
        }
        const best = w.step(data, probe);
        if (!best) return;
        const net = best.net;
        const loss = best.loss;
        const G = 80;
        const grid = net.predict_grid(WARP_VIEW.xmin, WARP_VIEW.xmax, WARP_VIEW.ymin, WARP_VIEW.ymax, G, G);
        const segs = contour_segments(grid, G, G, WARP_VIEW.xmin, WARP_VIEW.xmax, WARP_VIEW.ymin, WARP_VIEW.ymax, new Float64Array([0.5]));
        const bpts = new Float64Array((segs.length / 5) * 4);
        for (let i = 0, k = 0; i < segs.length; i += 5) {
            bpts[k++] = segs[i];
            bpts[k++] = segs[i + 1];
            bpts[k++] = segs[i + 2];
            bpts[k++] = segs[i + 3];
        }
        const reps = (pts: Float64Array) => Array.from({length: HIDDEN + 1}, (_, l) => (l === 0 ? pts : net.layer_outputs(pts, l)));
        const params = net.params();
        const off = params.length - 3;
        setWarp({
            lines: reps(lines),
            points: reps(coords),
            boundary: reps(bpts),
            final: [params[off], params[off + 1], params[off + 2]],
            loss,
            acc: net.accuracy(data),
            steps: net.steps,
            region: heatImage(grid, G, G, (p) => probabilityColor(p, 0.3)),
        });
    }, visible);

    const t = tl.t;
    const l0 = Math.min(HIDDEN - 1, Math.floor(t));
    const s = ease(t - l0);
    const at = (reps: Float64Array[], i: number) => [lerp(reps[l0][i], reps[l0 + 1][i], s), lerp(reps[l0][i + 1], reps[l0 + 1][i + 1], s)];
    const layerName = t < 0.02 ? "input space" : t > HIDDEN - 0.02 ? `after hidden layer ${HIDDEN}: a straight line now separates the classes` : `between layer ${l0} and ${l0 + 1}`;

    return (
        <Instrument hint="Play to morph the plane through each layer">
            <div ref={ref}>
                <FigureLayout
                    main={
                        <>
                            <Plot
                                label="Data and grid lines transformed by each hidden layer"
                                view={WARP_VIEW}
                                maxHeight={560}
                                draw={(p) => {
                                    p.clear();
                                    if (!warp) return;
                                    if (t < 1) p.image(warp.region, WARP_VIEW, true, 1 - ease(t));
                                    const fin = ease((t - (HIDDEN - 0.6)) / 0.6);
                                    if (fin > 0) {
                                        const [a, b, c] = warp.final;
                                        p.halfPlane(a, b, c, C.blue, 0.13 * fin);
                                        p.halfPlane(-a, -b, -c, C.orange, 0.1 * fin);
                                    }
                                    // Warped grid.
                                    for (let line = 0; line < LINES * 2; line++) {
                                        const pts: number[] = [];
                                        for (let k = 0; k < SAMPLES; k++) {
                                            const [x, y] = at(warp.lines, (line * SAMPLES + k) * 2);
                                            pts.push(x, y);
                                        }
                                        const mid = line % LINES === (LINES - 1) / 2;
                                        p.polyline(pts, {color: mid ? rgba(C.white, 0.45) : rgba(C.purple, 0.32), width: mid ? 1.4 : 1});
                                    }
                                    // Decision boundary, carried along.
                                    const bd = warp.boundary;
                                    for (let i = 0; i < bd[0].length; i += 4) {
                                        const [x1, y1] = at(bd, i);
                                        const [x2, y2] = at(bd, i + 2);
                                        p.line(x1, y1, x2, y2, {color: C.white, width: 2.5});
                                    }
                                    const pts = warp.points;
                                    for (let i = 0; i < data.length / 3; i++) {
                                        const [x, y] = at(pts, 2 * i);
                                        p.point(x, y, data[3 * i + 2] > 0.5 ? C.blue : C.orange, {r: 3.8});
                                    }
                                    p.textPx(layerName, 12, 22, {color: C.text2, size: 12.5});
                                }}
                            />
                            <Player timeline={tl} />
                        </>
                    }
                    side={
                        <div className="controls">
                            <Segmented
                                value={shape}
                                onChange={(v) => (setShape(v), tl.reset())}
                                options={[
                                    {value: "moons", label: "Moons"},
                                    {value: "xor", label: "XOR"},
                                    {value: "circles", label: "Circles"},
                                ]}
                            />
                            <p className="note">
                                Network: <span className="mono">2 → 2 → 2 → 2 → 1</span>, tanh. Every hidden layer has just two neurons, so each intermediate
                                representation is itself a plane we can draw.
                            </p>
                            {warp && (
                                <Stats>
                                    <Stat label="training step" value={warp.steps} />
                                    <Stat label="accuracy" value={`${Math.round(warp.acc * 100)}%`} color={warp.acc > 0.97 ? C.green : C.yellow} />
                                </Stats>
                            )}
                            <div className="control-row">
                                <Button onClick={() => (setSeed(randomSeed()), tl.reset())}>Retrain from new weights</Button>
                            </div>
                            <p className="note note-muted">
                                Narrow networks often get stuck in a poor solution, so twelve are trained side by side and the clearest one is kept. On{" "}
                                <em>circles</em> even the best will struggle, for the reason explained below.
                            </p>
                        </div>
                    }
                />
            </div>
        </Instrument>
    );
}

// ===========================================================================
// The output as a 3D surface, folding while it trains
// ===========================================================================

const NS = 36;
const SPAN = 1.25;

/** Blue for p → 1, orange for p → 0. */
function probRgb(p: number): [number, number, number] {
    const a = CLASS_RGB[0];
    const b = CLASS_RGB[1];
    const t = Math.max(0, Math.min(1, p));
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function Surface3D(props: {shape?: Shape; layers?: number[]}) {
    const shape = props.shape ?? "circles";
    const layers = useMemo(() => props.layers ?? [8, 8], [props.layers]);
    const data = useMemo(() => make_shape(shape, 160, 0.08, 4), [shape]);
    const [seed, setSeed] = useState(1);
    const [running, setRunning] = useState(true);
    const [frame, setFrame] = useState<{heights: Float64Array; loss: number; steps: number} | null>(null);
    const [ref, visible] = useInView<HTMLDivElement>();
    const sim = useRef<{net: Mlp; key: string} | null>(null);
    const acc = useRef(0);
    const pending = useRef(0);
    const key = `${shape}|${layers.join("-")}|${seed}`;

    useEffect(
        () => () => {
            sim.current?.net.free();
            sim.current = null;
        },
        []
    );

    useAnimationFrame(() => {
        let s = sim.current;
        if (!s || s.key !== key) {
            s?.net.free();
            s = {net: new Mlp(new Uint32Array([2, ...layers, 1]), 0, seed), key};
            sim.current = s;
        }
        let n = running && s.net.steps < 4000 ? takeSteps(acc, getSpeed(), 40) : 0;
        n += pending.current;
        pending.current = 0;
        if (n === 0 && frame) return;
        const loss = n > 0 ? s.net.train(data, n, 0.015, 0, 1, 0) : s.net.loss(data);
        const h = (2 * SPAN) / (NS - 1);
        const p = s.net.predict_grid(-SPAN - h / 2, SPAN + h / 2, -SPAN - h / 2, SPAN + h / 2, NS, NS);
        setFrame({heights: p, loss, steps: s.net.steps});
    }, visible);

    return (
        <Instrument hint="Height = the network's probability of blue · blue points sit at height 1, orange at 0 · drag to orbit">
            <div ref={ref}>
                <FigureLayout
                    main={
                        <Plot3D
                            label="Neural network output as a 3D surface"
                            initial={{pitch: 0.45, yaw: 0.6}}
                            draw={(sc) => {
                                if (!frame) return;
                                const heights = frame.heights.map((v) => v * 0.9 - 0.45);
                                drawSurface(sc, heights, NS, (z) => probRgb((z + 0.45) / 0.9), {alpha: 0.88, wire: 5});
                                for (let i = 0; i < data.length; i += 3) {
                                    const c = data[i + 2];
                                    sc.point([data[i] / SPAN, data[i + 1] / SPAN, c > 0.5 ? 0.47 : -0.47], classColor(c), 3.6, {bias: 0.01});
                                }
                            }}
                        />
                    }
                    side={
                        <div className="controls">
                            <LiveControls
                                running={running}
                                onToggle={() => setRunning(!running)}
                                onStep={() => (pending.current += 1)}
                                stepTitle="One training step"
                                onReset={() => setSeed(randomSeed())}
                                resetTitle="Start again from new random weights"
                            />
                            <Stats>
                                <Stat label="training step" value={frame?.steps ?? 0} />
                                <Stat label="loss" value={frame ? frame.loss.toFixed(3) : "—"} color={C.red} />
                            </Stats>
                            <p className="note">
                                The network's output is a single sheet over the plane. Training pushes it up under blue points and down under orange
                                ones, and with enough neurons it can fold into whatever shape that takes. Where the sheet crosses half height is the decision
                                boundary.
                            </p>
                            <p className="note note-muted">Architecture 2 → {layers.join(" → ")} → 1, tanh, trained with Adam.</p>
                        </div>
                    }
                />
            </div>
        </Instrument>
    );
}

// ===========================================================================
// Playground
// ===========================================================================

const SCENARIOS: Scenario[] = [
    {
        id: "spiral",
        title: "Untangle a spiral",
        prompt: <>Each little square is one neuron's response over the whole plane. Watch the simple early-layer edges combine into the spiral.</>,
        render: () => <PlaygroundFigure />,
    },
    {
        id: "three-neurons",
        title: "Fence in a circle",
        prompt: <>One hidden layer with just 3 neurons, so 3 lines. That is exactly enough to fence in the center. Try 2 neurons and watch it fail.</>,
        render: () => <PlaygroundFigure shape="circles" layers={[3]} />,
    },
    {
        id: "linear",
        title: "No nonlinearity",
        prompt: <>Linear activations: however many layers you stack, the boundary stays straight, so XOR is impossible. Switch the activation to tanh or ReLU and train again.</>,
        render: () => <PlaygroundFigure shape="xor" act={3} layers={[8, 8]} />,
    },
    {
        id: "relu",
        title: "ReLU folds",
        prompt: <>ReLU networks build boundaries from straight folds, so you get polygonal edges instead of smooth curves.</>,
        render: () => <PlaygroundFigure shape="moons" act={1} layers={[6, 6]} noise={0.15} />,
    },
    {
        id: "surface",
        title: "Output surface in 3D",
        prompt: <>The network's output as a 3D sheet. Blue points float at the top and orange ones rest on the bottom. Watch training fold the sheet into a dome that holds up the inner ring.</>,
        render: () => <Surface3D />,
    },
    {
        id: "surface-spiral",
        title: "Folding a spiral (3D)",
        prompt: <>A spiral needs a much more contorted sheet: ridges that wind around the center. Watch the network sculpt them over training.</>,
        render: () => <Surface3D shape="spiral" layers={[10, 10, 10]} />,
    },
    {
        id: "bend",
        title: "Layers bend space",
        prompt: <>A 2-2-2-2-1 network, so every layer is a plane you can draw. Press play to watch the data move through the layers until a straight line separates it.</>,
        render: () => <WarpFigure />,
    },
    {
        id: "rings",
        title: "Rings can't be untangled",
        prompt: <>With only two neurons per layer, the network can stretch and squeeze the plane but not tear it. The inner disc can't escape the ring around it. Compare XOR.</>,
        render: () => <WarpFigure shape="circles" />,
    },
    {
        id: "neuron",
        title: "One neuron",
        prompt: <>A single neuron: a weighted sum (a tilted ramp across the plane), then an activation that bends it into a soft step or a fold. Hover to trace a point.</>,
        render: () => <NeuronFigure />,
    },
];

export default function NeuralNetworks() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>A neuron computes a weighted sum and applies a nonlinearity. A layer is many neurons reading the same input:</li>
                    </ul>
                    <MathBlock>{"\\mathbf{h}_1 = f(W_1\\vx + \\mathbf{b}_1), \\quad \\mathbf{h}_2 = f(W_2\\mathbf{h}_1 + \\mathbf{b}_2), \\quad \\hat p = \\sigma(\\mathbf{w}_3\\cdot\\mathbf{h}_2 + b_3)"}</MathBlock>
                    <ul>
                        <li>
                            Training minimizes cross-entropy by gradient descent (Adam here). <strong>Backpropagation</strong> computes every gradient with
                            the chain rule, from the output layer back to the input.
                        </li>
                        <li>
                            Without a nonlinear <M>{"f"}</M>, the whole stack collapses into one linear map. The nonlinearity is what makes depth useful.
                        </li>
                        <li>
                            Geometrically, hidden layers rotate, stretch and squash the plane until the final layer, which is just logistic regression, can
                            separate the classes with a straight line.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
