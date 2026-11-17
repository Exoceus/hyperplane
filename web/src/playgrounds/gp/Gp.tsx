import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {FigureLayout} from "../../components/Figure";
import {M, MathBlock} from "../../components/Math";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {Plot, type PlotPointer} from "../../components/Plot";
import {fmt, useTimeline} from "../../lib/hooks";
import type {Painter, View} from "../../lib/painter";
import {C, rgba} from "../../lib/theme";
import {gp_fit_hyper, gp_log_marginal, gp_posterior, gp_samples, make_clusters_nd, make_linear, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.2, xmax: 1.2, ymin: -2.2, ymax: 2.2};
const M_GRID = 160;
const GRID = new Float64Array(Array.from({length: M_GRID}, (_, i) => VIEW.xmin + ((VIEW.xmax - VIEW.xmin) * i) / (M_GRID - 1)));
const SAMPLE_COLORS = [C.blue, C.pink, C.green, C.purple, C.orange];

const KERNELS = [
    {value: 0, label: "RBF"},
    {value: 1, label: "Matérn"},
    {value: 2, label: "Periodic"},
    {value: 3, label: "Linear"},
];

interface Obs {
    x: number;
    y: number;
}

function split(obs: Obs[]) {
    return {xs: new Float64Array(obs.map((o) => o.x)), ys: new Float64Array(obs.map((o) => o.y))};
}

/** Mean line, ±2σ band and optional sample paths. */
function drawPosterior(p: Painter, post: Float64Array, samples: Float64Array | null, nSamples: number) {
    const mean = post.subarray(0, M_GRID);
    const std = post.subarray(M_GRID);
    const band: number[] = [];
    for (let i = 0; i < M_GRID; i++) band.push(GRID[i], mean[i] + 2 * std[i]);
    for (let i = M_GRID - 1; i >= 0; i--) band.push(GRID[i], mean[i] - 2 * std[i]);
    p.polyline(band, {close: true, fill: C.blue, fillAlpha: 0.13, color: "none"});
    const inner: number[] = [];
    for (let i = 0; i < M_GRID; i++) inner.push(GRID[i], mean[i] + std[i]);
    for (let i = M_GRID - 1; i >= 0; i--) inner.push(GRID[i], mean[i] - std[i]);
    p.polyline(inner, {close: true, fill: C.blue, fillAlpha: 0.12, color: "none"});
    if (samples)
        for (let s = 0; s < nSamples; s++) {
            const pts: number[] = [];
            for (let i = 0; i < M_GRID; i++) pts.push(GRID[i], samples[s * M_GRID + i]);
            p.polyline(pts, {color: rgba(SAMPLE_COLORS[s % SAMPLE_COLORS.length], 0.7), width: 1.3});
        }
    const m: number[] = [];
    for (let i = 0; i < M_GRID; i++) m.push(GRID[i], mean[i]);
    p.polyline(m, {color: C.yellow, width: 2.6});
}

// ===========================================================================
// GP regression instrument
// ===========================================================================

const DEFAULT_OBS: Obs[] = [
    {x: -0.8, y: 0.4},
    {x: -0.35, y: -0.6},
    {x: 0.1, y: 0.3},
    {x: 0.55, y: 1.0},
];

function GpInstrument(props: {obs?: Obs[]; kernel?: number; length?: number; variance?: number; noise?: number; samples?: number}) {
    const [obs, setObs] = useState<Obs[]>(props.obs ?? DEFAULT_OBS);
    const [kind, setKind] = useState(props.kernel ?? 0);
    const [length, setLength] = useState(props.length ?? 0.3);
    const [variance, setVariance] = useState(props.variance ?? 1);
    const [noise, setNoise] = useState(props.noise ?? 0.001);
    const [nSamples, setNSamples] = useState(props.samples ?? 3);
    const [seed, setSeed] = useState(4);
    const [drag, setDrag] = useState<number | null>(null);

    const {xs, ys} = useMemo(() => split(obs), [obs]);
    const post = useMemo(() => gp_posterior(xs, ys, kind, length, variance, noise, GRID), [xs, ys, kind, length, variance, noise]);
    const samples = useMemo(() => (nSamples > 0 ? gp_samples(xs, ys, kind, length, variance, noise, GRID, nSamples, seed) : null), [xs, ys, kind, length, variance, noise, nSamples, seed]);
    const evidence = useMemo(() => gp_log_marginal(xs, ys, kind, length, variance, noise), [xs, ys, kind, length, variance, noise]);

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") {
            const hit = obs.findIndex((o) => Math.hypot((o.x - e.x) * e.kx, (o.y - e.y) * e.ky) < 11);
            if (hit >= 0 && (e.shift || e.button === 2)) setObs(obs.filter((_, i) => i !== hit));
            else if (hit >= 0) setDrag(hit);
            else setObs([...obs, {x: e.x, y: e.y}]);
        }
        if (e.type === "move" && drag !== null) setObs(obs.map((o, i) => (i === drag ? {x: Math.max(VIEW.xmin, Math.min(VIEW.xmax, e.x)), y: Math.max(VIEW.ymin, Math.min(VIEW.ymax, e.y))} : o)));
        if (e.type === "up") setDrag(null);
    };

    const fitHyper = () => {
        const [l, v] = gp_fit_hyper(xs, ys, kind, noise);
        setLength(l);
        setVariance(v);
    };

    return (
        <Instrument hint="Click to add an observation · drag to move it · shift-click to delete">
            <FigureLayout
                main={
                    <Plot
                        label="Gaussian process posterior mean, uncertainty band and samples"
                        view={VIEW}
                        aspect={0.62}
                        cursor={drag !== null ? "grabbing" : "crosshair"}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            p.grid(0.25);
                            drawPosterior(p, post, samples, nSamples);
                            for (const o of obs) p.point(o.x, o.y, C.white, {r: 5.5});
                        }}
                    />
                }
                side={
                    <div className="controls">
                        <Segmented label="kernel" value={kind} onChange={setKind} options={KERNELS} />
                        {kind !== 3 && (
                            <Slider label={<>length scale <M>{"\\ell"}</M></>} value={length} min={0.03} max={2} log onChange={setLength} color={C.yellow} format={(v) => v.toFixed(3)} />
                        )}
                        <Slider label={<>signal variance <M>{"\\sigma_f^2"}</M></>} value={variance} min={0.05} max={4} log onChange={setVariance} color={C.blue} />
                        <Slider label={<>noise <M>{"\\sigma_n^2"}</M></>} value={noise} min={1e-4} max={0.5} log onChange={setNoise} color={C.red} format={(v) => v.toExponential(1)} />
                        <Slider label="sample functions" value={nSamples} min={0} max={5} step={1} onChange={setNSamples} format={(v) => v.toFixed(0)} />
                        <Stats>
                            <Stat label="observations" value={obs.length} />
                            <Stat label="log evidence" value={fmt(evidence, 2)} color={C.green} />
                        </Stats>
                        <div className="control-row">
                            <Button variant="primary" onClick={fitHyper} disabled={obs.length < 2}>
                                Fit ℓ and σ² to data
                            </Button>
                            <Button onClick={() => setSeed(randomSeed())}>Resample</Button>
                            <Button onClick={() => setObs([])}>Clear</Button>
                        </div>
                        <p className="note note-muted">
                            Yellow is the posterior mean, the bands are ±1σ and ±2σ, and thin lines are plausible functions drawn from the posterior.
                        </p>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Bayesian optimization instrument
// ===========================================================================

const objective = (x: number) => 0.55 * Math.sin(3.2 * x + 0.4) + 0.4 * Math.sin(9 * x - 1.1) * Math.exp(-((x - 0.45) ** 2) * 3) + 0.15 * x;

const BO_STEPS = 14;
const BO_START: Obs[] = [
    {x: -0.7, y: objective(-0.7)},
    {x: 0.2, y: objective(0.2)},
];

/** UCB acquisition over the grid for a set of observations. */
function acquisition(obs: Obs[], kappa: number) {
    const {xs, ys} = split(obs);
    const post = gp_posterior(xs, ys, 0, 0.25, 0.6, 1e-4, GRID);
    const ucb = new Float64Array(M_GRID);
    let best = -1;
    for (let i = 0; i < M_GRID; i++) {
        ucb[i] = post[i] + kappa * post[M_GRID + i];
        if (Math.abs(GRID[i]) <= 1.0 && (best < 0 || ucb[i] > ucb[best])) best = i;
    }
    return {post, ucb, next: GRID[best]};
}

/** The whole Bayesian-optimization run: keep sampling wherever UCB is highest. */
function bayesOptRun(kappa: number): Obs[] {
    const obs = [...BO_START];
    for (let s = 0; s < BO_STEPS; s++) {
        const {next} = acquisition(obs, kappa);
        obs.push({x: next, y: objective(next)});
    }
    return obs;
}

function BayesOpt() {
    const [kappa, setKappa] = useState(2);
    const [reveal, setReveal] = useState(false);
    const run = useMemo(() => bayesOptRun(kappa), [kappa]);
    const tl = useTimeline(BO_STEPS, 1, {autoplay: true, runKey: run, duration: 11, delay: 0.6});
    const count = BO_START.length + Math.min(BO_STEPS, Math.floor(tl.t));
    const obs = useMemo(() => run.slice(0, count), [run, count]);
    const {post, ucb, next} = useMemo(() => acquisition(obs, kappa), [obs, kappa]);
    const truth = useMemo(() => {
        let best = {x: 0, y: -Infinity};
        for (let i = 0; i <= 2000; i++) {
            const x = -1 + i / 1000;
            if (objective(x) > best.y) best = {x, y: objective(x)};
        }
        return best;
    }, []);
    const found = obs.reduce((b, o) => (o.y > b.y ? o : b), {x: 0, y: -Infinity});
    const gap = truth.y - found.y;

    return (
        <Instrument hint="A hidden function. Each sample is expensive, so where should you look next?">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Bayesian optimization: GP surrogate of a hidden function"
                            view={{xmin: -1.05, xmax: 1.05, ymin: -1.6, ymax: 1.6}}
                            aspect={0.5}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25);
                                drawPosterior(p, post, null, 0);
                                if (reveal) {
                                    const t: number[] = [];
                                    for (let i = 0; i <= 300; i++) {
                                        const x = -1 + (2 * i) / 300;
                                        t.push(x, objective(x));
                                    }
                                    p.polyline(t, {color: C.green, width: 2, dash: [6, 5]});
                                    p.text("★", truth.x, truth.y, {color: C.green, size: 18, align: "center", baseline: "middle"});
                                }
                                p.line(next, -1.6, next, 1.6, {color: rgba(C.pink, 0.8), width: 1.5, dash: [4, 4]});
                                obs.forEach((o, i) => p.point(o.x, o.y, i === obs.length - 1 && obs.length > 2 ? C.pink : C.white, {r: 5.5}));
                                p.circle(found.x, found.y, 10, {stroke: C.yellow, width: 2});
                            }}
                        />
                        <div className="panel-label">
                            <span>Acquisition (upper confidence bound)</span>
                        </div>
                        <Plot
                            label="Upper confidence bound acquisition function"
                            view={{xmin: -1.05, xmax: 1.05, ymin: Math.min(...ucb) - 0.2, ymax: Math.max(...ucb) + 0.2}}
                            height={90}
                            draw={(p) => {
                                p.clear();
                                const pts: number[] = [];
                                for (let i = 0; i < M_GRID; i++) pts.push(GRID[i], ucb[i]);
                                p.polyline(pts, {color: C.pink, width: 2});
                                p.line(next, p.view.ymin, next, p.view.ymax, {color: rgba(C.pink, 0.8), width: 1.5, dash: [4, 4]});
                            }}
                        />
                        <Player
                            timeline={tl}
                            speeds={[0.5, 1, 3]}
                            status={
                                count >= run.length
                                    ? `Done: ${run.length} samples. Best found ${obs.reduce((b, o) => Math.max(b, o.y), -Infinity).toFixed(3)}.`
                                    : `Sample ${count + 1}: refit the GP, then evaluate the hidden function where μ + κσ is highest (pink line).`
                            }
                        />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label={<>exploration <M>{"\\kappa"}</M></>} value={kappa} min={0} max={5} onChange={setKappa} color={C.pink} />
                        <Stats>
                            <Stat label="samples used" value={obs.length} />
                            <Stat label="best found" value={found.y.toFixed(3)} color={C.yellow} />
                            <Stat label="gap to true max" value={reveal || gap < 1e-3 ? gap.toFixed(3) : "?"} color={gap < 0.01 ? C.green : undefined} />
                        </Stats>
                        <Toggle label="Reveal the hidden function" checked={reveal} onChange={setReveal} />
                        <p className="note">
                            The GP is a cheap stand-in for the expensive function. Its uncertainty says where you might be wrong. UCB samples where the
                            optimistic guess is highest: high mean (exploit) or high uncertainty (explore). Set κ = 0 and it gets greedy.
                        </p>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Learning one observation at a time
// ===========================================================================

const STREAM = 14;
const hidden = (x: number) => 0.75 * Math.sin(3 * x) + 0.35 * Math.cos(7 * x + 0.5);

function GpStream() {
    const [seed, setSeed] = useState(2);
    const [length, setLength] = useState(0.3);
    const [noise, setNoise] = useState(0.005);
    const [showTruth, setShowTruth] = useState(true);
    const sequence = useMemo(() => {
        const xs = make_linear(STREAM, 1, 0, 0, 0, 0, seed);
        const eps = make_clusters_nd(STREAM, new Float64Array([0]), new Float64Array([Math.sqrt(noise)]), 1, seed + 1);
        return Array.from({length: STREAM}, (_, i) => {
            const x = xs[3 * i] * 1.1;
            return {x, y: hidden(x) + eps[2 * i]};
        });
    }, [seed, noise]);
    const tl = useTimeline(STREAM, 1, {autoplay: true, runKey: sequence, duration: 10, delay: 0.6});
    const count = Math.min(STREAM, Math.floor(tl.t));
    const obs = useMemo(() => sequence.slice(0, count), [sequence, count]);
    const {xs, ys} = useMemo(() => split(obs), [obs]);
    const post = useMemo(() => gp_posterior(xs, ys, 0, length, 1, noise, GRID), [xs, ys, length, noise]);
    const samples = useMemo(() => gp_samples(xs, ys, 0, length, 1, noise, GRID, 3, seed + count), [xs, ys, length, noise, seed, count]);
    const meanStd = useMemo(() => {
        let s = 0;
        for (let i = 0; i < M_GRID; i++) s += post[M_GRID + i];
        return s / M_GRID;
    }, [post]);

    return (
        <Instrument hint="Observations of a hidden function arrive one at a time; the posterior updates after each">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Gaussian process posterior updating as data arrives"
                            view={VIEW}
                            aspect={0.62}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25);
                                if (showTruth) {
                                    const t: number[] = [];
                                    for (let i = 0; i <= 300; i++) {
                                        const x = VIEW.xmin + ((VIEW.xmax - VIEW.xmin) * i) / 300;
                                        t.push(x, hidden(x));
                                    }
                                    p.polyline(t, {color: rgba(C.green, 0.6), width: 1.5, dash: [6, 5]});
                                }
                                drawPosterior(p, post, samples, 3);
                                obs.forEach((o, i) => p.point(o.x, o.y, C.white, {r: 5.5, ring: i === obs.length - 1 && tl.t < STREAM ? C.yellow : undefined}));
                            }}
                        />
                        <Player
                            timeline={tl}
                            speeds={[0.5, 1, 3]}
                            status={count === 0 ? "No data yet: the prior says anything between −2 and 2 is plausible everywhere." : `${count} observation${count === 1 ? "" : "s"}: average uncertainty σ = ${meanStd.toFixed(3)}`}
                        />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label={<>length scale <M>{"\\ell"}</M></>} value={length} min={0.05} max={1.5} log onChange={setLength} color={C.yellow} format={(v) => v.toFixed(3)} />
                        <Slider label={<>noise <M>{"\\sigma_n^2"}</M></>} value={noise} min={1e-4} max={0.3} log onChange={setNoise} color={C.red} format={(v) => v.toExponential(1)} />
                        <Toggle label="Show the hidden function" checked={showTruth} onChange={setShowTruth} />
                        <p className="note">
                            Each new point is a constraint: the samples (thin lines) must pass near it, and the band pinches there. Between points the
                            kernel decides how far that information spreads.
                        </p>
                        <div className="control-row">
                            <Button onClick={() => setSeed(randomSeed())}>New data</Button>
                        </div>
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
        id: "stream",
        title: "One point at a time",
        prompt: <>Observations of a hidden function arrive one by one. After each, the posterior updates: the band pinches at the new point and the sampled functions snap into agreement there.</>,
        render: () => <GpStream />,
    },
    {
        id: "observe",
        title: "Add observations",
        prompt: <>Click to add data points. The band pinches to nothing at each observation and widens back to the prior far away. The model knows where it doesn't know.</>,
        render: () => <GpInstrument />,
    },
    {
        id: "length-scale",
        title: "Length scale",
        prompt: <>A tiny length scale says "nearby points barely relate", so the mean snaps back between observations and the samples go wild. Slide ℓ up for smoother beliefs.</>,
        render: () => <GpInstrument length={0.06} />,
    },
    {
        id: "noise",
        title: "Noisy measurements",
        prompt: <>With a large noise term the mean no longer has to pass through the points, and uncertainty never fully vanishes. Shrink the noise to force interpolation.</>,
        render: () => <GpInstrument noise={0.15} obs={[...DEFAULT_OBS, {x: 0.12, y: 0.75}, {x: -0.33, y: -0.2}, {x: 0.6, y: 0.6}]} />,
    },
    {
        id: "periodic",
        title: "Kernels encode beliefs",
        prompt: <>A periodic kernel assumes the pattern repeats, so a few points on the left let it extrapolate confidently to the right. Switch kernels to see different assumptions.</>,
        render: () => (
            <GpInstrument
                kernel={2}
                length={0.8}
                obs={[
                    {x: -1.0, y: 0.1},
                    {x: -0.8, y: 0.9},
                    {x: -0.55, y: 0.3},
                    {x: -0.3, y: -0.7},
                ]}
            />
        ),
    },
    {
        id: "fit",
        title: "Let the data choose",
        prompt: <>These hyperparameters are badly off. Press <b>Fit ℓ and σ² to data</b>: it maximizes the marginal likelihood, balancing data fit against model complexity.</>,
        render: () => (
            <GpInstrument
                length={1.5}
                variance={0.1}
                obs={Array.from({length: 10}, (_, i) => {
                    const x = -1 + (i + 0.3) / 5;
                    return {x, y: Math.sin(5 * x) * 0.9};
                })}
            />
        ),
    },
    {
        id: "bayes-opt",
        title: "Bayesian optimization",
        prompt: <>Find the maximum of a hidden function in as few samples as possible. Each round, the GP is refit and the next sample goes wherever μ + κσ peaks. Slide κ to trade exploring for exploiting.</>,
        render: () => <BayesOpt />,
    },
];

export default function Gp() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            A Gaussian process is a distribution over <em>functions</em>. Any finite set of values <M>{"f(x_1), \\ldots, f(x_n)"}</M> is
                            jointly Gaussian, with covariance given by a kernel <M>{"k(x, x')"}</M> that encodes how similar nearby points should be.
                        </li>
                        <li>Conditioning on observations gives a closed-form posterior at any new point:</li>
                    </ul>
                    <MathBlock>{"\\mu(x_*) = \\mathbf k_*^\\top (K + \\sigma_n^2 I)^{-1}\\mathbf y, \\qquad \\sigma^2(x_*) = k(x_*, x_*) - \\mathbf k_*^\\top (K + \\sigma_n^2 I)^{-1}\\mathbf k_*"}</MathBlock>
                    <ul>
                        <li>
                            RBF: <M>{"k = \\sigma_f^2 \\exp\\!\\big(-\\tfrac{(x-x')^2}{2\\ell^2}\\big)"}</M>. The length scale <M>{"\\ell"}</M> sets how far
                            information spreads. Hyperparameters can be fit by maximizing the marginal likelihood <M>{"p(\\mathbf y \\mid X, \\theta)"}</M>.
                        </li>
                        <li>
                            Calibrated uncertainty powers Bayesian optimization: fit a GP to a few expensive evaluations, then sample where an acquisition
                            function like UCB <M>{"\\mu + \\kappa\\sigma"}</M> is largest.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
