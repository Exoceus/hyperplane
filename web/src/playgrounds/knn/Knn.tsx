import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider} from "../../components/Controls";
import {FigureLayout} from "../../components/Figure";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {classRegionColor} from "../../lib/colors";
import {appendRow, drawLabelled, hitPoint, removeRow} from "../../lib/data";
import {useTimeline} from "../../lib/hooks";
import {heatImage, type Painter, type View} from "../../lib/painter";
import {C, classColor, rgba} from "../../lib/theme";
import {knn_accuracy_curve, knn_grid, knn_query, make_blobs, make_shape, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const G = 200;
const SCAN = 44;
const KMAX = 40;

type Kind = "blobs" | "moons" | "circles" | "xor" | "spiral";
const KINDS: {value: Kind; label: string}[] = [
    {value: "blobs", label: "3 blobs"},
    {value: "moons", label: "Moons"},
    {value: "circles", label: "Circles"},
    {value: "xor", label: "XOR"},
    {value: "spiral", label: "Spiral"},
];

function generate(kind: Kind, seed: number, noise: number): Float64Array {
    if (kind === "blobs") return make_blobs(150, 3, 0.16 + noise * 0.25, seed);
    return make_shape(kind, 160, noise, seed);
}

/** Outline of the metric's "ball" of radius r: circle, diamond or square. */
function drawBall(p: Painter, cx: number, cy: number, r: number, metric: number, color: string) {
    if (metric === 0) {
        p.circle(cx, cy, r * p.kx, {stroke: color, width: 1.5, dash: [5, 4]});
        return;
    }
    const pts = metric === 1 ? [cx + r, cy, cx, cy + r, cx - r, cy, cx, cy - r] : [cx + r, cy + r, cx - r, cy + r, cx - r, cy - r, cx + r, cy - r];
    p.polyline(pts, {close: true, color, width: 1.5, dash: [5, 4]});
}

function KnnFigure(props: {kind?: Kind; noise?: number; k?: number; metric?: number}) {
    const [kind, setKind] = useState<Kind>(props.kind ?? "moons");
    const [seed, setSeed] = useState(4);
    const [noise, setNoise] = useState(props.noise ?? 0.45);
    const [k, setK] = useState(props.k ?? 5);
    const [metric, setMetric] = useState(props.metric ?? 0);
    const [extra, setExtra] = useState<Float64Array>(new Float64Array());
    const [brush, setBrush] = useState(0);
    const [query, setQuery] = useState<{x: number; y: number} | null>({x: 0.1, y: 0.15});

    const base = useMemo(() => generate(kind, seed, noise), [kind, seed, noise]);
    const test = useMemo(() => generate(kind, seed + 7919, noise), [kind, seed, noise]);
    const data = useMemo(() => {
        const out = new Float64Array(base.length + extra.length);
        out.set(base);
        out.set(extra, base.length);
        return out;
    }, [base, extra]);
    const numClasses = kind === "blobs" ? 3 : 2;

    const regions = useMemo(() => {
        const g = knn_grid(data, k, metric, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G);
        return heatImage(g, G, G, (c, i) => classRegionColor(c, 0.3 + 0.7 * g[2 * i + 1], 0.3), 2);
    }, [data, k, metric]);

    // The scan: the same classifier evaluated cell by cell on a coarse grid,
    // replayed in reading order so you can watch each query get answered.
    const scan = useMemo(() => knn_grid(data, k, metric, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, SCAN, SCAN), [data, k, metric]);
    const tl = useTimeline(SCAN * SCAN, 1, {autoplay: true, runKey: scan, duration: 6});
    const scanning = !tl.atEnd;
    const cell = Math.min(SCAN * SCAN - 1, Math.floor(tl.t));
    const painted = useMemo(
        () => heatImage(scan, SCAN, SCAN, (c, i) => (i <= cell ? [...classRegionColor(c, 0.3 + 0.7 * scan[2 * i + 1], 0.34), 255] : [0, 0, 0, 0]) as [number, number, number, number], 2),
        [scan, cell]
    );
    const probe = useMemo(
        () =>
            scanning
                ? {x: VIEW.xmin + ((cell % SCAN) + 0.5) * ((VIEW.xmax - VIEW.xmin) / SCAN), y: VIEW.ymax - (Math.floor(cell / SCAN) + 0.5) * ((VIEW.ymax - VIEW.ymin) / SCAN)}
                : query,
        [scanning, cell, query]
    );

    const curve = useMemo(() => {
        const c = knn_accuracy_curve(data, test, KMAX, metric);
        const train = new Float64Array(KMAX);
        const held = new Float64Array(KMAX);
        for (let i = 0; i < KMAX; i++) {
            train[i] = c[2 * i];
            held[i] = c[2 * i + 1];
        }
        return {train, held};
    }, [data, test, metric]);

    const neighbours = useMemo(() => {
        if (!probe) return null;
        const r = knn_query(data, probe.x, probe.y, k, metric);
        const kk = r.length / 2;
        const idx = Array.from(r.subarray(0, kk));
        const votes = new Array(numClasses).fill(0);
        for (const i of idx) votes[data[3 * i + 2]]++;
        return {idx, radius: r[r.length - 1], votes};
    }, [data, probe, k, metric, numClasses]);

    const onPointer = (e: PlotPointer) => {
        if (e.type === "move" || e.type === "down") setQuery({x: e.x, y: e.y});
        if (e.type === "down") {
            const nBase = base.length / 3;
            const hit = hitPoint(data, 3, e.x, e.y, e.kx, e.ky, 8);
            if (hit >= nBase) setExtra(removeRow(extra, 3, hit - nBase));
            else if (hit < 0) setExtra(appendRow(extra, [e.x, e.y, brush]));
        }
    };

    const reset = () => setExtra(new Float64Array());

    return (
        <Instrument hint="Watch each cell get classified by its neighbors · then hover to probe · click to add a point">
            <FigureLayout
                sideWidth={300}
                main={
                    <>
                    <Plot
                        label="k-nearest-neighbour decision regions"
                        view={VIEW}
                        maxHeight={620}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            if (scanning) p.image(painted, VIEW, false);
                            else p.image(regions, VIEW);
                            p.grid(0.25, {axes: false});
                            if (neighbours && probe) {
                                for (const i of neighbours.idx) p.line(probe.x, probe.y, data[3 * i], data[3 * i + 1], {color: rgba(C.white, 0.55), width: 1.2});
                                drawBall(p, probe.x, probe.y, neighbours.radius, metric, rgba(C.white, 0.75));
                            }
                            drawLabelled(p, data, {r: 4});
                            if (neighbours) for (const i of neighbours.idx) p.circle(data[3 * i], data[3 * i + 1], 7.5, {stroke: C.white, width: 1.5});
                            if (probe) p.circle(probe.x, probe.y, 5, {fill: scanning ? C.yellow : C.white, stroke: C.canvas, width: 2});
                        }}
                    />
                    <Player
                        timeline={tl}
                        status={scanning ? `Classifying cell ${cell + 1} of ${SCAN * SCAN}: find its ${k} nearest points, take a majority vote, paint the cell.` : "Every cell classified. Hover anywhere to probe a point yourself."}
                    />
                    </>
                }
                side={
                    <div className="controls">
                        <Segmented value={kind} onChange={(v) => (setKind(v), reset(), setBrush(0))} options={KINDS} />
                        <Slider label={<>neighbors <M>{"k"}</M></>} value={k} min={1} max={KMAX} step={1} onChange={setK} format={(v) => v.toFixed(0)} color={C.yellow} />
                        <Slider label="noise" value={noise} min={0} max={0.6} onChange={(v) => (setNoise(v), reset())} />
                        <Segmented
                            label="distance"
                            value={metric}
                            onChange={setMetric}
                            options={[
                                {value: 0, label: "Euclidean", title: "straight-line distance (L2)"},
                                {value: 1, label: "Manhattan", title: "sum of |Δx| + |Δy| (L1)"},
                                {value: 2, label: "Chebyshev", title: "max of |Δx|, |Δy| (L∞)"},
                            ]}
                        />
                        {neighbours && (
                            <div className="side-block">
                                <span className="control-label">Votes at the probe</span>
                                <div style={{display: "flex", gap: 3, height: 14, borderRadius: 4, overflow: "hidden"}}>
                                    {neighbours.votes.map((v, c) =>
                                        v > 0 ? <div key={c} style={{flex: v, background: classColor(c)}} title={`${v} votes`} /> : null
                                    )}
                                </div>
                                <span className="note">
                                    {neighbours.votes.map((v, c) => (
                                        <span key={c} style={{color: classColor(c), marginRight: 10}}>
                                            {v} vote{v === 1 ? "" : "s"}
                                        </span>
                                    ))}
                                </span>
                            </div>
                        )}
                        <div className="panel-label">
                            <span>Accuracy vs. k</span>
                        </div>
                        <LineChart
                            label="Training and held-out accuracy for each k"
                            series={[
                                {values: curve.train, color: C.muted, x0: 1, dash: [4, 3]},
                                {values: curve.held, color: C.green, x0: 1},
                            ]}
                            height={110}
                            xMin={1}
                            yMax={1}
                            marker={k}
                            yFormat={(v) => `${Math.round(v * 100)}%`}
                            onPick={(x) => setK(Math.max(1, Math.min(KMAX, Math.round(x))))}
                        />
                        <p className="note note-muted">
                            <span style={{color: "var(--muted)"}}>- - training points</span> ·{" "}
                            <span className="c-green">— fresh test points</span>. Held-out accuracy at k = {k}: {Math.round(curve.held[k - 1] * 100)}%.
                        </p>
                        <Segmented
                            label="Clicking adds"
                            value={brush}
                            onChange={setBrush}
                            options={Array.from({length: numClasses}, (_, c) => ({value: c, label: <span style={{color: classColor(c)}}>● class {c}</span>}))}
                        />
                        <div className="control-row">
                            <Button onClick={() => (setSeed(randomSeed()), reset())}>New data</Button>
                            {extra.length > 0 && <Button onClick={reset}>Clear added</Button>}
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

const SCENARIOS: Scenario[] = [
    {
        id: "k1",
        title: "k = 1 memorizes",
        prompt: <>Every point owns its own little territory, including the noisy ones, so you get islands. Training accuracy is a perfect 100%. Check the green test curve to see the real story.</>,
        render: () => <KnnFigure k={1} />,
    },
    {
        id: "smooth",
        title: "Smooth it out",
        prompt: <>With k = 15 the noise gets outvoted and the boundary follows the real shape. Hover anywhere to see who votes.</>,
        render: () => <KnnFigure k={15} />,
    },
    {
        id: "too-smooth",
        title: "Too smooth",
        prompt: <>On a spiral, a big k reaches across the arms and blurs the structure away. Slide k down until the spiral reappears, and watch the test accuracy peak.</>,
        render: () => <KnnFigure kind="spiral" noise={0.15} k={35} />,
    },
    {
        id: "metric",
        title: "Shape of “near”",
        prompt: <>Switch between Euclidean, Manhattan and Chebyshev. The dashed neighborhood becomes a circle, a diamond or a square, and the region edges follow.</>,
        render: () => <KnnFigure kind="blobs" noise={0.3} k={7} metric={1} />,
    },
    {
        id: "draw",
        title: "Draw your own",
        prompt: <>Pick a class under “Clicking adds” and click to plant points. Try sneaking an orange island into the blue region and find the k that ignores it.</>,
        render: () => <KnnFigure kind="xor" noise={0.1} k={3} />,
    },
];

export default function Knn() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            No training step: to classify <M>{"\\vx"}</M>, find the <M>{"k"}</M> closest stored points and take a majority vote.
                        </li>
                    </ul>
                    <MathBlock>{"\\hat y(\\vx) = \\arg\\max_c \\; \\big|\\{ i \\in N_k(\\vx) : y_i = c \\}\\big|"}</MathBlock>
                    <ul>
                        <li>
                            Small <M>{"k"}</M> gives low bias and high variance (it overfits the noise). Large <M>{"k"}</M> gives high bias and low variance (it
                            blurs the boundary). Pick <M>{"k"}</M> on held-out data.
                        </li>
                        <li>
                            The metric defines the shape of a neighborhood: <M>{"L_2"}</M> is a circle, <M>{"L_1"}</M> a diamond, <M>{"L_\\infty"}</M> a
                            square. Standardize features first, or the largest-scale feature dominates.
                        </li>
                        <li>The decision regions here are recomputed in WebAssembly for every pixel: about five million distance computations per redraw.</li>
                    </ul>
                </>
            }
        />
    );
}
