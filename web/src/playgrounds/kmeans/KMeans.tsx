import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {FigureLayout, Panel} from "../../components/Figure";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {classRegionColor} from "../../lib/colors";
import {ease, lerp, useTimeline} from "../../lib/hooks";
import {heatImage, type Painter, type View} from "../../lib/painter";
import {C, classColor, rgba} from "../../lib/theme";
import {kmeans_elbow, kmeans_init, kmeans_regions, kmeans_run, make_blobs, make_linear, make_shape, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const G = 110;

type Kind = "blobs" | "moons" | "circles" | "uniform";
const KINDS: {value: Kind; label: string}[] = [
    {value: "blobs", label: "Blobs"},
    {value: "moons", label: "Moons"},
    {value: "circles", label: "Rings"},
    {value: "uniform", label: "Uniform"},
];

function generate(kind: Kind, blobs: number, seed: number): Float64Array {
    if (kind === "blobs") return make_blobs(240, blobs, 0.13, seed);
    if (kind === "uniform") return make_linear(240, 1, 0, 0, 0, 0, seed);
    return make_shape(kind, 240, 0.12, seed);
}

/** A centroid: dark disc, colored ring and a cross. */
function drawCentroid(p: Painter, x: number, y: number, color: string, highlight = false) {
    p.circle(x, y, highlight ? 11 : 9.5, {fill: C.canvas, stroke: color, width: 3});
    const d = 4.5;
    p.line(x - d / p.kx, y, x + d / p.kx, y, {color, width: 2.2});
    p.line(x, y - d / p.ky, x, y + d / p.ky, {color, width: 2.2});
}

// ===========================================================================
// Figure 1 — Lloyd's algorithm, phase by phase
// ===========================================================================

function LloydFigure(props: {kind?: Kind; blobs?: number; k?: number; method?: number; init?: number[]}) {
    const [kind, setKind] = useState<Kind>(props.kind ?? "blobs");
    const [blobs, setBlobs] = useState(props.blobs ?? 4);
    const [seed, setSeed] = useState(6);
    const [k, setK] = useState(props.k ?? 4);
    const [method, setMethod] = useState(props.method ?? 2);
    const [initSeed, setInitSeed] = useState(3);
    const [manual, setManual] = useState<Float64Array | null>(props.init ? new Float64Array(props.init) : null);
    const [drag, setDrag] = useState<number | null>(null);

    const data = useMemo(() => generate(kind, blobs, seed), [kind, blobs, seed]);
    const n = data.length / 3;
    const autoInit = useMemo(() => kmeans_init(data, 3, k, method, initSeed), [data, k, method, initSeed]);
    const init = manual && manual.length === k * 2 ? manual : autoInit;

    const run = useMemo(() => {
        const r = kmeans_run(data, 3, init, 60);
        const out = {centroids: r.centroids, assignments: r.assignments, inertia: r.inertia, iterations: r.iterations, converged: r.converged};
        r.free();
        return out;
    }, [data, init]);
    const T = run.iterations;
    const L = 2 * T;
    const tl = useTimeline(L, 1.4, {autoplay: true, runKey: run, delay: 0.6});

    // Decode the current phase.
    const u = Math.min(L, Math.max(0, Math.ceil(tl.t - 1e-9)));
    const f = u === 0 ? 0 : 1 - (u - tl.t);
    const iter = Math.floor((u - 1) / 2);
    const phase: "init" | "assign" | "update" = u === 0 ? "init" : u % 2 === 1 ? "assign" : "update";
    const cAt = (i: number) => run.centroids.subarray(i * k * 2, (i + 1) * k * 2);
    const aAt = (i: number) => (i < 0 ? null : run.assignments.subarray(i * n, (i + 1) * n));

    let centroids: Float64Array;
    if (phase === "update") {
        const a = cAt(iter);
        const b = cAt(iter + 1);
        const e = ease(f);
        centroids = a.map((v, i) => lerp(v, b[i], e));
    } else {
        centroids = cAt(Math.max(0, iter)).slice();
    }
    const assign = phase === "init" ? null : aAt(iter);
    const prevAssign = phase === "assign" ? aAt(iter - 1) : assign;
    const recolor = phase === "assign" ? ease(f) : 1;

    const regions = useMemo(() => {
        const g = kmeans_regions(centroids, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G);
        return heatImage(g, G, G, (c) => classRegionColor(c, 1, 0.16));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [centroids.join(",")]);

    const inertiaNow = u === 0 ? null : run.inertia[u - 1];
    const statusText =
        phase === "init" ? (
            <>
                {k} centroids placed. Drag one to choose your own starting positions.
            </>
        ) : phase === "assign" ? (
            <>
                <b>Assign</b> (iteration {iter + 1}): every point joins the cluster of its <em>nearest</em> centroid. The shaded regions show which
                centroid is nearest.
            </>
        ) : (
            <>
                <b>Update</b> (iteration {iter + 1}): every centroid moves to the <em>mean</em> of the points assigned to it.
            </>
        );

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") {
            let best = -1;
            let bestD = 16 * 16;
            for (let c = 0; c < k; c++) {
                const d = ((centroids[2 * c] - e.x) * e.kx) ** 2 + ((centroids[2 * c + 1] - e.y) * e.ky) ** 2;
                if (d < bestD) {
                    bestD = d;
                    best = c;
                }
            }
            if (best >= 0) {
                setDrag(best);
                tl.reset();
            }
        }
        if (e.type === "move" && drag !== null) {
            const next = Float64Array.from(init);
            next[2 * drag] = Math.max(-1.25, Math.min(1.25, e.x));
            next[2 * drag + 1] = Math.max(-1.25, Math.min(1.25, e.y));
            setManual(next);
        }
        if (e.type === "up") setDrag(null);
    };

    const reinit = () => {
        setManual(null);
        setInitSeed(randomSeed());
        tl.reset();
    };

    return (
        <Instrument hint="Drag any ✛ centroid to rerun from a new starting point">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="k-means clustering step by step"
                            view={VIEW}
                            maxHeight={580}
                            cursor={drag !== null ? "grabbing" : tl.t === 0 ? "grab" : "default"}
                            onPointer={onPointer}
                            draw={(p) => {
                                p.clear();
                                if (phase !== "init") p.image(regions, VIEW, false, phase === "assign" ? ease(f * 1.5) : 1);
                                p.grid(0.25, {axes: false});
                                // Pull lines during the update phase.
                                if (phase === "update" && assign) {
                                    const alpha = 0.35 * Math.sin(Math.PI * Math.min(1, f * 1.2));
                                    for (let i = 0; i < n; i++) {
                                        const c = assign[i];
                                        p.line(data[3 * i], data[3 * i + 1], centroids[2 * c], centroids[2 * c + 1], {color: classColor(c), width: 1, alpha});
                                    }
                                }
                                for (let i = 0; i < n; i++) {
                                    const now = assign ? classColor(assign[i]) : C.muted;
                                    const before = prevAssign ? classColor(prevAssign[i]) : C.muted;
                                    const changed = recolor < 1 && now !== before;
                                    p.point(data[3 * i], data[3 * i + 1], changed && recolor < 0.5 ? before : now, {r: 3.8, alpha: changed ? Math.abs(recolor - 0.5) * 2 : 1});
                                }
                                // Faint trail of each centroid's path so far.
                                const upto = phase === "update" ? iter : Math.max(0, iter);
                                for (let c = 0; c < k; c++) {
                                    const trail: number[] = [];
                                    for (let i = 0; i <= upto; i++) trail.push(run.centroids[i * k * 2 + 2 * c], run.centroids[i * k * 2 + 2 * c + 1]);
                                    trail.push(centroids[2 * c], centroids[2 * c + 1]);
                                    p.polyline(trail, {color: rgba(classColor(c), 0.5), width: 1.5, dash: [3, 4]});
                                }
                                for (let c = 0; c < k; c++) drawCentroid(p, centroids[2 * c], centroids[2 * c + 1], classColor(c), drag === c);
                            }}
                        />
                        <Player
                            timeline={tl}
                            speeds={[0.7, 1.4, 4]}
                            status={
                                tl.atEnd && L > 0 ? (
                                    <span className="c-green">
                                        {run.converged ? `Converged after ${T} iterations: the assignments stopped changing.` : `Stopped after ${T} iterations.`}
                                    </span>
                                ) : (
                                    statusText
                                )
                            }
                        />
                    </>
                }
                side={
                    <div className="controls">
                        <Segmented value={kind} onChange={(v) => (setKind(v), setManual(null), tl.reset())} options={KINDS} />
                        {kind === "blobs" && (
                            <Slider label="hidden blobs in the data" value={blobs} min={2} max={6} step={1} onChange={(v) => (setBlobs(v), setManual(null), tl.reset())} format={(v) => v.toFixed(0)} />
                        )}
                        <Slider label={<>clusters <M>{"k"}</M></>} value={k} min={1} max={8} step={1} onChange={(v) => (setK(v), setManual(null), tl.reset())} format={(v) => v.toFixed(0)} color={C.yellow} />
                        <Segmented
                            label="initialization"
                            value={method}
                            onChange={(v) => (setMethod(v), setManual(null), tl.reset())}
                            options={[
                                {value: 2, label: "Anywhere", title: "uniformly random positions"},
                                {value: 0, label: "Random points", title: "k distinct data points"},
                                {value: 1, label: "k-means++", title: "spread-out data points"},
                            ]}
                        />
                        <Stats>
                            <Stat label="iteration" value={phase === "init" ? "—" : iter + 1} />
                            <Stat label="inertia J" value={inertiaNow === null ? "—" : inertiaNow.toFixed(2)} color={C.yellow} />
                        </Stats>
                        <LineChart
                            label="Inertia after every phase"
                            series={[{values: run.inertia, color: C.yellow, x0: 1, upto: Math.max(0, tl.t - 1), dots: true}]}
                            height={100}
                            xMin={1}
                            xMax={Math.max(2, L)}
                            xLabel="phase"
                            yLabel="J"
                        />
                        <div className="control-row">
                            <Button onClick={reinit}>Re-initialize</Button>
                            <Button onClick={() => (setSeed(randomSeed()), setManual(null), tl.reset())}>New data</Button>
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Figure 2 — choosing k
// ===========================================================================

const KMAX = 10;

function ElbowFigure() {
    const [truth, setTruth] = useState(3);
    const [seed, setSeed] = useState(21);
    const [k, setK] = useState(3);
    const data = useMemo(() => make_blobs(260, truth, 0.11, seed), [truth, seed]);
    const elbow = useMemo(() => kmeans_elbow(data, 3, KMAX, 6, seed), [data, seed]);
    const fit = useMemo(() => {
        let best: {assign: Float64Array; centroids: Float64Array; j: number} | null = null;
        for (let r = 0; r < 6; r++) {
            const run = kmeans_run(data, 3, kmeans_init(data, 3, k, 1, seed + r * 101), 100);
            const T = run.iterations;
            const j = run.inertia[run.inertia.length - 1];
            if (!best || j < best.j) {
                const n = data.length / 3;
                best = {assign: run.assignments.slice((T - 1) * n, T * n), centroids: run.centroids.slice(T * k * 2, (T + 1) * k * 2), j};
            }
            run.free();
        }
        return best!;
    }, [data, k, seed]);

    return (
        <Instrument hint="Click the curve to choose k">
            <div className="duo">
                <Panel label={`k = ${k}`} right={`J = ${fit.j.toFixed(2)}`}>
                    <Plot
                        label="Best k-means clustering for the chosen k"
                        view={VIEW}
                        draw={(p) => {
                            p.clear();
                            p.grid(0.25, {axes: false});
                            const n = data.length / 3;
                            for (let i = 0; i < n; i++) p.point(data[3 * i], data[3 * i + 1], classColor(fit.assign[i]), {r: 3.6});
                            for (let c = 0; c < k; c++) drawCentroid(p, fit.centroids[2 * c], fit.centroids[2 * c + 1], classColor(c));
                        }}
                    />
                </Panel>
                <Panel label="Inertia vs. k (best of 6 runs)">
                    <LineChart
                        label="Elbow curve"
                        series={[{values: elbow, color: C.yellow, x0: 1, dots: true}]}
                        height={300}
                        xMin={1}
                        xMax={KMAX}
                        yMin={0}
                        marker={k}
                        xLabel="k"
                        onPick={(x) => setK(Math.max(1, Math.min(KMAX, Math.round(x))))}
                    />
                    <div className="controls" style={{marginTop: 10}}>
                        <Slider label="clusters actually in the data" value={truth} min={1} max={7} step={1} onChange={setTruth} format={(v) => v.toFixed(0)} />
                        <div className="control-row">
                            <Button onClick={() => setSeed(randomSeed())}>New data</Button>
                        </div>
                    </div>
                </Panel>
            </div>
        </Instrument>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

const SCENARIOS: Scenario[] = [
    {
        id: "blobs",
        title: "Four blobs",
        prompt: <>Step through it. <b>Assign</b>: every point joins its nearest centroid (shaded regions). <b>Update</b>: every centroid jumps to the mean of its points. Repeat until nothing changes.</>,
        render: () => <LloydFigure />,
    },
    {
        id: "bad-start",
        title: "Bad start",
        prompt: <>All four centroids start huddled in one corner. Play it: k-means converges, but to a poor local minimum. Then switch the initialization to k-means++ and compare.</>,
        render: () => <LloydFigure init={[-1.1, 1.1, -0.95, 1.15, -1.15, 0.95, -0.9, 0.95]} />,
    },
    {
        id: "wrong-k",
        title: "Wrong k",
        prompt: <>Four real blobs, six centroids. k-means always returns exactly k clusters, so some blobs get split in two. Slide k to see the inertia trade-off.</>,
        render: () => <LloydFigure k={6} method={1} />,
    },
    {
        id: "moons",
        title: "Moons break it",
        prompt: <>Two obvious groups that aren't round. Nearest-centroid boundaries are always straight, so k-means slices right across both moons.</>,
        render: () => <LloydFigure kind="moons" k={2} method={1} />,
    },
    {
        id: "uniform",
        title: "No structure at all",
        prompt: <>Uniform noise, with no clusters to find. k-means still confidently carves it into k cells. A clustering always exists, even when it means nothing.</>,
        render: () => <LloydFigure kind="uniform" k={5} method={1} />,
    },
    {
        id: "elbow",
        title: "How many clusters?",
        prompt: <>Inertia always drops as k grows. Look for the elbow where adding clusters stops paying off, and click the curve to try each k.</>,
        render: () => <ElbowFigure />,
    },
];

export default function KMeans() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            Represent each cluster by its centroid <M>{"\\boldsymbol\\mu_c"}</M> and minimize the total squared distance from points to their
                            centroid (the inertia):
                        </li>
                    </ul>
                    <MathBlock>{"J = \\sum_i \\big\\lVert \\vx_i - \\boldsymbol\\mu_{c(i)} \\big\\rVert^2"}</MathBlock>
                    <ul>
                        <li>
                            <strong>Lloyd's algorithm</strong> alternates two exact half-steps: assign each point to its nearest centroid, then move each
                            centroid to the mean of its points. Neither can increase <M>{"J"}</M>, so it always converges.
                        </li>
                        <li>
                            It converges to a <em>local</em> minimum that depends on the start. k-means++ spreads the initial centroids out, which makes bad
                            minima much rarer.
                        </li>
                        <li>Cluster boundaries are straight (Voronoi cells), so k-means assumes round, similarly sized clusters, and you have to choose k.</li>
                    </ul>
                </>
            }
        />
    );
}
