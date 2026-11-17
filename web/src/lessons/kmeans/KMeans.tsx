import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {Callout, Figure, FigureLayout, Panel} from "../../components/Figure";
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

function LloydFigure() {
    const [kind, setKind] = useState<Kind>("blobs");
    const [blobs, setBlobs] = useState(4);
    const [seed, setSeed] = useState(6);
    const [k, setK] = useState(4);
    const [method, setMethod] = useState(2);
    const [initSeed, setInitSeed] = useState(3);
    const [manual, setManual] = useState<Float64Array | null>(null);
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
    const tl = useTimeline(L, 1.4);

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
                {k} centroids placed. Drag them to choose your own starting positions, then press play.
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
        <Figure n={1} title="Lloyd's algorithm: assign, update, repeat" hint="Drag the ✛ centroids before pressing play">
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
        </Figure>
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
        <Figure n={2} title="How many clusters?" hint="Click the curve to choose k">
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
        </Figure>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

export default function KMeans() {
    return (
        <>
            <p>
                Every lesson so far had labels: each point arrived marked blue or orange, and the model learned to reproduce the marks. Often you have no
                labels at all, only data, and the question becomes <em>is there structure here?</em> <strong>k-means</strong> is the classic first answer.
                Tell it how many groups to look for, and it finds <M>{"k"}</M> centers that summarize the data as well as possible.
            </p>

            <h2 id="objective">What counts as a good clustering?</h2>
            <p>
                k-means represents each cluster by a single point, its <strong>centroid</strong> <M>{"\\boldsymbol\\mu_c"}</M>, and assigns each data point
                to exactly one cluster. A clustering is good when every point sits close to its own centroid. Formally, it minimizes the total squared
                distance, called the <strong>inertia</strong>:
            </p>
            <MathBlock>{"J = \\sum_{i=1}^n \\big\\lVert \\vx_i - \\boldsymbol\\mu_{c(i)} \\big\\rVert^2"}</MathBlock>
            <p>
                Here <M>{"c(i)"}</M> is the cluster of point <M>{"i"}</M>. Finding the very best clustering is NP-hard in general. But there is a simple
                procedure that gets good answers fast: improve the assignments with the centroids fixed, then improve the centroids with the
                assignments fixed, and repeat.
            </p>

            <h2 id="lloyd">Lloyd's algorithm</h2>
            <ol>
                <li>
                    <strong>Assign.</strong> Hold the centroids still. The best cluster for each point is simply the nearest centroid. The plane splits
                    into regions of "closest to centroid <M>{"c"}</M>", a <em>Voronoi diagram</em>, shaded below.
                </li>
                <li>
                    <strong>Update.</strong> Hold the assignments still. For each cluster, the point minimizing total squared distance to its members is
                    their <strong>mean</strong>, so every centroid jumps to the center of mass of its points.
                </li>
            </ol>

            <LloydFigure />

            <p>
                Why must this stop? Each phase solves its half of the problem exactly, so neither can increase <M>{"J"}</M>. The inertia curve only ever
                goes down. There are finitely many ways to assign <M>{"n"}</M> points to <M>{"k"}</M> clusters, and the algorithm never revisits an
                assignment (that would require <M>{"J"}</M> to go back up), so it must eventually reach a step where nothing changes.
            </p>

            <Callout kind="try">
                <ul>
                    <li>
                        With <em>Anywhere</em> initialization, re-initialize a few times. Sometimes two centroids end up sharing one blob while another
                        blob has none. The algorithm converges, but to a <strong>local minimum</strong>. Lloyd's algorithm can't climb out, because doing so
                        would require <M>{"J"}</M> to rise.
                    </li>
                    <li>
                        Switch to <em>k-means++</em>, which picks each initial centroid from the data with probability proportional to its squared
                        distance from the centroids chosen so far. Spreading the starts out makes bad local minima much rarer.
                    </li>
                </ul>
            </Callout>

            <h2 id="limits">What k-means assumes</h2>
            <p>
                Because each point simply goes to its nearest center, the boundaries between clusters are always straight lines (the edges of the Voronoi
                cells). Each cluster is implicitly a round blob. Try <em>Moons</em> or <em>Rings</em>: the data has two obvious groups, but they aren't
                round, and k-means slices straight through them. Try <em>Uniform</em>: k-means will still find <M>{"k"}</M> clusters, because it always
                does, even when there is no structure at all.
            </p>

            <h2 id="choosing-k">Choosing k</h2>
            <p>
                You have to choose <M>{"k"}</M>, and the objective can't choose it for you. Inertia always drops as <M>{"k"}</M> grows, and it reaches
                zero at <M>{"k = n"}</M>, with every point its own cluster. A common heuristic is the <strong>elbow</strong>: plot the best inertia for
                each <M>{"k"}</M> and look for the bend where adding a cluster stops paying off.
            </p>

            <ElbowFigure />

            <Callout kind="note" title="Beyond the elbow">
                The elbow is often ambiguous. Other criteria, such as silhouette scores, the gap statistic, or a likelihood penalized by BIC in{" "}
                <em>Gaussian mixture models</em>, give more principled answers. Mixture models also replace k-means' hard assignments with probabilities.
            </Callout>

            <section className="takeaways">
                <h2>Takeaways</h2>
                <ul>
                    <li>k-means summarizes data with k centroids, minimizing the total squared distance from points to their centroid.</li>
                    <li>Lloyd's algorithm alternates two exact half-steps, assign to the nearest and move to the mean, so inertia never increases and it always converges.</li>
                    <li>It converges to a local minimum that depends on initialization. k-means++ helps.</li>
                    <li>It assumes round, similarly sized clusters, and it needs k chosen in advance.</li>
                </ul>
            </section>
        </>
    );
}
