import {useMemo, useState} from "react";
import {Button, Segmented, Slider} from "../../components/Controls";
import {Callout, Figure, FigureLayout} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {classRegionColor} from "../../lib/colors";
import {appendRow, drawLabelled, hitPoint, removeRow} from "../../lib/data";
import {heatImage, type Painter, type View} from "../../lib/painter";
import {C, classColor, rgba} from "../../lib/theme";
import {knn_accuracy_curve, knn_grid, knn_query, make_blobs, make_shape, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const G = 200;
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

function KnnFigure() {
    const [kind, setKind] = useState<Kind>("moons");
    const [seed, setSeed] = useState(4);
    const [noise, setNoise] = useState(0.45);
    const [k, setK] = useState(5);
    const [metric, setMetric] = useState(0);
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
        if (!query) return null;
        const r = knn_query(data, query.x, query.y, k, metric);
        const kk = r.length / 2;
        const idx = Array.from(r.subarray(0, kk));
        const votes = new Array(numClasses).fill(0);
        for (const i of idx) votes[data[3 * i + 2]]++;
        return {idx, radius: r[r.length - 1], votes};
    }, [data, query, k, metric, numClasses]);

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
        <Figure n={1} title="Ask the neighbors" hint="Hover to probe · click to add a point of the chosen class">
            <FigureLayout
                sideWidth={300}
                main={
                    <Plot
                        label="k-nearest-neighbour decision regions"
                        view={VIEW}
                        maxHeight={620}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            p.image(regions, VIEW);
                            p.grid(0.25, {axes: false});
                            if (neighbours && query) {
                                for (const i of neighbours.idx) p.line(query.x, query.y, data[3 * i], data[3 * i + 1], {color: rgba(C.white, 0.55), width: 1.2});
                                drawBall(p, query.x, query.y, neighbours.radius, metric, rgba(C.white, 0.75));
                            }
                            drawLabelled(p, data, {r: 4});
                            if (neighbours) for (const i of neighbours.idx) p.circle(data[3 * i], data[3 * i + 1], 7.5, {stroke: C.white, width: 1.5});
                            if (query) p.circle(query.x, query.y, 5, {fill: C.white, stroke: C.canvas, width: 2});
                        }}
                    />
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
        </Figure>
    );
}

export default function Knn() {
    return (
        <>
            <p>
                Here is a classifier with no training step at all. To label a new point, find the <M>{"k"}</M> training points closest to it and let
                them vote. That is <strong>k-nearest neighbors</strong>. It makes no assumption that boundaries are lines, and with enough data it can
                learn almost any shape. In exchange you have to answer two questions it can't answer for you: how many neighbors, and what does "near"
                mean?
            </p>

            <h2 id="algorithm">The algorithm</h2>
            <p>Given a query point <M>{"\\vx"}</M>:</p>
            <ol>
                <li>
                    Compute the distance from <M>{"\\vx"}</M> to every training point.
                </li>
                <li>
                    Keep the <M>{"k"}</M> closest, <M>{"N_k(\\vx)"}</M>.
                </li>
                <li>Predict the most common label among them.</li>
            </ol>
            <MathBlock>{"\\hat y(\\vx) = \\arg\\max_{c} \\; \\big|\\{\\, i \\in N_k(\\vx) : y_i = c \\,\\}\\big|"}</MathBlock>
            <p>
                Doing this for every pixel of the plane paints the <strong>decision regions</strong> below. Each pixel is its own query against every
                training point. That is over five million distance computations for the full picture, recomputed in WebAssembly every time you change
                something.
            </p>

            <KnnFigure />

            <h2 id="k">Choosing k: noise versus blur</h2>
            <p>
                With <M>{"k = 1"}</M>, every training point owns a small territory: the region of space closer to it than to any other point, a{" "}
                <em>Voronoi cell</em>. The boundary follows every quirk of the data, including the noise. Islands appear around mislabeled points, and
                accuracy on the training data is a perfect 100%, since each point is its own nearest neighbor. That number is meaningless. The green
                curve, measured on <em>fresh</em> points from the same distribution, tells the real story.
            </p>
            <p>
                As <M>{"k"}</M> grows, each vote averages over a wider neighborhood. Noise gets outvoted and the boundary smooths out. Push{" "}
                <M>{"k"}</M> too far and the vote starts reaching across into the other class's territory, blurring real structure away. At the extreme,{" "}
                <M>{"k = n"}</M> predicts the majority class everywhere.
            </p>
            <Callout>
                This is the <strong>bias–variance trade-off</strong> in its clearest form. Small <M>{"k"}</M> gives low bias and high variance: the
                model is flexible enough to fit anything, including noise. Large <M>{"k"}</M> gives high bias and low variance: the model is stable but
                too rigid. The best <M>{"k"}</M> is where held-out accuracy peaks, and you can only find it with data the model hasn't seen.
            </Callout>

            <h2 id="distance">What does "near" mean?</h2>
            <p>
                The algorithm is only as good as its distance function. The usual choices are members of one family, the <M>{"L_p"}</M> norms:
            </p>
            <MathBlock>{"d_2 = \\sqrt{\\Delta x^2 + \\Delta y^2}, \\qquad d_1 = |\\Delta x| + |\\Delta y|, \\qquad d_\\infty = \\max\\big(|\\Delta x|, |\\Delta y|\\big)"}</MathBlock>
            <p>
                Each one has a different "ball", the set of points within distance <M>{"r"}</M>. For Euclidean distance it is a circle. For Manhattan it
                is a diamond, and for Chebyshev a square. Switch metrics and watch the dashed neighborhood change shape, and the region boundaries pick up
                45° or axis-aligned edges.
            </p>
            <Callout kind="note" title="Two practical warnings">
                <ul>
                    <li>
                        <strong>Scale matters.</strong> If one feature is measured in meters and another in millimeters, the second will dominate every
                        distance. Standardize features before using k-NN.
                    </li>
                    <li>
                        <strong>High dimensions are strange.</strong> In hundreds of dimensions, nearly all points are roughly the same distance from
                        each other, so "nearest" carries little information. This is the <em>curse of dimensionality</em>, and it is why k-NN works best
                        on low-dimensional or carefully embedded data.
                    </li>
                </ul>
            </Callout>

            <section className="takeaways">
                <h2>Takeaways</h2>
                <ul>
                    <li>k-NN memorizes the training set and classifies by a majority vote of the k closest points. It has no training step and no model.</li>
                    <li>Small k overfits (jagged, noisy regions). Large k underfits (blurred regions). Choose k on held-out data.</li>
                    <li>The distance metric defines the neighborhood's shape and the geometry of the boundaries.</li>
                    <li>All the cost moves to prediction time: every query compares against every stored point.</li>
                </ul>
            </section>
        </>
    );
}
