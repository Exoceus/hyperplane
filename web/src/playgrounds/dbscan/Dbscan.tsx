import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {M, MathBlock} from "../../components/Math";
import {Playground, type Scenario} from "../../components/Playground";
import {Plot, type PlotPointer} from "../../components/Plot";
import {Actions, Group, Rig} from "../../components/Rig";
import {useTimeline} from "../../lib/hooks";
import type {View} from "../../lib/painter";
import {C, classColor, rgba} from "../../lib/theme";
import {dbscan, k_distance, make_blobs, make_gaussian, make_linear, make_shape, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};

type Kind = "moons" | "rings" | "noisy" | "density" | "spiral";
const KINDS: {value: Kind; label: string}[] = [
    {value: "moons", label: "Moons"},
    {value: "rings", label: "Rings"},
    {value: "noisy", label: "Blobs + noise"},
    {value: "density", label: "Mixed density"},
    {value: "spiral", label: "Spiral"},
];

function labelled(xy: Float64Array): Float64Array {
    const out = new Float64Array((xy.length / 2) * 3);
    for (let i = 0, k = 0; i < xy.length; i += 2) {
        out[k++] = xy[i];
        out[k++] = xy[i + 1];
        out[k++] = 0;
    }
    return out;
}

function concat(a: Float64Array, b: Float64Array): Float64Array {
    const out = new Float64Array(a.length + b.length);
    out.set(a);
    out.set(b, a.length);
    return out;
}

function generate(kind: Kind, seed: number): Float64Array {
    switch (kind) {
        case "moons":
            return make_shape("moons", 300, 0.05, seed);
        case "rings":
            return make_shape("circles", 300, 0.05, seed);
        case "noisy":
            return concat(make_blobs(240, 3, 0.08, seed), make_linear(70, 1, 0, 0, 0, 0, seed + 1));
        case "density":
            return concat(labelled(make_gaussian(170, -0.5, 0.25, 0.07, 0.07, 0, seed)), labelled(make_gaussian(110, 0.45, -0.2, 0.3, 0.3, 0, seed + 1)));
        default:
            return make_shape("spiral", 320, 0.04, seed);
    }
}

const NOISE = "#5b6578";

function DbscanInstrument(props: {kind?: Kind; eps?: number; minPts?: number}) {
    const [kind, setKind] = useState<Kind>(props.kind ?? "moons");
    const [seed, setSeed] = useState(3);
    const [eps, setEps] = useState(props.eps ?? 0.12);
    const [minPts, setMinPts] = useState(props.minPts ?? 5);
    const [showEps, setShowEps] = useState(false);
    const data = useMemo(() => generate(kind, seed), [kind, seed]);
    const n = data.length / 3;

    const run = useMemo(() => {
        const r = dbscan(data, 3, eps, minPts);
        const out = {labels: r.labels, kinds: r.kinds, order: r.order, clusters: r.clusters};
        r.free();
        return out;
    }, [data, eps, minPts]);
    const kd = useMemo(() => k_distance(data, 3, Math.max(1, minPts - 1)), [data, minPts]);
    const events = run.order.length / 3;
    const tl = useTimeline(events, 60, {autoplay: true, runKey: run});
    const shown = Math.floor(tl.t + 1e-9);
    const done = tl.atEnd;

    // Cluster of each point at the current moment of the animation (−1 = not yet reached).
    const now = useMemo(() => {
        const out = new Int32Array(n).fill(-1);
        for (let e = 0; e < Math.min(shown, events); e++) out[run.order[3 * e]] = run.order[3 * e + 1];
        return out;
    }, [run, shown, events, n]);

    const noise = run.kinds.filter((k) => k === 0).length;
    const cores = run.kinds.filter((k) => k === 2).length;
    const latest = shown > 0 && shown <= events ? run.order[3 * (shown - 1)] : -1;
    const kdMax = Math.max(0.05, ...kd.slice(0, Math.max(1, Math.floor(kd.length * 0.02))));
    const reset = () => tl.reset();

    const onKd = (e: PlotPointer) => {
        if (e.type === "down" || (e.type === "move" && e.dragging)) {
            setEps(Math.max(0.01, Math.min(0.6, e.y)));
            reset();
        }
    };

    return (
        <Rig
            hint="Clusters spread from core point to core point · change ε or minPts to rerun"
            visual={
                <Plot
                    label="DBSCAN clustering"
                    view={VIEW}
                    maxHeight={580}
                    draw={(p) => {
                        p.clear();
                        p.grid(0.25, {axes: false});
                        if (showEps)
                            for (let i = 0; i < n; i++)
                                if (run.kinds[i] === 2) p.circle(data[3 * i], data[3 * i + 1], eps * p.kx, {fill: rgba(classColor(run.labels[i] + 1), 0.05), stroke: rgba(classColor(run.labels[i] + 1), 0.18), width: 1});
                        // Expansion edges.
                        for (let e = 0; e < Math.min(shown, events); e++) {
                            const parent = run.order[3 * e + 2];
                            if (parent < 0) continue;
                            const i = run.order[3 * e];
                            p.line(data[3 * parent], data[3 * parent + 1], data[3 * i], data[3 * i + 1], {color: rgba(classColor(run.order[3 * e + 1] + 1), 0.35), width: 1});
                        }
                        for (let i = 0; i < n; i++) {
                            const x = data[3 * i];
                            const y = data[3 * i + 1];
                            if (now[i] >= 0) {
                                const col = classColor(now[i] + 1);
                                if (run.kinds[i] === 2) p.point(x, y, col, {r: 4.2});
                                else p.circle(x, y, 3.6, {fill: C.canvas, stroke: col, width: 2});
                            } else if (done && run.kinds[i] === 0) {
                                const s = 3.5 / p.kx;
                                p.line(x - s, y - s, x + s, y + s, {color: NOISE, width: 1.6});
                                p.line(x - s, y + s, x + s, y - s, {color: NOISE, width: 1.6});
                            } else p.circle(x, y, 2.8, {fill: rgba(C.muted, 0.55)});
                        }
                        if (latest >= 0 && !done) p.circle(data[3 * latest], data[3 * latest + 1], eps * p.kx, {stroke: C.yellow, width: 1.6, dash: [4, 4]});
                    }}
                />
            }
            caption={done ? `${run.clusters} clusters · ${cores} core · ${n - cores - noise} border · ${noise} noise points` : `${shown} of ${events} points reached`}
            controls={
                <>
                    <Player timeline={tl} />
                    <Group label="Data">
                        <Segmented value={kind} onChange={(v) => (setKind(v), reset())} options={KINDS} />
                        <Actions>
                            <Button onClick={() => (setSeed(randomSeed()), reset())}>New data</Button>
                        </Actions>
                    </Group>
                    <Group label="DBSCAN">
                        <Slider label={<>radius <M>{"\\varepsilon"}</M></>} value={eps} min={0.02} max={0.5} onChange={(v) => (setEps(v), reset())} color={C.yellow} format={(v) => v.toFixed(3)} />
                        <Slider label="min points (incl. itself)" value={minPts} min={2} max={20} step={1} onChange={(v) => (setMinPts(v), reset())} format={(v) => v.toFixed(0)} />
                        <Toggle label="Show ε-neighborhoods of core points" checked={showEps} onChange={setShowEps} />
                    </Group>
                </>
            }
            readouts={
                <>
                    <Stats>
                        <Stat label="clusters" value={run.clusters} color={C.green} />
                        <Stat label="noise" value={noise} />
                    </Stats>
                    <div className="panel-label">
                        <span>k-distance · click to set the radius</span>
                    </div>
                    <Plot
                        label="Sorted distance to the k-th nearest neighbour"
                        view={{xmin: 0, xmax: kd.length, ymin: 0, ymax: Math.max(kdMax, eps) * 1.12}}
                        height={110}
                        cursor="row-resize"
                        onPointer={onKd}
                        draw={(p) => {
                            p.clear();
                            const pts: number[] = [];
                            kd.forEach((v, i) => pts.push(i, v));
                            p.polyline(pts, {color: C.text2, width: 2});
                            p.line(0, eps, kd.length, eps, {color: C.yellow, width: 1.5, dash: [4, 4]});
                            p.textPx(`ε = ${eps.toFixed(3)}`, p.width - 6, p.sy(eps) - 5, {color: C.yellow, size: 10.5, align: "right", font: "mono"});
                        }}
                    />
                    <p className="note">
                        <span className="c-blue">●</span> core · <span className="c-blue">○</span> border · <span style={{color: NOISE}}>×</span> noise.
                        Points above the yellow line in the k-distance plot will be noise.
                    </p>
                </>
            }
        />
    );
}

const SCENARIOS: Scenario[] = [
    {
        id: "moons",
        title: "Moons, done right",
        prompt: <>Clusters spread from core point to core point, so they can follow any shape, including the two moons that k-means slices in half.</>,
        render: () => <DbscanInstrument />,
    },
    {
        id: "rings",
        title: "Rings",
        prompt: <>Nested rings are two dense, connected shapes. DBSCAN separates them without being told how many clusters there are.</>,
        render: () => <DbscanInstrument kind="rings" eps={0.13} minPts={4} />,
    },
    {
        id: "noise",
        title: "Noise stays noise",
        prompt: <>Scattered background points never reach the density threshold, so they are labeled noise (×) instead of being forced into a cluster.</>,
        render: () => <DbscanInstrument kind="noisy" eps={0.1} minPts={6} />,
    },
    {
        id: "eps-small",
        title: "ε too small",
        prompt: <>With a tiny radius almost nothing counts as dense, so you get fragments and noise everywhere. Drag ε up on the k-distance plot until the moons reconnect.</>,
        render: () => <DbscanInstrument eps={0.045} />,
    },
    {
        id: "eps-big",
        title: "ε too big",
        prompt: <>With a huge radius everything is "dense" and the moons merge into one cluster. The k-distance elbow suggests a better value.</>,
        render: () => <DbscanInstrument eps={0.32} />,
    },
    {
        id: "density",
        title: "Mixed density",
        prompt: <>One tight blob and one sparse one. Any single ε is too big for one or too small for the other. That is DBSCAN's main weakness, and HDBSCAN was built to fix it.</>,
        render: () => <DbscanInstrument kind="density" eps={0.08} minPts={6} />,
    },
];

export default function Dbscan() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            A point is <strong>core</strong> if at least <M>{"\\text{minPts}"}</M> points lie within distance <M>{"\\varepsilon"}</M> of it:
                        </li>
                    </ul>
                    <MathBlock>{"\\big|\\{\\, j : \\lVert \\vx_j - \\vx_i \\rVert \\le \\varepsilon \\,\\}\\big| \\;\\ge\\; \\text{minPts}"}</MathBlock>
                    <ul>
                        <li>
                            Core points within <M>{"\\varepsilon"}</M> of each other belong to the same cluster. Non-core points within <M>{"\\varepsilon"}</M>{" "}
                            of a core point are <strong>border</strong> points. Everything else is <strong>noise</strong>.
                        </li>
                        <li>
                            There is no k to choose, clusters can be any shape, and outliers are left out. In exchange you have to choose{" "}
                            <M>{"\\varepsilon"}</M>, and one radius has to fit every cluster's density.
                        </li>
                        <li>
                            To pick <M>{"\\varepsilon"}</M>, sort every point's distance to its <M>{"(\\text{minPts}-1)"}</M>-th neighbor and look for the
                            elbow, where distances start shooting up into the noise.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
