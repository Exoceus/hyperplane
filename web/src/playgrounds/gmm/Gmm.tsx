import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {Panel} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Playground, type Scenario} from "../../components/Playground";
import {Plot} from "../../components/Plot";
import {Actions, Group, Rig} from "../../components/Rig";
import {lerp, useTimeline} from "../../lib/hooks";
import {heatImage, type Painter, type View} from "../../lib/painter";
import {C, CLASS_RGB, classColor, sequential} from "../../lib/theme";
import {gmm_grid, gmm_run, kmeans_init, kmeans_run, make_blobs, make_gaussian, make_shape, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const CS = 6; // component stride: [mx, my, sxx, sxy, syy, weight]
const G = 110;

type Kind = "stretched" | "overlap" | "sizes" | "blobs" | "moons";
const KINDS: {value: Kind; label: string}[] = [
    {value: "stretched", label: "Stretched"},
    {value: "overlap", label: "Overlapping"},
    {value: "sizes", label: "Big & small"},
    {value: "blobs", label: "Blobs"},
    {value: "moons", label: "Moons"},
];

function concat(...parts: Float64Array[]): Float64Array {
    const out = new Float64Array(parts.reduce((s, p) => s + p.length, 0));
    let o = 0;
    for (const p of parts) {
        out.set(p, o);
        o += p.length;
    }
    return out;
}

function strip(labelled: Float64Array): Float64Array {
    const out = new Float64Array((labelled.length / 3) * 2);
    for (let i = 0, k = 0; i < labelled.length; i += 3) {
        out[k++] = labelled[i];
        out[k++] = labelled[i + 1];
    }
    return out;
}

function generate(kind: Kind, s: number): Float64Array {
    switch (kind) {
        case "stretched":
            return concat(make_gaussian(110, -0.5, 0.5, 0.42, 0.06, 0.45, s), make_gaussian(110, 0.5, 0.45, 0.36, 0.07, -0.85, s + 1), make_gaussian(110, 0.05, -0.55, 0.48, 0.07, 0.05, s + 2));
        case "overlap":
            return concat(make_gaussian(160, -0.25, 0.0, 0.3, 0.28, 0, s), make_gaussian(160, 0.3, 0.1, 0.24, 0.24, 0, s + 1));
        case "sizes":
            return concat(make_gaussian(220, -0.35, 0.0, 0.42, 0.38, 0, s), make_gaussian(45, 0.65, 0.6, 0.06, 0.06, 0, s + 1), make_gaussian(45, 0.75, -0.55, 0.07, 0.05, 0, s + 2));
        case "blobs":
            return strip(make_blobs(320, 4, 0.12, s));
        default:
            return strip(make_shape("moons", 300, 0.12, s));
    }
}

/** Ellipse at `k` standard deviations for a 2×2 covariance. */
function ellipse(p: Painter, mx: number, my: number, sxx: number, sxy: number, syy: number, k: number, opts: {color: string; width: number; fill?: string; fillAlpha?: number; dash?: number[]}) {
    const tr = sxx + syy;
    const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - (sxx * syy - sxy * sxy)));
    const l1 = tr / 2 + disc;
    const l2 = Math.max(0, tr / 2 - disc);
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const a = k * Math.sqrt(l1);
    const b = k * Math.sqrt(l2);
    const pts: number[] = [];
    for (let i = 0; i <= 72; i++) {
        const t = (i / 72) * Math.PI * 2;
        const u = a * Math.cos(t);
        const v = b * Math.sin(t);
        pts.push(mx + u * Math.cos(th) - v * Math.sin(th), my + u * Math.sin(th) + v * Math.cos(th));
    }
    p.polyline(pts, {close: true, color: opts.color, width: opts.width, fill: opts.fill, fillAlpha: opts.fillAlpha, dash: opts.dash});
}

interface Run {
    params: Float64Array;
    resp: Float64Array;
    ll: Float64Array;
    iterations: number;
    converged: boolean;
}

function fit(points: Float64Array, k: number, seed: number): Run {
    const r = gmm_run(points, 2, k, seed, 300);
    const out = {params: r.params, resp: r.responsibilities, ll: r.logLikelihood, iterations: r.iterations, converged: r.converged};
    r.free();
    return out;
}

function GmmInstrument(props: {kind?: Kind; k?: number}) {
    const [kind, setKind] = useState<Kind>(props.kind ?? "stretched");
    const [dataSeed, setDataSeed] = useState(5);
    const [k, setK] = useState(props.k ?? 3);
    const [initSeed, setInitSeed] = useState(2);
    const [density, setDensity] = useState(false);
    const points = useMemo(() => generate(kind, dataSeed), [kind, dataSeed]);
    const n = points.length / 2;
    const run = useMemo(() => fit(points, k, initSeed), [points, k, initSeed]);
    const T = run.iterations;
    const tl = useTimeline(T, 4, {autoplay: true, runKey: run});

    const i0 = Math.min(T, Math.floor(tl.t));
    const i1 = Math.min(T, i0 + 1);
    const s = tl.t - i0;
    const comp = (c: number, field: number) => lerp(run.params[(i0 * k + c) * CS + field], run.params[(i1 * k + c) * CS + field], s);
    const current = useMemo(() => {
        const out = new Float64Array(k * CS);
        for (let c = 0; c < k; c++) for (let f = 0; f < CS; f++) out[c * CS + f] = comp(c, f);
        return out;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [run, k, i0, i1, s]);

    const densityImg = useMemo(() => {
        if (!density) return null;
        const g = gmm_grid(current, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G);
        let max = 0;
        for (let i = 1; i < g.length; i += 2) max = Math.max(max, g[i]);
        return heatImage(g, G, G, (_, i) => sequential(0.05 + 0.75 * Math.sqrt(g[2 * i + 1] / max)).map((v) => v * 0.85) as [number, number, number], 1);
    }, [density, current]);

    const reset = () => tl.reset();
    const ll = run.ll[i0];

    return (
        <Rig
            hint="Each EM iteration re-weights the points (E) and refits every ellipse (M); change anything to rerun"
            visual={
                <Plot
                    label="Gaussian mixture components fitting data"
                    view={VIEW}
                    maxHeight={580}
                    draw={(p) => {
                        p.clear();
                        if (densityImg) p.image(densityImg, VIEW);
                        p.grid(0.25, {axes: false});
                        // Points blended by responsibility.
                        for (let i = 0; i < n; i++) {
                            let r = 0;
                            let g = 0;
                            let b = 0;
                            for (let c = 0; c < k; c++) {
                                const w = lerp(run.resp[(i0 * n + i) * k + c], run.resp[(i1 * n + i) * k + c], s);
                                const rgb = CLASS_RGB[(c + 1) % CLASS_RGB.length];
                                r += w * rgb[0];
                                g += w * rgb[1];
                                b += w * rgb[2];
                            }
                            p.point(points[2 * i], points[2 * i + 1], `rgb(${r | 0},${g | 0},${b | 0})`, {r: 3.6});
                        }
                        for (let c = 0; c < k; c++) {
                            const col = classColor(c + 1);
                            const [mx, my, sxx, sxy, syy, w] = [0, 1, 2, 3, 4, 5].map((f) => current[c * CS + f]);
                            ellipse(p, mx, my, sxx, sxy, syy, 2, {color: col, width: 1.4, fill: col, fillAlpha: 0.06 + w * 0.15, dash: [5, 4]});
                            ellipse(p, mx, my, sxx, sxy, syy, 1, {color: col, width: 2.2});
                            p.circle(mx, my, 5.5, {fill: C.canvas, stroke: col, width: 2.5});
                            p.text(`${Math.round(w * 100)}%`, mx, my, {color: col, size: 11, font: "mono", dx: 9, dy: -8});
                        }
                    }}
                />
            }
            caption={tl.atEnd && T > 0 ? (run.converged ? `Converged after ${T} iterations.` : `Stopped after ${T} iterations.`) : `Iteration ${i0} of ${T}`}
            controls={
                <>
                    <Player timeline={tl} />
                    <Group label="Data">
                        <Segmented value={kind} onChange={(v) => (setKind(v), reset())} options={KINDS} />
                        <Actions>
                            <Button onClick={() => (setDataSeed(randomSeed()), reset())}>New data</Button>
                        </Actions>
                    </Group>
                    <Group label="Mixture">
                        <Slider label={<>components <M>{"k"}</M></>} value={k} min={1} max={7} step={1} onChange={(v) => (setK(v), reset())} format={(v) => v.toFixed(0)} color={C.yellow} />
                        <Toggle label="Show mixture density" checked={density} onChange={setDensity} />
                        <Actions>
                            <Button onClick={() => (setInitSeed(randomSeed()), reset())}>Re-initialize</Button>
                        </Actions>
                    </Group>
                </>
            }
            readouts={
                <>
                    <Stats>
                        <Stat label="iteration" value={i0} />
                        <Stat label="log-likelihood / point" value={ll.toFixed(3)} color={C.yellow} />
                    </Stats>
                    <LineChart
                        label="Log-likelihood per iteration"
                        series={[{values: run.ll, color: C.yellow, upto: tl.t, dots: T < 40}]}
                        height={100}
                        yMin={T >= 2 ? run.ll[1] - 0.15 * Math.max(0.05, run.ll[T] - run.ll[1]) : undefined}
                        xLabel="iteration"
                        yLabel="log L"
                    />
                    <p className="note">
                        Ellipses are 1σ and 2σ contours, and the labels are mixing weights <M>{"\\pi_k"}</M>. Each point's color mixes its
                        responsibilities, so blended colors mean uncertain membership.
                    </p>
                </>
            }
        />
    );
}

function CompareInstrument() {
    const [seed, setSeed] = useState(5);
    const points = useMemo(() => generate("stretched", seed), [seed]);
    const n = points.length / 2;
    const km = useMemo(() => {
        const r = kmeans_run(points, 2, kmeans_init(points, 2, 3, 1, seed), 100);
        const a = r.assignments.slice((r.iterations - 1) * n, r.iterations * n);
        const cents = r.centroids.slice(r.iterations * 6, (r.iterations + 1) * 6);
        r.free();
        return {a, cents};
    }, [points, seed, n]);
    const gm = useMemo(() => fit(points, 3, seed), [points, seed]);
    const last = gm.params.slice(gm.iterations * 3 * CS, (gm.iterations + 1) * 3 * CS);
    const resp = gm.resp.slice(gm.iterations * n * 3, (gm.iterations + 1) * n * 3);

    return (
        <Rig
            hint="Same stretched clusters, same k = 3"
            visual={
                <div className="duo">
                    <Panel label="k-means: nearest centroid">
                        <Plot
                            label="k-means clustering of stretched clusters"
                            view={VIEW}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25, {axes: false});
                                for (let i = 0; i < n; i++) p.point(points[2 * i], points[2 * i + 1], classColor(km.a[i] + 1), {r: 3.6});
                                for (let c = 0; c < 3; c++) p.circle(km.cents[2 * c], km.cents[2 * c + 1], 6, {fill: C.canvas, stroke: classColor(c + 1), width: 2.5});
                            }}
                        />
                    </Panel>
                    <Panel label="Gaussian mixture: most likely component">
                        <Plot
                            label="Gaussian mixture clustering of stretched clusters"
                            view={VIEW}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25, {axes: false});
                                for (let i = 0; i < n; i++) {
                                    let best = 0;
                                    for (let c = 1; c < 3; c++) if (resp[i * 3 + c] > resp[i * 3 + best]) best = c;
                                    p.point(points[2 * i], points[2 * i + 1], classColor(best + 1), {r: 3.6});
                                }
                                for (let c = 0; c < 3; c++) {
                                    const [mx, my, sxx, sxy, syy] = [0, 1, 2, 3, 4].map((f) => last[c * CS + f]);
                                    ellipse(p, mx, my, sxx, sxy, syy, 2, {color: classColor(c + 1), width: 1.6});
                                }
                            }}
                        />
                    </Panel>
                </div>
            }
            controls={
                <Group label="Data">
                    <Actions>
                        <Button onClick={() => setSeed(randomSeed())}>New data</Button>
                    </Actions>
                </Group>
            }
            readouts={
                <span className="note">k-means can only draw straight Voronoi boundaries between round clusters. A GMM learns each cluster's shape.</span>
            }
        />
    );
}

const SCENARIOS: Scenario[] = [
    {
        id: "stretched",
        title: "Stretched clusters",
        prompt: <>Each iteration, the ellipses shift, rotate and stretch to fit the long, thin clusters. That is something k-means' round clusters can't do.</>,
        render: () => <GmmInstrument />,
    },
    {
        id: "soft",
        title: "Soft membership",
        prompt: <>Two overlapping clouds. Points in the overlap get blended colors: the model is unsure, and it says so. Toggle the density to see the combined bumps.</>,
        render: () => <GmmInstrument kind="overlap" k={2} />,
    },
    {
        id: "sizes",
        title: "Big & small",
        prompt: <>One wide cluster and two tiny ones. The mixing weights (percentages) and covariances let components differ in size and population.</>,
        render: () => <GmmInstrument kind="sizes" k={3} />,
    },
    {
        id: "vs-kmeans",
        title: "k-means vs. GMM",
        prompt: <>Same data, same k. k-means slices the long clusters with straight boundaries. The mixture model follows their shapes.</>,
        render: () => <CompareInstrument />,
    },
    {
        id: "too-many",
        title: "Too many components",
        prompt: <>Seven components for four blobs: extra ones split clusters or shrink onto a few points. The likelihood always improves with more components, so it can't choose k on its own.</>,
        render: () => <GmmInstrument kind="blobs" k={7} />,
    },
    {
        id: "moons",
        title: "Non-Gaussian shapes",
        prompt: <>Ellipses can't bend. With k = 2 they straddle the moons. Raise k and the Gaussians tile each moon piece by piece.</>,
        render: () => <GmmInstrument kind="moons" k={2} />,
    },
];

export default function Gmm() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>Model the data as a weighted sum of Gaussians, each with its own mean, covariance (ellipse) and weight:</li>
                    </ul>
                    <MathBlock>{"p(\\vx) = \\sum_{k} \\pi_k\\, \\mathcal N(\\vx \\mid \\boldsymbol\\mu_k, \\Sigma_k)"}</MathBlock>
                    <ul>
                        <li>
                            <strong>E-step:</strong> compute each point's responsibilities{" "}
                            <M>{"r_{ik} = \\pi_k \\mathcal N(\\vx_i\\mid\\boldsymbol\\mu_k,\\Sigma_k) / p(\\vx_i)"}</M>, a soft assignment.
                        </li>
                        <li>
                            <strong>M-step:</strong> refit every component as a responsibility-weighted mean and covariance, with{" "}
                            <M>{"\\pi_k = \\tfrac1n\\sum_i r_{ik}"}</M>.
                        </li>
                        <li>
                            Each EM iteration never decreases the likelihood, but it can still stall in a local optimum. k-means is the limiting case with
                            hard assignments and identical round covariances.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
