import {useEffect, useMemo, useRef, useState} from "react";
import {Button, Icon, Slider, Stat, Stats} from "../../components/Controls";
import {FigureLayout, Panel} from "../../components/Figure";
import {M, MathBlock} from "../../components/Math";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {Plot} from "../../components/Plot";
import {useAnimationFrame, useInView} from "../../lib/hooks";
import type {View} from "../../lib/painter";
import {C, classColor, hexToRgb, mix, type RGB} from "../../lib/theme";
import {make_clusters_nd, make_swiss_roll, pca, randomSeed, Tsne} from "../../lib/wasm";

type Kind = "orthogonal" | "sizes" | "distances" | "hierarchy" | "swiss";

interface Dataset {
    points: Float64Array;
    stride: number;
    dim: number;
    /** Category per point, or a continuous value in [0, 1] when `continuous`. */
    color: Float64Array;
    continuous: boolean;
    description: string;
}

function axis(dim: number, k: number, scale: number): number[] {
    const v = new Array(dim).fill(0);
    v[k] = scale;
    return v;
}

function build(kind: Kind, seed: number): Dataset {
    const dim = 10;
    const fromClusters = (n: number, centers: number[][], spreads: number[], desc: string): Dataset => {
        const pts = make_clusters_nd(n, new Float64Array(centers.flat()), new Float64Array(spreads), dim, seed);
        const color = new Float64Array(n);
        for (let i = 0; i < n; i++) color[i] = pts[i * (dim + 1) + dim];
        return {points: pts, stride: dim + 1, dim, color, continuous: false, description: desc};
    };
    switch (kind) {
        case "orthogonal":
            return fromClusters(
                300,
                Array.from({length: 6}, (_, k) => axis(dim, k, 4)),
                new Array(6).fill(0.55),
                "6 Gaussian clusters in 10D, one along each of 6 axes"
            );
        case "sizes":
            return fromClusters(240, [axis(dim, 0, 12), axis(dim, 0, -12)], [0.3, 3], "2 clusters in 10D; the right one is 10× wider");
        case "distances":
            return fromClusters(240, [axis(dim, 0, 0), axis(dim, 0, 3), axis(dim, 1, 25)], [0.6, 0.6, 0.6], "3 clusters in 10D: two close together, one 8× farther away");
        case "hierarchy": {
            const centers: number[][] = [];
            for (let g = 0; g < 3; g++)
                for (let s = 0; s < 3; s++) {
                    const c = axis(dim, g, 14);
                    c[3 + s] = 2.6;
                    centers.push(c);
                }
            return fromClusters(360, centers, new Array(9).fill(0.45), "3 groups × 3 sub-clusters in 10D");
        }
        default: {
            const pts = make_swiss_roll(400, 0.02, seed);
            const color = new Float64Array(400);
            for (let i = 0; i < 400; i++) color[i] = pts[i * 4 + 3];
            return {points: pts, stride: 4, dim: 3, color, continuous: true, description: "A 2D sheet rolled up in 3D, colored by position along the roll"};
        }
    }
}

const RAMP = [C.blue, C.purple, C.pink, C.orange, C.yellow].map(hexToRgb);
function rampColor(t: number): string {
    const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(x));
    const [r, g, b] = mix(RAMP[i], RAMP[i + 1], x - i) as RGB;
    return `rgb(${r | 0},${g | 0},${b | 0})`;
}

function pointColor(d: Dataset, i: number) {
    return d.continuous ? rampColor(d.color[i]) : classColor(d.color[i] + 1);
}

function project2(d: Dataset): Float64Array {
    const n = d.points.length / d.stride;
    const raw = new Float64Array(n * d.dim);
    for (let i = 0; i < n; i++) for (let k = 0; k < d.dim; k++) raw[i * d.dim + k] = d.points[i * d.stride + k];
    const r = pca(raw, d.dim);
    const D = d.dim;
    const mean = r.subarray(0, D);
    const vecs = r.subarray(D + D * D + D);
    const out = new Float64Array(n * 2);
    for (let i = 0; i < n; i++)
        for (let c = 0; c < 2; c++) {
            let s = 0;
            for (let k = 0; k < D; k++) s += (raw[i * D + k] - mean[k]) * vecs[c * D + k];
            out[2 * i + c] = s;
        }
    return out;
}

function fit(y: ArrayLike<number>): number {
    let r = 1e-6;
    for (let i = 0; i < y.length; i++) r = Math.max(r, Math.abs(y[i]));
    return r * 1.12;
}

function scatter(p: import("../../lib/painter").Painter, y: ArrayLike<number>, d: Dataset) {
    p.clear();
    p.grid(p.view.xmax / 4, {axes: false});
    const n = y.length / 2;
    for (let i = 0; i < n; i++) p.point(y[2 * i], y[2 * i + 1], pointColor(d, i), {r: 3.6});
}

function TsneInstrument(props: {kind?: Kind; perplexity?: number}) {
    const kind = props.kind ?? "orthogonal";
    const [seed, setSeed] = useState(4);
    const [perplexity, setPerplexity] = useState(props.perplexity ?? 30);
    const [running, setRunning] = useState(true);
    const data = useMemo(() => build(kind, seed), [kind, seed]);
    const pcaView = useMemo(() => project2(data), [data]);
    const [frame, setFrame] = useState<{y: Float64Array; iter: number; r: number; kl: number} | null>(null);
    const [restart, setRestart] = useState(0);
    const [ref, visible] = useInView<HTMLDivElement>();
    const sim = useRef<Tsne | null>(null);
    const scale = useRef(1);
    const klAt = useRef({iter: -1000, kl: NaN});

    useEffect(() => {
        const t = new Tsne(data.points, data.stride, data.dim, perplexity, data.points.length / data.stride > 300 ? 120 : 80, randomSeed());
        sim.current = t;
        scale.current = 1e-3;
        klAt.current = {iter: -1000, kl: NaN};
        return () => {
            t.free();
            if (sim.current === t) sim.current = null;
        };
    }, [data, perplexity, restart]);

    useAnimationFrame(() => {
        const t = sim.current;
        if (!t) return;
        if (running && t.iteration < 1000) t.step(4);
        else if (frame && frame.iter === t.iteration) return;
        const y = t.embedding();
        const target = fit(y);
        scale.current = scale.current < 1e-2 ? target : scale.current + (target - scale.current) * 0.15;
        if (t.iteration - klAt.current.iter >= 40 || (t.iteration >= 1000 && klAt.current.iter < 1000)) klAt.current = {iter: t.iteration, kl: t.kl()};
        setFrame({y, iter: t.iteration, r: scale.current, kl: klAt.current.kl});
    }, visible);

    const pr = fit(pcaView);
    const pv: View = {xmin: -pr, xmax: pr, ymin: -pr, ymax: pr};
    const r = frame?.r ?? 1;
    const tv: View = {xmin: -r, xmax: r, ymin: -r, ymax: r};

    return (
        <Instrument hint={data.description}>
            <div ref={ref}>
                <FigureLayout
                    sideWidth={280}
                    main={
                        <div className="duo">
                            <Panel label="PCA: best linear 2D view">
                                <Plot label="PCA projection of high-dimensional data" view={pv} draw={(p) => scatter(p, pcaView, data)} />
                            </Panel>
                            <Panel label="t-SNE" right={`iteration ${frame?.iter ?? 0}${(frame?.iter ?? 0) < 100 ? " · exaggerating" : ""}`}>
                                <Plot label="t-SNE embedding, optimizing live" view={tv} draw={(p) => (frame ? scatter(p, frame.y, data) : p.clear())} />
                            </Panel>
                        </div>
                    }
                    side={
                        <div className="controls">
                            <Slider label="perplexity (≈ neighbors considered)" value={perplexity} min={2} max={80} step={1} onChange={setPerplexity} color={C.yellow} format={(v) => v.toFixed(0)} />
                            <div className="control-row">
                                <Button variant="primary" onClick={() => setRunning(!running)} title={running ? "Pause" : "Play"}>
                                    {running ? Icon.pause : Icon.play}
                                </Button>
                                <Button onClick={() => setRestart(restart + 1)}>{Icon.reset} Restart</Button>
                                <Button onClick={() => setSeed(randomSeed())}>New sample</Button>
                            </div>
                            <Stats>
                                <Stat label="iteration" value={frame?.iter ?? 0} />
                                <Stat label="KL(P‖Q)" value={frame && Number.isFinite(frame.kl) ? frame.kl.toFixed(3) : "—"} color={C.yellow} />
                            </Stats>
                            <p className="note">
                                Colors are the true groups. Neither method sees them. PCA keeps the directions of largest variance. t-SNE keeps each point's
                                nearest neighbors close.
                            </p>
                        </div>
                    }
                />
            </div>
        </Instrument>
    );
}

const SCENARIOS: Scenario[] = [
    {
        id: "hidden",
        title: "Clusters PCA can't see",
        prompt: <>Six clusters in 10 dimensions, each along its own axis. PCA's best flat view piles most of them on top of each other. t-SNE pulls each one into its own island.</>,
        render: () => <TsneInstrument />,
    },
    {
        id: "perplexity",
        title: "Perplexity",
        prompt: <>Perplexity is roughly how many neighbors each point pays attention to. At 3 the clusters shatter into fragments. Slide it up and watch the real structure come back.</>,
        render: () => <TsneInstrument kind="hierarchy" perplexity={3} />,
    },
    {
        id: "sizes",
        title: "Cluster sizes lie",
        prompt: <>In 10D one cluster is ten times wider than the other. t-SNE draws them about the same size, because it adapts to local density. Size in a t-SNE plot means nothing.</>,
        render: () => <TsneInstrument kind="sizes" />,
    },
    {
        id: "distances",
        title: "Distances lie",
        prompt: <>Two clusters are close and the third is far away (PCA shows it). t-SNE separates all three, but the gaps between islands don't reflect the true distances.</>,
        render: () => <TsneInstrument kind="distances" />,
    },
    {
        id: "hierarchy",
        title: "Groups of groups",
        prompt: <>Three groups, each made of three sub-clusters. At moderate perplexity t-SNE shows both levels: islands that sit near their siblings.</>,
        render: () => <TsneInstrument kind="hierarchy" perplexity={30} />,
    },
    {
        id: "swiss",
        title: "Swiss roll",
        prompt: <>A 2D sheet rolled up in 3D. PCA squashes the layers on top of each other. t-SNE keeps neighbors along the sheet together, so the color gradient stays in order.</>,
        render: () => <TsneInstrument kind="swiss" perplexity={30} />,
    },
];

export default function TsnePlayground() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            Convert high-dimensional distances into neighbor probabilities. Each point gets its own Gaussian width <M>{"\\sigma_i"}</M>,
                            tuned so its neighborhood has the chosen <strong>perplexity</strong> (an effective neighbor count):
                        </li>
                    </ul>
                    <MathBlock>{"p_{j\\mid i} = \\frac{\\exp(-\\lVert\\vx_i-\\vx_j\\rVert^2 / 2\\sigma_i^2)}{\\sum_{k\\neq i}\\exp(-\\lVert\\vx_i-\\vx_k\\rVert^2/2\\sigma_i^2)}, \\qquad q_{ij} \\propto \\big(1 + \\lVert\\mathbf y_i - \\mathbf y_j\\rVert^2\\big)^{-1}"}</MathBlock>
                    <ul>
                        <li>
                            Place points in 2D so the heavy-tailed neighbor probabilities <M>{"q"}</M> match <M>{"p"}</M>, by gradient descent on{" "}
                            <M>{"\\mathrm{KL}(P\\,\\|\\,Q)"}</M>. The heavy tail lets dissimilar points sit far apart, which makes clusters pop.
                        </li>
                        <li>
                            t-SNE preserves <em>local</em> neighborhoods. Cluster sizes and the distances between clusters are not meaningful, and different
                            runs and perplexities can look different. PCA is linear and keeps global variance.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
