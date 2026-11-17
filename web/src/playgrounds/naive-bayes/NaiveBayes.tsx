import {useMemo, useState} from "react";
import {Button, Player, Segmented, Stat, Stats, Toggle} from "../../components/Controls";
import {FigureLayout} from "../../components/Figure";
import {M, MathBlock} from "../../components/Math";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {Plot} from "../../components/Plot";
import {Plot3D} from "../../components/Plot3D";
import {classRegionColor} from "../../lib/colors";
import {useTimeline} from "../../lib/hooks";
import {heatImage, type Painter, type View} from "../../lib/painter";
import {drawSurface} from "../../lib/scene3d";
import {C, CLASS_RGB, classColor, rgba} from "../../lib/theme";
import {contour_segments, gaussian_bayes_density, gaussian_bayes_fit, gaussian_bayes_grid, make_blobs, make_gaussian, make_two_blobs, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const G = 100;
const CS = 6; // [prior, mx, my, vxx, vxy, vyy]

type Kind = "blobs" | "unequal" | "tilted" | "three";
const KINDS: {value: Kind; label: string}[] = [
    {value: "blobs", label: "Two blobs"},
    {value: "unequal", label: "Tight vs. wide"},
    {value: "tilted", label: "Correlated"},
    {value: "three", label: "3 classes"},
];

/** Interleave per-class point lists so a stream alternates between classes. */
function interleave(groups: Float64Array[]): Float64Array {
    const rows: number[][] = [];
    const max = Math.max(...groups.map((g) => g.length / 2));
    for (let i = 0; i < max; i++) groups.forEach((g, c) => i < g.length / 2 && rows.push([g[2 * i], g[2 * i + 1], c]));
    return new Float64Array(rows.flat());
}

function generate(kind: Kind, seed: number): Float64Array {
    switch (kind) {
        case "unequal":
            return interleave([make_gaussian(110, 0.1, 0.05, 0.13, 0.13, 0, seed), make_gaussian(110, -0.05, -0.05, 0.5, 0.45, 0, seed + 1)]);
        case "tilted":
            return interleave([make_gaussian(110, -0.18, 0.18, 0.5, 0.07, Math.PI / 4, seed), make_gaussian(110, 0.18, -0.18, 0.5, 0.07, Math.PI / 4, seed + 1)]);
        case "three":
            return make_blobs(240, 3, 0.2, seed);
        default:
            return make_two_blobs(200, 1.1, 0.24, seed);
    }
}

/** log P(c) + log N(x, y | μ_c, Σ_c), for scoring points. */
function logJoint(prm: ArrayLike<number>, x: number, y: number): number {
    const [prior, mx, my, a, b, d] = Array.from(prm);
    const det = Math.max(a * d - b * b, 1e-300);
    const dx = x - mx;
    const dy = y - my;
    return Math.log(Math.max(prior, 1e-300)) - (0.5 * (d * dx * dx - 2 * b * dx * dy + a * dy * dy)) / det - 0.5 * Math.log(det);
}

function ellipse(p: Painter, prm: ArrayLike<number>, k: number, color: string, width: number, dash?: number[]) {
    const [, mx, my, a, b, d] = Array.from(prm);
    const tr = a + d;
    const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - (a * d - b * b)));
    const l1 = tr / 2 + disc;
    const l2 = Math.max(0, tr / 2 - disc);
    const th = 0.5 * Math.atan2(2 * b, a - d);
    const pts: number[] = [];
    for (let i = 0; i <= 72; i++) {
        const t = (i / 72) * Math.PI * 2;
        const u = k * Math.sqrt(l1) * Math.cos(t);
        const v = k * Math.sqrt(l2) * Math.sin(t);
        pts.push(mx + u * Math.cos(th) - v * Math.sin(th), my + u * Math.sin(th) + v * Math.cos(th));
    }
    p.polyline(pts, {close: true, color, width, dash});
}

function NbInstrument(props: {kind?: Kind; full?: boolean}) {
    const [kind, setKind] = useState<Kind>(props.kind ?? "blobs");
    const [full, setFull] = useState(props.full ?? false);
    const [seed, setSeed] = useState(5);
    const data = useMemo(() => generate(kind, seed), [kind, seed]);
    const n = data.length / 3;
    const classes = kind === "three" ? 3 : 2;
    const runKey = useMemo(() => ({data, full}), [data, full]);
    const tl = useTimeline(n, 1, {autoplay: true, runKey, duration: 7});
    const used = Math.max(2, Math.min(n, Math.floor(tl.t)));

    const fit = useMemo(() => {
        const params = gaussian_bayes_fit(data, used, classes, full, 1e-3);
        const grid = gaussian_bayes_grid(params, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G);
        const regions = heatImage(grid, G, G, (c, i) => classRegionColor(c, Math.max(0, (grid[2 * i + 1] - 1 / classes) / (1 - 1 / classes)) * 0.85 + 0.15, 0.34), 2);
        let contour: Float64Array | null = null;
        if (classes === 2) {
            const p1 = new Float64Array(G * G);
            for (let i = 0; i < G * G; i++) p1[i] = grid[2 * i] === 1 ? grid[2 * i + 1] : 1 - grid[2 * i + 1];
            contour = contour_segments(p1, G, G, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, new Float64Array([0.5]));
        }
        let correct = 0;
        for (let i = 0; i < n; i++) {
            let best = 0;
            let bestV = -Infinity;
            for (let c = 0; c < classes; c++) {
                const v = logJoint(params.subarray(c * CS, (c + 1) * CS), data[3 * i], data[3 * i + 1]);
                if (v > bestV) {
                    bestV = v;
                    best = c;
                }
            }
            if (best === data[3 * i + 2]) correct++;
        }
        return {params, regions, contour, acc: correct / n};
    }, [data, used, classes, full, n]);

    return (
        <Instrument hint="Training points stream in one at a time · each class's Gaussian (ellipses) and the decision regions update after every point">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Gaussian naive Bayes decision regions"
                            view={VIEW}
                            maxHeight={580}
                            draw={(p) => {
                                p.clear();
                                p.image(fit.regions, VIEW);
                                p.grid(0.25, {axes: false});
                                if (fit.contour) p.contours(fit.contour, () => ({color: C.white, width: 2.5}));
                                for (let i = 0; i < n; i++) {
                                    const x = data[3 * i];
                                    const y = data[3 * i + 1];
                                    if (i < used) p.point(x, y, classColor(data[3 * i + 2]), {r: 4, ring: i === used - 1 && !tl.atEnd ? C.yellow : undefined});
                                    else p.circle(x, y, 2.5, {fill: rgba(C.muted, 0.35)});
                                }
                                for (let c = 0; c < classes; c++) {
                                    const prm = fit.params.subarray(c * CS, (c + 1) * CS);
                                    ellipse(p, prm, 1, classColor(c), 2.2);
                                    ellipse(p, prm, 2, classColor(c), 1.4, [5, 4]);
                                    p.circle(prm[1], prm[2], 5, {fill: C.canvas, stroke: classColor(c), width: 2.5});
                                }
                            }}
                        />
                        <Player timeline={tl} speeds={[0.5, 1, 3]} status={tl.atEnd ? `Trained on all ${n} points.` : `Learned from ${used} of ${n} points so far: each class is just a mean and a spread.`} />
                    </>
                }
                side={
                    <div className="controls">
                        <Segmented value={kind} onChange={setKind} options={KINDS} />
                        <Toggle label="Full covariance (QDA, not naive)" checked={full} onChange={setFull} />
                        <Stats>
                            <Stat label="points used" value={used} />
                            <Stat label="accuracy (all points)" value={`${Math.round(fit.acc * 100)}%`} color={C.green} />
                            {Array.from({length: classes}, (_, c) => (
                                <Stat key={c} label={`prior P(class ${c})`} value={fit.params[c * CS].toFixed(2)} color={classColor(c)} />
                            ))}
                        </Stats>
                        <p className="note">
                            {full
                                ? "Each class gets a full covariance, so its ellipse can tilt. That makes this QDA."
                                : "Naive Bayes assumes the features are independent within each class, so every ellipse stays aligned with the axes."}
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
// The class densities in 3D
// ===========================================================================

const N3 = 40;

function Densities3D() {
    const [kind, setKind] = useState<Kind>("blobs");
    const data = useMemo(() => generate(kind, 5), [kind]);
    const classes = kind === "three" ? 3 : 2;
    const surfaces = useMemo(() => {
        const params = gaussian_bayes_fit(data, data.length / 3, classes, false, 1e-3);
        const h = 2.6 / (N3 - 1);
        const dens = Array.from({length: classes}, (_, c) => gaussian_bayes_density(params, c, -1.3 - h / 2, 1.3 + h / 2, -1.3 - h / 2, 1.3 + h / 2, N3, N3));
        let max = 0;
        dens.forEach((d) => d.forEach((v) => (max = Math.max(max, v))));
        return dens.map((d) => d.map((v) => Math.sqrt(v / max) * 0.95 - 0.45));
    }, [data, classes]);
    const tl = useTimeline(1, 1, {autoplay: true, runKey: surfaces, duration: 2.5});
    const rise = tl.t;

    return (
        <Instrument hint="Each class's Gaussian (weighted by its prior) is a hill · the prediction is whichever hill is taller · drag to orbit">
            <FigureLayout
                main={
                    <>
                        <Plot3D
                            label="Class-conditional densities as 3D hills"
                            initial={{pitch: 0.5, yaw: 0.6}}
                            draw={(sc) => {
                                // Floor colored by the winning class.
                                const floor = new Float64Array(N3 * N3).fill(-0.45);
                                const win = (i: number) => {
                                    let best = 0;
                                    for (let c = 1; c < classes; c++) if (surfaces[c][i] > surfaces[best][i]) best = c;
                                    return best;
                                };
                                drawSurface(sc, floor, N3, () => [20, 24, 32], {alpha: 1});
                                for (let c = 0; c < classes; c++) {
                                    const hts = surfaces[c].map((v, i) => -0.45 + (v + 0.45) * rise + (win(i) === c ? 0.002 : 0));
                                    drawSurface(sc, hts, N3, () => CLASS_RGB[c], {alpha: 0.82, wire: 4});
                                }
                            }}
                        />
                        <Player timeline={tl} showSpeed={false} status="The decision boundary sits exactly where two hills cross." />
                    </>
                }
                side={
                    <div className="controls">
                        <Segmented value={kind} onChange={setKind} options={KINDS} />
                        <p className="note">
                            Height is <M>{"P(c)\\,p(\\vx \\mid c)"}</M> (square-rooted so both hills are visible). Bayes' rule picks the class whose hill is tallest at each point, so the boundary is
                            where two hills cross. A tight class makes a tall, narrow peak, and a wide class makes a low plateau, which is why their
                            boundary curves into a circle.
                        </p>
                    </div>
                }
            />
        </Instrument>
    );
}

const SCENARIOS: Scenario[] = [
    {
        id: "stream",
        title: "Learn from a stream",
        prompt: <>Training points arrive one at a time. Naive Bayes "trains" by computing each class's mean and spread, so watch the ellipses settle and the boundary move with every new point.</>,
        render: () => <NbInstrument />,
    },
    {
        id: "curved",
        title: "Curved boundaries",
        prompt: <>One class is tight and the other is wide. Different spreads give a curved (quadratic) boundary even though every model is "just a Gaussian".</>,
        render: () => <NbInstrument kind="unequal" />,
    },
    {
        id: "naive",
        title: "The naive assumption",
        prompt: <>Both classes are stretched along the diagonal, so x and y are correlated. Naive Bayes can only draw axis-aligned ellipses and misses it. Toggle full covariance (QDA) to fix it.</>,
        render: () => <NbInstrument kind="tilted" />,
    },
    {
        id: "three",
        title: "Three classes",
        prompt: <>With three classes, each region belongs to whichever class makes the point most probable. Watch the boundaries shift as the priors and spreads are learned.</>,
        render: () => <NbInstrument kind="three" />,
    },
    {
        id: "hills",
        title: "Densities in 3D",
        prompt: <>Each class is a hill of probability. Wherever one hill is taller, that class wins. Orbit to see where the two hills cross: that is the decision boundary. Then try Tight vs. wide.</>,
        render: () => <Densities3D />,
    },
];

export default function NaiveBayes() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>Pick the class that makes the point most probable, by Bayes' rule:</li>
                    </ul>
                    <MathBlock>{"\\hat y = \\arg\\max_c \\, P(c)\\, p(\\vx \\mid c), \\qquad p(\\vx\\mid c) = \\prod_j \\mathcal N(x_j \\mid \\mu_{cj}, \\sigma_{cj}^2)"}</MathBlock>
                    <ul>
                        <li>
                            <strong>Naive</strong> means the features are assumed independent given the class. Each class is then just a mean and a
                            variance per feature, so "training" is a single pass of averaging. It's fast, needs little data, and still works surprisingly
                            well.
                        </li>
                        <li>
                            Unequal variances make the boundary quadratic. When features are correlated, the independence assumption is wrong. Allowing a
                            full covariance per class gives quadratic discriminant analysis (QDA).
                        </li>
                        <li>
                            It's a <em>generative</em> model: it describes how each class produces data. Logistic regression is the discriminative cousin
                            that models <M>{"P(c\\mid\\vx)"}</M> directly.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
