import {useMemo, useState} from "react";
import {Button, Segmented, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {FigureLayout} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {Plot, type PlotPointer} from "../../components/Plot";
import {hitPoint, setRow} from "../../lib/data";
import {fmt} from "../../lib/hooks";
import type {View} from "../../lib/painter";
import {C, rgba} from "../../lib/theme";
import {constrained_quadratic, contour_segments, make_curve, poly_curve, poly_error_curve, poly_fit, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.1, xmax: 1.1, ymin: -1.35, ymax: 1.35};
const MAX_DEGREE = 20;
const truth = (x: number) => 0.65 * Math.sin(2.8 * x) + 0.25 * x;

// ===========================================================================
// Polynomial fitting instrument
// ===========================================================================

function PolyInstrument(props: {degree?: number; n?: number; penalty?: number; lambda?: number}) {
    const [n, setN] = useState(props.n ?? 14);
    const [seed, setSeed] = useState(7);
    const [degree, setDegree] = useState(props.degree ?? 10);
    const [penalty, setPenalty] = useState(props.penalty ?? -1);
    const [lambda, setLambda] = useState(props.lambda ?? 1e-3);
    const [showTest, setShowTest] = useState(true);
    const [edits, setEdits] = useState<{key: string; data: Float64Array} | null>(null);
    const [drag, setDrag] = useState<number | null>(null);

    const key = `${n}|${seed}`;
    const generated = useMemo(() => make_curve(n, 0.18, seed), [n, seed]);
    const train = edits && edits.key === key ? edits.data : generated;
    const test = useMemo(() => make_curve(300, 0.18, seed + 101), [seed]);
    const lam = penalty < 0 ? 0 : lambda;
    const pen = Math.max(0, penalty);

    const coefs = useMemo(() => poly_fit(train, degree, lam, pen), [train, degree, lam, pen]);
    const curve = useMemo(() => poly_curve(coefs, VIEW.xmin, VIEW.xmax, 500), [coefs]);
    const errors = useMemo(() => {
        const e = poly_error_curve(train, test, MAX_DEGREE, lam, pen);
        const tr = new Float64Array(MAX_DEGREE + 1);
        const te = new Float64Array(MAX_DEGREE + 1);
        for (let d = 0; d <= MAX_DEGREE; d++) {
            tr[d] = e[2 * d] + 1e-5;
            te[d] = e[2 * d + 1] + 1e-5;
        }
        return {tr, te};
    }, [train, test, lam, pen]);
    const norm = Math.hypot(...Array.from(coefs).slice(1));

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") {
            const i = hitPoint(train, 2, e.x, e.y, e.kx, e.ky, 12);
            if (i >= 0) setDrag(i);
        }
        if (e.type === "move" && drag !== null)
            setEdits({key, data: setRow(train, 2, drag, [Math.max(-1, Math.min(1, e.x)), Math.max(VIEW.ymin + 0.05, Math.min(VIEW.ymax - 0.05, e.y))])});
        if (e.type === "up") setDrag(null);
    };

    return (
        <Instrument hint="Drag the blue training points · hollow circles are fresh test points the model never sees">
            <FigureLayout
                sideWidth={310}
                main={
                    <>
                        <Plot
                            label="Polynomial fit to noisy data"
                            view={VIEW}
                            aspect={0.62}
                            cursor={drag !== null ? "grabbing" : "default"}
                            onPointer={onPointer}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25);
                                const t: number[] = [];
                                for (let i = 0; i <= 200; i++) {
                                    const x = -1 + (2 * i) / 200;
                                    t.push(x, truth(x));
                                }
                                p.polyline(t, {color: rgba(C.green, 0.55), width: 1.5, dash: [6, 6]});
                                if (showTest) for (let i = 0; i < test.length; i += 2) p.circle(test[i], test[i + 1], 2.6, {stroke: rgba(C.muted, 0.6), width: 1});
                                p.polyline(curve, {color: C.yellow, width: 2.8});
                                for (let i = 0; i < train.length; i += 2) p.point(train[i], train[i + 1], C.blue, {r: 5});
                            }}
                        />
                        <div className="duo">
                            <div>
                                <div className="panel-label">
                                    <span>Error vs. degree (log)</span>
                                </div>
                                <LineChart
                                    label="Training and test error for each polynomial degree"
                                    series={[
                                        {values: errors.tr, color: C.muted, dash: [4, 3]},
                                        {values: errors.te, color: C.green},
                                    ]}
                                    height={110}
                                    logY
                                    marker={degree}
                                    xLabel="degree"
                                    onPick={(x) => setDegree(Math.max(0, Math.min(MAX_DEGREE, Math.round(x))))}
                                />
                            </div>
                            <div>
                                <div className="panel-label">
                                    <span>Coefficients (symlog)</span>
                                </div>
                                <Plot
                                    label="Polynomial coefficients"
                                    view={{xmin: -0.7, xmax: MAX_DEGREE + 0.7, ymin: -4.2, ymax: 4.2}}
                                    height={110}
                                    draw={(p) => {
                                        p.clear();
                                        p.line(-0.7, 0, MAX_DEGREE + 0.7, 0, {color: C.axis, width: 1});
                                        for (let k = 0; k < coefs.length; k++) {
                                            const c = coefs[k];
                                            const h = Math.sign(c) * Math.log10(1 + Math.abs(c) * 100);
                                            if (c === 0) {
                                                p.circle(k, 0, 2.5, {stroke: C.muted, width: 1});
                                                continue;
                                            }
                                            p.polyline([k - 0.32, 0, k + 0.32, 0, k + 0.32, h, k - 0.32, h], {close: true, fill: c > 0 ? C.blue : C.orange, color: "none"});
                                        }
                                    }}
                                />
                            </div>
                        </div>
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label="polynomial degree" value={degree} min={0} max={MAX_DEGREE} step={1} onChange={setDegree} format={(v) => v.toFixed(0)} color={C.yellow} />
                        <Slider label="training points" value={n} min={6} max={200} step={1} onChange={setN} format={(v) => v.toFixed(0)} color={C.blue} />
                        <Segmented
                            label="regularization"
                            value={penalty}
                            onChange={setPenalty}
                            options={[
                                {value: -1, label: "None"},
                                {value: 0, label: "Ridge (L2)"},
                                {value: 1, label: "Lasso (L1)"},
                            ]}
                        />
                        {penalty >= 0 && (
                            <Slider label={<>strength <M>{"\\lambda"}</M></>} value={lambda} min={1e-6} max={1} log onChange={setLambda} color={C.purple} format={(v) => v.toExponential(1)} />
                        )}
                        <Toggle label="Show test points" checked={showTest} onChange={setShowTest} />
                        <Stats>
                            <Stat label="train error" value={fmt(errors.tr[degree] - 1e-5, 4)} />
                            <Stat label="test error" value={fmt(errors.te[degree] - 1e-5, 4)} color={C.green} />
                            <Stat label="‖coefficients‖" value={fmt(norm, 2)} color={C.yellow} />
                            <Stat label="zero coefficients" value={Array.from(coefs).slice(1).filter((c) => c === 0).length} />
                        </Stats>
                        <div className="control-row">
                            <Button onClick={() => (setSeed(randomSeed()), setEdits(null))}>New sample</Button>
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Why L1 makes zeros: constraint geometry
// ===========================================================================

const GVIEW: View = {xmin: -2.2, xmax: 2.2, ymin: -2.2, ymax: 2.2};
const GG = 140;
const A = {a11: 1, a12: 0.55, a22: 0.75};
const LEVELS = [0.05, 0.2, 0.45, 0.8, 1.25, 1.8, 2.5, 3.3, 4.2];

function GeometryInstrument(props: {norm?: number}) {
    const [norm, setNorm] = useState(props.norm ?? 1);
    const [c, setC] = useState({x: 1.55, y: 0.7});
    const [t, setT] = useState(0.9);
    const [drag, setDrag] = useState(false);

    const grid = useMemo(() => {
        const g = new Float64Array(GG * GG);
        for (let j = 0; j < GG; j++)
            for (let i = 0; i < GG; i++) {
                const x = GVIEW.xmin + ((i + 0.5) * (GVIEW.xmax - GVIEW.xmin)) / GG;
                const y = GVIEW.ymax - ((j + 0.5) * (GVIEW.ymax - GVIEW.ymin)) / GG;
                g[j * GG + i] = A.a11 * (x - c.x) ** 2 + 2 * A.a12 * (x - c.x) * (y - c.y) + A.a22 * (y - c.y) ** 2;
            }
        return g;
    }, [c]);
    const contours = useMemo(() => contour_segments(grid, GG, GG, GVIEW.xmin, GVIEW.xmax, GVIEW.ymin, GVIEW.ymax, new Float64Array(LEVELS)), [grid]);
    // The constrained optimum, and the loss contour that just touches the region there.
    const {w, touching} = useMemo(() => {
        const sol = constrained_quadratic(A.a11, A.a12, A.a22, c.x, c.y, t, norm);
        const level = A.a11 * (sol[0] - c.x) ** 2 + 2 * A.a12 * (sol[0] - c.x) * (sol[1] - c.y) + A.a22 * (sol[1] - c.y) ** 2;
        return {w: sol, touching: contour_segments(grid, GG, GG, GVIEW.xmin, GVIEW.xmax, GVIEW.ymin, GVIEW.ymax, new Float64Array([level]))};
    }, [grid, c, t, norm]);
    const sparse = Math.abs(w[0]) < 1e-9 || Math.abs(w[1]) < 1e-9;

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") setDrag(Math.hypot((e.x - c.x) * e.kx, (e.y - c.y) * e.ky) < 18);
        if (e.type === "move" && drag) setC({x: Math.max(-2, Math.min(2, e.x)), y: Math.max(-2, Math.min(2, e.y))});
        if (e.type === "up") setDrag(false);
    };

    return (
        <Instrument hint="Drag the yellow star: the best coefficients if there were no penalty">
            <FigureLayout
                main={
                    <Plot
                        label="Loss contours meeting an L1 or L2 constraint region"
                        view={GVIEW}
                        maxHeight={560}
                        cursor={drag ? "grabbing" : "default"}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            p.grid(0.5, {majorEvery: 2});
                            p.ticks(1, 1, {xLabel: "w₁", yLabel: "w₂"});
                            p.contours(contours, () => ({color: rgba(C.purple, 0.45), width: 1.2}));
                            if (norm === 1) p.polyline([t, 0, 0, t, -t, 0, 0, -t], {close: true, fill: C.blue, fillAlpha: 0.16, color: C.blue, width: 2});
                            else p.circle(0, 0, t * p.kx, {fill: "rgba(88,196,221,0.16)", stroke: C.blue, width: 2});
                            p.contours(touching, () => ({color: C.yellow, width: 2}));
                            p.line(c.x, c.y, w[0], w[1], {color: rgba(C.white, 0.4), width: 1, dash: [3, 4]});
                            p.text("★", c.x, c.y, {color: C.yellow, size: 22, align: "center", baseline: "middle"});
                            p.circle(w[0], w[1], 7, {fill: sparse ? C.green : C.white, stroke: C.canvas, width: 2.5});
                        }}
                    />
                }
                side={
                    <div className="controls">
                        <Segmented
                            label="penalty"
                            value={norm}
                            onChange={setNorm}
                            options={[
                                {value: 1, label: "L1 (lasso)"},
                                {value: 2, label: "L2 (ridge)"},
                            ]}
                        />
                        <Slider label="budget t (smaller = stronger penalty)" value={t} min={0.1} max={2.2} onChange={setT} color={C.blue} />
                        <Stats>
                            <Stat label="w₁" value={w[0].toFixed(3)} color={Math.abs(w[0]) < 1e-9 ? C.green : undefined} />
                            <Stat label="w₂" value={w[1].toFixed(3)} color={Math.abs(w[1]) < 1e-9 ? C.green : undefined} />
                        </Stats>
                        <p className="note" style={{color: sparse ? C.green : undefined}}>
                            {sparse ? "One coefficient is exactly zero: the contour touched a corner." : "Both coefficients are nonzero."}
                        </p>
                        <p className="note">
                            Regularizing means minimizing the loss (purple ellipses around the star) while keeping the coefficients inside the blue region.
                            The answer is where the growing ellipses first touch the region (yellow). The diamond's corners stick out along the axes, so
                            they are usually touched first, which puts a coefficient at exactly zero.
                        </p>
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
        id: "wiggle",
        title: "Wiggle to fit",
        prompt: <>Degree 10 through 14 noisy points: the curve threads almost every point and whips around between them. Drag a single point and watch the whole curve react.</>,
        render: () => <PolyInstrument />,
    },
    {
        id: "sweet-spot",
        title: "Find the sweet spot",
        prompt: <>Training error (dashed) falls with every extra degree, but test error (green) is U-shaped. Click the chart to try degrees and find the bottom of the U.</>,
        render: () => <PolyInstrument degree={3} />,
    },
    {
        id: "ridge",
        title: "Ridge tames it",
        prompt: <>The same degree-15 model, but large coefficients now cost something. Slide λ up: the curve calms down and test error drops, until it is too stiff to bend at all.</>,
        render: () => <PolyInstrument degree={15} penalty={0} lambda={2e-4} />,
    },
    {
        id: "lasso",
        title: "Lasso zeros out",
        prompt: <>An L1 penalty drives many coefficients to exactly zero (the empty circles). It doesn't just shrink the model, it selects which terms to keep.</>,
        render: () => <PolyInstrument degree={15} penalty={1} lambda={3e-3} />,
    },
    {
        id: "more-data",
        title: "More data",
        prompt: <>With 150 points even an unregularized degree-15 polynomial barely wiggles. Overfitting is about flexibility <em>relative to</em> the amount of data. Slide the points down.</>,
        render: () => <PolyInstrument degree={15} n={150} />,
    },
    {
        id: "geometry",
        title: "Why L1 gives zeros",
        prompt: <>Drag the star (the unpenalized optimum) around. With the L1 diamond the solution keeps snapping to a corner, an exact zero. Switch to L2: the circle has no corners, so no zeros.</>,
        render: () => <GeometryInstrument />,
    },
];

export default function Overfitting() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            A degree-<M>{"d"}</M> polynomial has <M>{"d + 1"}</M> knobs. With enough knobs it can pass through every training point,
                            noise included. Training error then says nothing about new data.
                        </li>
                        <li>Regularization adds a price for large coefficients:</li>
                    </ul>
                    <MathBlock>{"\\min_{\\mathbf c}\\; \\frac1n\\sum_i \\big(y_i - p_{\\mathbf c}(x_i)\\big)^2 + \\lambda \\begin{cases} \\sum_k c_k^2 & \\text{ridge (L2)} \\\\ \\sum_k |c_k| & \\text{lasso (L1)} \\end{cases}"}</MathBlock>
                    <ul>
                        <li>
                            Ridge shrinks every coefficient smoothly. Lasso pushes many to exactly zero, which is <em>feature selection</em>. The penalized
                            problem is equivalent to minimizing the loss inside a ball <M>{"\\lVert\\mathbf c\\rVert \\le t"}</M>, and the L1 ball has
                            corners.
                        </li>
                        <li>
                            Pick <M>{"d"}</M> and <M>{"\\lambda"}</M> by held-out error: the bottom of the U-shaped test curve. (Coefficients here are in
                            the numerically stable Legendre basis.)
                        </li>
                    </ul>
                </>
            }
        />
    );
}
