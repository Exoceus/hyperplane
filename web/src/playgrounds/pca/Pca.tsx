import {useMemo, useState} from "react";
import {Button, Legend, Player, Slider, Stat, Stats} from "../../components/Controls";
import {FigureLayout, Panel} from "../../components/Figure";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {ease, useTimeline} from "../../lib/hooks";
import type {Painter, View} from "../../lib/painter";
import {C, rgba} from "../../lib/theme";
import {make_gaussian, pca, power_iteration, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const N = 70;
const SAMPLES = 180;
const POWER_STEPS = 12;

interface Pca2 {
    mean: [number, number];
    cxx: number;
    cxy: number;
    cyy: number;
    l1: number;
    l2: number;
    v1: [number, number];
    v2: [number, number];
}

/** Unpack the WASM `pca(points, 2)` result. */
function pca2(points: Float64Array): Pca2 {
    const r = pca(points, 2);
    return {mean: [r[0], r[1]], cxx: r[2], cxy: r[3], cyy: r[5], l1: r[6], l2: r[7], v1: [r[8], r[9]], v2: [r[10], r[11]]};
}

/** Variance of the data projected onto the unit vector at angle θ: uᵀΣu. */
function varianceAlong(s: Pca2, theta: number) {
    const c = Math.cos(theta);
    const sn = Math.sin(theta);
    return s.cxx * c * c + 2 * s.cxy * sn * c + s.cyy * sn * sn;
}

function drawEllipse(p: Painter, s: Pca2, k: number, color: string) {
    const pts: number[] = [];
    const a = k * Math.sqrt(Math.max(0, s.l1));
    const b = k * Math.sqrt(Math.max(0, s.l2));
    for (let i = 0; i <= 90; i++) {
        const t = (i / 90) * Math.PI * 2;
        const u = a * Math.cos(t);
        const v = b * Math.sin(t);
        pts.push(s.mean[0] + u * s.v1[0] + v * s.v2[0], s.mean[1] + u * s.v1[1] + v * s.v2[1]);
    }
    p.polyline(pts, {color, width: 1.5, dash: [5, 5]});
}

// ===========================================================================
// Figure 1 — the direction of maximum variance
// ===========================================================================

function DirectionFigure(props: {stretch?: number}) {
    const [seed, setSeed] = useState(8);
    const [stretch, setStretch] = useState(props.stretch ?? 0.3);
    const data = useMemo(() => make_gaussian(N, 0.05, -0.05, 0.5, 0.5 * stretch, 0.6, seed), [seed, stretch]);
    const stats = useMemo(() => pca2(data), [data]);
    // Manual angle (dragging) vs. the angle found by power iteration.
    const [manualAngle, setManualAngle] = useState(2.2);
    const [manual, setManual] = useState(false);
    const [runId, setRunId] = useState(0);
    const [dragging, setDragging] = useState(false);

    // Power iteration from the current guess: v ← Σv / ‖Σv‖, replayed whenever the data changes.
    const path = useMemo(() => {
        const cov = new Float64Array([stats.cxx, stats.cxy, stats.cxy, stats.cyy]);
        const raw = power_iteration(cov, 2, new Float64Array([Math.cos(manualAngle), Math.sin(manualAngle)]), POWER_STEPS);
        const angles: number[] = [];
        for (let k = 0; k <= POWER_STEPS; k++) {
            let a = Math.atan2(raw[2 * k + 1], raw[2 * k]);
            const prev = k === 0 ? manualAngle : angles[k - 1];
            while (a - prev > Math.PI / 2) a -= Math.PI;
            while (a - prev < -Math.PI / 2) a += Math.PI;
            angles.push(a);
        }
        return {angles, runId};
        // manualAngle is read only as the starting guess when a run begins.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stats, runId]);
    const tl = useTimeline(POWER_STEPS, 1, {autoplay: true, runKey: path, duration: 4, delay: 0.6});
    const k = Math.min(POWER_STEPS - 1, Math.floor(tl.t));
    const theta = manual ? manualAngle : path.angles[k] + (path.angles[k + 1] - path.angles[k]) * ease(tl.t - k);
    const iteration = Math.min(POWER_STEPS, Math.round(tl.t));

    const along = varianceAlong(stats, theta);
    const total = stats.cxx + stats.cyy;
    const residual = Math.max(0, total - along);
    const pcAngle = Math.atan2(stats.v1[1], stats.v1[0]);

    const curve = useMemo(() => {
        const out = new Float64Array(SAMPLES + 1);
        for (let i = 0; i <= SAMPLES; i++) out[i] = varianceAlong(stats, (i / SAMPLES) * Math.PI);
        return out;
    }, [stats]);
    const residualCurve = useMemo(() => curve.map((v) => total - v), [curve, total]);
    const deg = ((((theta * 180) / Math.PI) % 180) + 180) % 180;
    const setAngle = (a: number) => {
        setManual(true);
        setManualAngle(a);
    };
    const runPower = () => {
        setManualAngle(theta);
        setManual(false);
        setRunId(runId + 1);
    };

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") setDragging(true);
        if (e.type === "up") setDragging(false);
        if ((e.type === "down" || (e.type === "move" && dragging)) && Math.hypot(e.x - stats.mean[0], e.y - stats.mean[1]) > 0.05) {
            let a = Math.atan2(e.y - stats.mean[1], e.x - stats.mean[0]);
            while (a - theta > Math.PI / 2) a -= Math.PI;
            while (a - theta < -Math.PI / 2) a += Math.PI;
            setAngle(a);
        }
    };

    const ux = Math.cos(theta);
    const uy = Math.sin(theta);
    const [mx, my] = stats.mean;

    return (
        <Instrument hint="Drag anywhere to rotate the line">
            <FigureLayout
                main={
                    <>
                    <Plot
                        label="Projecting 2D data onto a rotatable line"
                        view={VIEW}
                        maxHeight={560}
                        cursor={dragging ? "grabbing" : "grab"}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            p.grid(0.25);
                            p.line(mx - ux * 3, my - uy * 3, mx + ux * 3, my + uy * 3, {color: C.yellow, width: 2.5});
                            for (let i = 0; i < data.length; i += 2) {
                                const dx = data[i] - mx;
                                const dy = data[i + 1] - my;
                                const s = dx * ux + dy * uy;
                                const px = mx + s * ux;
                                const py = my + s * uy;
                                p.line(data[i], data[i + 1], px, py, {color: rgba(C.pink, 0.55), width: 1.2});
                                p.circle(px, py, 3, {fill: C.yellow});
                            }
                            for (let i = 0; i < data.length; i += 2) p.point(data[i], data[i + 1], C.teal, {r: 4.2});
                            p.circle(mx, my, 5, {fill: C.white, stroke: C.canvas, width: 2});
                            p.arrow(mx, my, mx + ux * 0.45, my + uy * 0.45, {color: C.yellow, width: 3, head: 12});
                            p.text("u", mx + ux * 0.45, my + uy * 0.45, {color: C.yellow, size: 16, font: "serif", weight: 600, dx: 8, dy: -6});
                        }}
                    />
                    <Player
                        timeline={tl}
                        speeds={[0.5, 1, 3]}
                        status={
                            manual ? (
                                <>Your guess: θ = {deg.toFixed(0)}°. Press <b>Run power iteration</b> to let the algorithm improve it.</>
                            ) : (
                                <>
                                    Power iteration step {iteration}: <M>{"\\vu \\leftarrow \\Sigma\\vu / \\lVert\\Sigma\\vu\\rVert"}</M>. Each multiply by the covariance
                                    tilts the line toward the direction of most spread.
                                </>
                            )
                        }
                    />
                    </>
                }
                side={
                    <div className="controls">
                        <div className="side-block">
                            <span className="control-label">Total variance, split two ways</span>
                            <div style={{display: "flex", height: 16, borderRadius: 5, overflow: "hidden"}}>
                                <div style={{flex: along, background: C.yellow}} />
                                <div style={{flex: residual, background: C.pink}} />
                            </div>
                            <Legend
                                items={[
                                    {color: C.yellow, label: `along u: ${along.toFixed(3)}`, shape: "square"},
                                    {color: C.pink, label: `off the line: ${residual.toFixed(3)}`, shape: "square"},
                                ]}
                            />
                        </div>
                        <div className="panel-label">
                            <span>Variance vs. angle</span>
                            <span className="panel-label-right">θ = {deg.toFixed(0)}°</span>
                        </div>
                        <LineChart
                            label="Variance of the projection as a function of the angle"
                            series={[
                                {values: curve, color: C.yellow, dx: 180 / SAMPLES},
                                {values: residualCurve, color: C.pink, dx: 180 / SAMPLES, dash: [4, 3]},
                            ]}
                            height={120}
                            xMax={180}
                            yMin={0}
                            marker={deg}
                            xLabel="θ (degrees)"
                            onPick={(x) => setAngle((Math.max(0, Math.min(180, x)) * Math.PI) / 180)}
                        />
                        <Stats>
                            <Stat label="θ" value={`${deg.toFixed(0)}°`} color={C.yellow} />
                            <Stat label="best θ" value={`${(((((pcAngle * 180) / Math.PI) % 180) + 180) % 180).toFixed(0)}°`} color={C.green} />
                        </Stats>
                        <Slider label="How elongated the cloud is" value={stretch} min={0.08} max={1} onChange={setStretch} color={C.teal} format={(v) => `${v.toFixed(2)}`} />
                        <div className="control-row">
                            <Button variant="primary" onClick={runPower}>
                                Run power iteration
                            </Button>
                            <Button onClick={() => setSeed(randomSeed())}>New sample</Button>
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Figure 2 — rotate into the eigenbasis, then drop an axis
// ===========================================================================

function texMatrix(a: number, b: number, d: number) {
    const f = (v: number) => (Math.abs(v) < 5e-4 ? "0.000" : v.toFixed(3));
    return `\\Sigma = \\begin{pmatrix} \\yel{${f(a)}} & \\pnk{${f(b)}} \\\\ \\pnk{${f(b)}} & \\tea{${f(d)}} \\end{pmatrix}`;
}

function RotateFigure() {
    const [seed, setSeed] = useState(15);
    const data = useMemo(() => make_gaussian(N, 0.2, 0.15, 0.48, 0.15, 2.4, seed), [seed]);
    const base = useMemo(() => pca2(data), [data]);
    const tl = useTimeline(2, 0.5, {autoplay: true, runKey: data});
    const t1 = ease(Math.min(1, tl.t));
    const t2 = ease(Math.max(0, tl.t - 1));

    const angle = Math.atan2(base.v1[1], base.v1[0]);
    const moved = useMemo(() => {
        const out = new Float64Array(data.length);
        const r = -angle * t1;
        const c = Math.cos(r);
        const s = Math.sin(r);
        for (let i = 0; i < data.length; i += 2) {
            const x = data[i] - base.mean[0] * t1;
            const y = data[i + 1] - base.mean[1] * t1;
            out[i] = c * x - s * y;
            out[i + 1] = (s * x + c * y) * (1 - t2);
        }
        return out;
    }, [data, base, angle, t1, t2]);
    const cur = useMemo(() => pca2(moved), [moved]);
    const kept = base.l1 / (base.l1 + base.l2);

    const stage =
        tl.t < 0.02
            ? "The original data. Its covariance matrix has large off-diagonal entries: x and y move together."
            : tl.t < 0.98
              ? "Centering and rotating the cloud…"
              : tl.t < 1.02
                ? "In the principal-axis frame the off-diagonal entries vanish. The new coordinates are uncorrelated."
                : tl.t < 1.98
                  ? "Dropping the second coordinate…"
                  : `Projected onto PC1: one number per point, keeping ${(kept * 100).toFixed(0)}% of the variance.`;

    return (
        <Instrument hint="Center, rotate, then project; it replays whenever you resample">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Data rotated into its principal axes and projected"
                            view={VIEW}
                            maxHeight={520}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25);
                                drawEllipse(p, cur, 2, rgba(C.white, 0.6));
                                p.arrow(cur.mean[0], cur.mean[1], cur.mean[0] + cur.v1[0] * 2 * Math.sqrt(cur.l1), cur.mean[1] + cur.v1[1] * 2 * Math.sqrt(cur.l1), {
                                    color: C.yellow,
                                    width: 3,
                                });
                                if (cur.l2 > 1e-6)
                                    p.arrow(cur.mean[0], cur.mean[1], cur.mean[0] + cur.v2[0] * 2 * Math.sqrt(cur.l2), cur.mean[1] + cur.v2[1] * 2 * Math.sqrt(cur.l2), {
                                        color: C.teal,
                                        width: 3,
                                    });
                                for (let i = 0; i < moved.length; i += 2) p.point(moved[i], moved[i + 1], C.blue, {r: 4});
                            }}
                        />
                        <Player timeline={tl} speeds={[0.25, 0.5, 1]} status={stage} />
                    </>
                }
                side={
                    <div className="controls">
                        <Panel label="Covariance of what you see">
                            <div style={{fontSize: 17, padding: "6px 0"}}>
                                <M>{texMatrix(cur.cxx, cur.cxy, cur.cyy)}</M>
                            </div>
                        </Panel>
                        <p className="note">
                            The <span className="c-yellow">diagonal</span> entries are the variances along each axis, and the{" "}
                            <span className="c-pink">off-diagonal</span> entry is the covariance between them. Rotating to the principal axes drives the
                            off-diagonal entry to zero.
                        </p>
                        <Stats>
                            <Stat label={<M>{"\\lambda_1"}</M>} value={base.l1.toFixed(3)} color={C.yellow} />
                            <Stat label={<M>{"\\lambda_2"}</M>} value={base.l2.toFixed(3)} color={C.teal} />
                            <Stat label="PC1 keeps" value={`${(kept * 100).toFixed(1)}%`} color={C.green} />
                        </Stats>
                        <div className="control-row">
                            <Button onClick={() => (setSeed(randomSeed()), tl.reset())}>New sample</Button>
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

const SCENARIOS: Scenario[] = [
    {
        id: "best-line",
        title: "Find the best line",
        prompt: <>Drag to rotate the yellow line. Projecting onto it keeps the yellow part of the variance and loses the pink part. Find the angle that keeps the most, then check with <b>Find the best line</b>.</>,
        render: () => <DirectionFigure />,
    },
    {
        id: "round",
        title: "A round cloud",
        prompt: <>When the cloud is round, every direction keeps the same variance, the curve goes flat, and there is no meaningful first component. Stretch it back out.</>,
        render: () => <DirectionFigure stretch={1} />,
    },
    {
        id: "needle",
        title: "Almost a line",
        prompt: <>A very elongated cloud: one direction holds nearly all the variance, so one number per point loses almost nothing.</>,
        render: () => <DirectionFigure stretch={0.08} />,
    },
    {
        id: "rotate",
        title: "Rotate, then project",
        prompt: <>PCA centers the data and rotates it onto its principal axes. Watch the off-diagonal covariance go to zero, then drop the second axis.</>,
        render: () => <RotateFigure />,
    },
];

export default function Pca() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            Project centered data onto a unit direction <M>{"\\vu"}</M>. The variance it keeps is a quadratic form in the covariance matrix:
                        </li>
                    </ul>
                    <MathBlock>{"\\operatorname{Var}(\\vu\\cdot\\vx) = \\vu^\\top\\Sigma\\,\\vu, \\qquad \\max_{\\lVert\\vu\\rVert = 1} \\vu^\\top\\Sigma\\vu \\;\\Rightarrow\\; \\Sigma\\vu = \\lambda\\vu"}</MathBlock>
                    <ul>
                        <li>
                            The best direction is the top <strong>eigenvector</strong> of <M>{"\\Sigma"}</M>, and its eigenvalue is the variance captured.
                            The next components are orthogonal and capture what's left.
                        </li>
                        <li>
                            By Pythagoras, kept variance plus squared distance to the line equals the total. Maximizing the variance you keep is the same as
                            minimizing the reconstruction error.
                        </li>
                        <li>
                            In the eigenvector basis the covariance is diagonal. Dropping axes with small <M>{"\\lambda"}</M> is dimensionality reduction.
                            Standardize features first, and remember PCA is linear.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
