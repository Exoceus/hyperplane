import {useMemo, useState} from "react";
import {Button, Legend, Meter, Player, Slider, Stat, Stats, Toggle} from "../../components/Controls";
import {FigureLayout, Panel} from "../../components/Figure";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {hitPoint, setRow} from "../../lib/data";
import {fmt, lerp, useTimeline} from "../../lib/hooks";
import {heatImage, type View} from "../../lib/painter";
import {C, rgba, sequential} from "../../lib/theme";
import {useTween} from "../../lib/tween";
import {contour_segments, linreg_fit, linreg_gd, linreg_grad, linreg_loss_grid, linreg_mse, make_regression, randomSeed} from "../../lib/wasm";

// ===========================================================================
// Figure 1 — least squares, literally
// ===========================================================================

const SQ_VIEW: View = {xmin: -1.35, xmax: 1.35, ymin: -1.05, ymax: 1.05};
const XL = -1.15;
const XR = 1.15;

function SquaresFigure({outlier = false}: {outlier?: boolean}) {
    const [data, setData] = useState<Float64Array>(() => {
        const base = make_regression(9, 0.55, 0.05, 0.28, 0, 4);
        if (!outlier) return base;
        const out = base.slice();
        out[2 * 7] = 0.75;
        out[2 * 7 + 1] = -0.9;
        return out;
    });
    const [line, animateLine, setLine] = useTween([-0.5, 0.6], 0.7); // [yL, yR] at x = XL, XR
    const [drag, setDrag] = useState<{kind: "point"; i: number} | {kind: "handle"; side: 0 | 1} | null>(null);
    const [showSquares, setShowSquares] = useState(true);

    const m = (line[1] - line[0]) / (XR - XL);
    const b = line[0] - m * XL;
    const n = data.length / 2;
    const sse = linreg_mse(data, m, b) * n;
    const best = linreg_fit(data);
    const bestSse = linreg_mse(data, best[0], best[1]) * n;

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") {
            const yAt = (x: number) => m * x + b;
            const nearHandle = ([0, 1] as const).find((s) => {
                const x = s === 0 ? XL : XR;
                return Math.hypot((e.x - x) * e.kx, (e.y - yAt(x)) * e.ky) < 16;
            });
            if (nearHandle !== undefined) return setDrag({kind: "handle", side: nearHandle});
            const i = hitPoint(data, 2, e.x, e.y, e.kx, e.ky, 14);
            if (i >= 0) return setDrag({kind: "point", i});
        }
        if (e.type === "up") setDrag(null);
        if (e.type === "move" && drag) {
            const y = Math.max(SQ_VIEW.ymin + 0.02, Math.min(SQ_VIEW.ymax - 0.02, e.y));
            if (drag.kind === "handle") {
                const next = [...line];
                next[drag.side] = y;
                setLine(next);
            } else {
                const x = Math.max(-1.1, Math.min(1.1, e.x));
                setData(setRow(data, 2, drag.i, [x, y]));
            }
        }
    };

    return (
        <Instrument hint="Drag the handles on the line, or drag the points">
            <FigureLayout
                main={
                    <Plot
                        label="A line through data with squared residuals drawn as squares"
                        view={SQ_VIEW}
                        cursor={drag ? "grabbing" : "grab"}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            p.grid(0.25);
                            for (let i = 0; i < data.length; i += 2) {
                                const x = data[i];
                                const y = data[i + 1];
                                const yh = m * x + b;
                                const r = y - yh;
                                if (showSquares) {
                                    const side = Math.abs(r);
                                    const dir = x + side > SQ_VIEW.xmax ? -1 : 1;
                                    p.polyline([x, yh, x, y, x + dir * side, y, x + dir * side, yh], {
                                        close: true,
                                        fill: C.yellow,
                                        fillAlpha: 0.13,
                                        color: rgba(C.yellow, 0.55),
                                        width: 1.2,
                                    });
                                }
                                p.line(x, y, x, yh, {color: C.red, width: 2});
                            }
                            p.infiniteLine(m, -1, b, {color: C.white, width: 2.5});
                            for (const s of [0, 1]) {
                                const x = s === 0 ? XL : XR;
                                p.circle(x, m * x + b, 8, {fill: C.canvas, stroke: C.white, width: 2.5});
                            }
                            for (let i = 0; i < data.length; i += 2) p.point(data[i], data[i + 1], C.blue, {r: 5.5});
                        }}
                    />
                }
                side={
                    <div className="controls">
                        <Meter label="Total area of the squares" right={sse.toFixed(3)} value={sse} max={Math.max(sse, bestSse * 4)} color={C.yellow} />
                        <Meter label="Smallest possible" right={bestSse.toFixed(3)} value={bestSse} max={Math.max(sse, bestSse * 4)} color={C.green} />
                        <Stats>
                            <Stat label="slope m" value={m.toFixed(2)} />
                            <Stat label="intercept b" value={b.toFixed(2)} />
                        </Stats>
                        <Toggle label="Draw the squares" checked={showSquares} onChange={setShowSquares} />
                        <div className="control-row">
                            <Button variant="primary" onClick={() => animateLine([best[0] * XL + best[1], best[0] * XR + best[1]])}>
                                Snap to least squares
                            </Button>
                            <Button onClick={() => setData(make_regression(9, (Math.random() - 0.5) * 1.6, (Math.random() - 0.5) * 0.4, 0.28, 0, randomSeed()))}>
                                New points
                            </Button>
                        </div>
                        <Legend
                            items={[
                                {color: C.red, label: "residual", shape: "line"},
                                {color: C.yellow, label: "residual²", shape: "square"},
                            ]}
                        />
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// Figure 2 — the loss landscape and gradient descent
// ===========================================================================

const DATA_VIEW: View = {xmin: -1.4, xmax: 2.8, ymin: -1.6, ymax: 2.6};
const PARAM_VIEW: View = {xmin: -1.6, xmax: 2.4, ymin: -2.1, ymax: 1.9};
const GRID = 140;
const STEPS = 200;

function LandscapeFigure(props: {lr?: number; center?: boolean}) {
    const [seed, setSeed] = useState(12);
    const [center, setCenter] = useState(props.center ?? false);
    const [lr, setLr] = useState(props.lr ?? 0.25);
    const [start, setStart] = useState({m: -1.1, b: 1.4});
    const [dragging, setDragging] = useState(false);

    const raw = useMemo(() => make_regression(40, 0.8, -0.6, 0.22, 1.4, seed), [seed]);
    const data = useMemo(() => {
        if (!center) return raw;
        let mean = 0;
        for (let i = 0; i < raw.length; i += 2) mean += raw[i];
        mean /= raw.length / 2;
        const out = raw.slice();
        for (let i = 0; i < out.length; i += 2) out[i] -= mean;
        return out;
    }, [raw, center]);

    const best = useMemo(() => linreg_fit(data), [data]);
    const lossMin = useMemo(() => linreg_mse(data, best[0], best[1]), [data, best]);

    const surface = useMemo(() => {
        const g = linreg_loss_grid(data, PARAM_VIEW.xmin, PARAM_VIEW.xmax, PARAM_VIEW.ymin, PARAM_VIEW.ymax, GRID, GRID);
        let max = 0;
        for (const v of g) max = Math.max(max, v);
        const tf = (v: number) => Math.log1p((v - lossMin) * 8) / Math.log1p((max - lossMin) * 8);
        const img = heatImage(g, GRID, GRID, (v) => sequential(0.06 + 0.66 * tf(v)).map((c) => c * 0.78) as [number, number, number]);
        const levels = Array.from({length: 14}, (_, i) => lossMin + Math.expm1(((i + 1) / 15) ** 1.6 * Math.log1p((max - lossMin) * 8)) / 8);
        const contours = contour_segments(g, GRID, GRID, PARAM_VIEW.xmin, PARAM_VIEW.xmax, PARAM_VIEW.ymin, PARAM_VIEW.ymax, new Float64Array(levels));
        return {img, contours};
    }, [data, lossMin]);

    const traj = useMemo(() => linreg_gd(data, start.m, start.b, lr, STEPS), [data, start, lr]);
    const K = traj.length / 3 - 1;
    const tl = useTimeline(K, 20, {autoplay: true, runKey: traj});
    const k = Math.min(K, Math.floor(tl.t));
    const f = tl.t - k;
    const k1 = Math.min(K, k + 1);
    const cur = {m: lerp(traj[3 * k], traj[3 * k1], f), b: lerp(traj[3 * k + 1], traj[3 * k1 + 1], f)};
    const loss = linreg_mse(data, cur.m, cur.b);
    const grad = linreg_grad(data, cur.m, cur.b);
    const diverged = !Number.isFinite(traj[traj.length - 1]) || traj[traj.length - 1] > traj[2] * 10;

    const converge = useMemo(() => {
        for (let i = 0; i <= K; i++) if (traj[3 * i + 2] - lossMin < 1e-3) return i;
        return null;
    }, [traj, K, lossMin]);

    const losses = useMemo(() => {
        const out = new Float64Array(K + 1);
        for (let i = 0; i <= K; i++) out[i] = traj[3 * i + 2] - lossMin + 1e-6;
        return out;
    }, [traj, K, lossMin]);

    const onParam = (e: PlotPointer) => {
        if (e.type === "down") {
            setDragging(true);
            setStart({m: e.x, b: e.y});
            tl.reset();
        }
        if (e.type === "move" && dragging) {
            setStart({m: e.x, b: e.y});
            tl.reset();
        }
        if (e.type === "up") setDragging(false);
    };

    return (
        <Instrument hint="Click or drag in parameter space to choose a starting line">
            <div className="duo">
                <Panel label="Data space" right={`ŷ = ${cur.m.toFixed(2)}x ${cur.b >= 0 ? "+" : "−"} ${Math.abs(cur.b).toFixed(2)}`}>
                    <Plot
                        label="Data and the current line"
                        view={DATA_VIEW}
                        draw={(p) => {
                            p.clear();
                            p.grid(0.25);
                            p.ticks(1, 1);
                            p.infiniteLine(best[0], -1, best[1], {color: C.green, width: 1.5, dash: [6, 6], alpha: 0.7});
                            for (let i = 0; i < data.length; i += 2) {
                                const yh = cur.m * data[i] + cur.b;
                                p.line(data[i], data[i + 1], data[i], yh, {color: rgba(C.red, 0.6), width: 1.2});
                            }
                            p.infiniteLine(cur.m, -1, cur.b, {color: C.yellow, width: 2.5});
                            for (let i = 0; i < data.length; i += 2) p.point(data[i], data[i + 1], C.blue, {r: 4});
                            p.circle(0, cur.b, 5, {fill: C.yellow});
                            p.text("b", 0, cur.b, {color: C.yellow, dx: -14, dy: 4, font: "serif", size: 15});
                        }}
                    />
                </Panel>
                <Panel label="Parameter space (m, b)" right={`loss ${fmt(loss, 3)}`}>
                    <Plot
                        label="Mean squared error as a function of slope and intercept"
                        view={PARAM_VIEW}
                        onPointer={onParam}
                        cursor={dragging ? "grabbing" : "crosshair"}
                        draw={(p) => {
                            p.clear();
                            p.image(surface.img, PARAM_VIEW);
                            p.contours(surface.contours, () => ({color: "rgba(255,255,255,0.16)", width: 1}));
                            p.grid(0.5, {majorEvery: 2});
                            p.ticks(1, 1, {xLabel: "m", yLabel: "b"});
                            const pts: number[] = [];
                            for (let i = 0; i <= k; i++) pts.push(traj[3 * i], traj[3 * i + 1]);
                            pts.push(cur.m, cur.b);
                            p.polyline(pts, {color: C.yellow, width: 1.6, alpha: 0.9});
                            for (let i = 0; i <= k; i++) p.circle(traj[3 * i], traj[3 * i + 1], 2.2, {fill: C.yellow, alpha: 0.7});
                            // Star at the minimum.
                            p.text("★", best[0], best[1], {color: C.green, size: 18, align: "center", baseline: "middle", halo: true});
                            // Negative gradient arrow.
                            const gn = Math.hypot(grad[0], grad[1]);
                            if (gn > 1e-6) {
                                const s = Math.min(0.5, gn * 0.12) / gn;
                                p.arrow(cur.m, cur.b, cur.m - grad[0] * s, cur.b - grad[1] * s, {color: C.red, width: 2.5, head: 9});
                            }
                            p.circle(cur.m, cur.b, 6, {fill: C.yellow, stroke: C.canvas, width: 2});
                            p.circle(start.m, start.b, 6, {stroke: C.white, width: 1.5, dash: [3, 3]});
                        }}
                    />
                </Panel>
            </div>
            <div className="figure-layout" style={{marginTop: 14, "--side": "300px"} as React.CSSProperties}>
                <div className="figure-main">
                    <Player timeline={tl} speeds={[5, 20, 60]} status={diverged ? <span className="c-red">The steps overshoot the valley and the loss explodes, so lower the learning rate.</span> : undefined} />
                    <LineChart label="Loss above the minimum, log scale" series={[{values: losses, color: C.yellow, upto: tl.t}]} logY height={120} xLabel="step" yLabel="L − L*  (log)" xMax={K} />
                </div>
                <div className="figure-side">
                    <Slider label="Learning rate η" value={lr} min={0.01} max={1.2} log onChange={(v) => (setLr(v), tl.reset())} color={C.yellow} format={(v) => v.toFixed(3)} />
                    <Toggle label="Center the inputs (subtract the mean of x)" checked={center} onChange={(v) => (setCenter(v), tl.reset())} />
                    <Stats>
                        <Stat label="step" value={k} />
                        <Stat label="steps to converge" value={converge ?? `> ${K}`} color={converge !== null ? C.green : C.red} />
                    </Stats>
                    <div className="control-row">
                        <Button onClick={() => (setSeed(randomSeed()), tl.reset())}>New data</Button>
                    </div>
                </div>
            </div>
        </Instrument>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

const SCENARIOS: Scenario[] = [
    {
        id: "by-eye",
        title: "Fit by eye",
        prompt: <>Drag the two white handles to make the total yellow area as small as you can, then hit <b>Snap to least squares</b> to see how close you got.</>,
        render: () => <SquaresFigure />,
    },
    {
        id: "outlier",
        title: "One outlier",
        prompt: <>One point sits far below the rest. Snap to least squares: its giant square drags the whole line toward it. Then drag that point back into line.</>,
        render: () => <SquaresFigure outlier />,
    },
    {
        id: "landscape",
        title: "Walk the loss bowl",
        prompt: <>Each line is one point in (m, b) space. Click anywhere on the loss surface to start gradient descent there and watch the line in data space follow.</>,
        render: () => <LandscapeFigure />,
    },
    {
        id: "too-big",
        title: "Step too big",
        prompt: <>The learning rate is just past the stability limit: each step overshoots the valley a little more, and the loss explodes.</>,
        render: () => <LandscapeFigure lr={0.34} />,
    },
    {
        id: "centered",
        title: "Center the inputs",
        prompt: <>Same data, but x is shifted to have mean zero. The long tilted valley becomes a round bowl, and gradient descent converges in a handful of steps.</>,
        render: () => <LandscapeFigure center lr={0.9} />,
    },
];

export default function LinearRegression() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            The model predicts <M>{"\\hat y = m x + b"}</M>. Least squares picks the line that minimizes the sum of squared residuals: the
                            total area of the yellow squares.
                        </li>
                    </ul>
                    <MathBlock>{"L(m, b) = \\frac1n \\sum_i \\big(y_i - (m x_i + b)\\big)^2"}</MathBlock>
                    <ul>
                        <li>
                            In parameter space <M>{"L"}</M> is a quadratic bowl. Gradient descent steps along{" "}
                            <M>{"-\\nabla L = -\\frac2n \\sum_i (\\hat y_i - y_i)\\,(x_i, 1)"}</M>.
                        </li>
                        <li>
                            The bottom has a closed form, the normal equations: <M>{"m^\\ast = \\operatorname{Cov}(x,y)/\\operatorname{Var}(x)"}</M>,{" "}
                            <M>{"b^\\ast = \\bar y - m^\\ast \\bar x"}</M>.
                        </li>
                        <li>
                            Squaring makes the loss smooth and solvable, but it also makes outliers very influential. Uncentered inputs stretch the bowl into
                            a narrow valley, which slows gradient descent down.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
