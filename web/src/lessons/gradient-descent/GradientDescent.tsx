import {useMemo, useState} from "react";
import {Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {Callout, Figure, FigureLayout} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {clamp, fmt, lerp, useTimeline} from "../../lib/hooks";
import {heatImage, type View} from "../../lib/painter";
import {C, rgba, sequential} from "../../lib/theme";
import {contour_segments, func1d_curve, func1d_eval, gd1d, optimize, surface_eval, surface_grid} from "../../lib/wasm";

// ===========================================================================
// Figure 1 — gradient descent in one dimension
// ===========================================================================

const FUNCS = [
    {id: 0, name: "Parabola", tex: "f(x) = x^2", view: {xmin: -2.2, xmax: 2.2, ymin: -0.5, ymax: 4.6}, x0: -1.8},
    {id: 1, name: "Double well", tex: "f(x) = \\tfrac12 x^4 - x^2 + 0.3x + 0.6", view: {xmin: -1.9, xmax: 1.9, ymin: -0.6, ymax: 2.6}, x0: 1.7},
    {id: 2, name: "Plateau", tex: "f(x) = 1 - \\frac{1}{1 + x^2}", view: {xmin: -5, xmax: 5, ymin: -0.15, ymax: 1.15}, x0: 4.2},
];

function regime(lr: number) {
    const r = 1 - 2 * lr;
    if (Math.abs(lr - 0.5) < 0.015) return {text: "η = ½: lands exactly on the minimum in one step", color: C.green};
    if (r > 0) return {text: `each step multiplies x by ${r.toFixed(2)}: a smooth slide`, color: C.green};
    if (r > -1) return {text: `each step multiplies x by ${r.toFixed(2)}: it overshoots and oscillates, but shrinks`, color: C.yellow};
    return {text: `each step multiplies x by ${r.toFixed(2)}: |factor| > 1, so it diverges`, color: C.red};
}

function OneDFigure() {
    const [fi, setFi] = useState(0);
    const fn = FUNCS[fi];
    const [lr, setLr] = useState(0.15);
    const [x0, setX0] = useState(fn.x0);
    const curve = useMemo(() => func1d_curve(fn.id, fn.view.xmin, fn.view.xmax, 400), [fn]);
    const traj = useMemo(() => gd1d(fn.id, x0, lr, 60), [fn, x0, lr]);
    const K = traj.length / 2 - 1;
    const tl = useTimeline(K, 3);
    const k = Math.min(K, Math.floor(tl.t));
    const s = tl.t - k;
    const k1 = Math.min(K, k + 1);
    const x = lerp(traj[2 * k], traj[2 * k1], s);
    const yLin = lerp(traj[2 * k + 1], traj[2 * k1 + 1], s);
    const H = fn.view.ymax - fn.view.ymin;
    const hop = 4 * s * (1 - s) * Math.min(H * 0.12, Math.abs(traj[2 * k1] - traj[2 * k]) * 0.8 * (H / (fn.view.xmax - fn.view.xmin)));
    const [fx, dfx] = func1d_eval(fn.id, traj[2 * k]);
    const reg = fn.id === 0 ? regime(lr) : null;

    const choose = (i: number) => {
        setFi(i);
        setX0(FUNCS[i].x0);
        tl.reset();
    };

    return (
        <Figure n={1} title="Rolling downhill, one step at a time" hint="Click anywhere on the plot to move the starting point">
            <FigureLayout
                main={
                    <Plot
                        label="One-dimensional gradient descent on a curve"
                        view={fn.view}
                        aspect={0.62}
                        onPointer={(e) => {
                            if (e.type === "down" || (e.type === "move" && e.dragging)) {
                                setX0(clamp(e.x, fn.view.xmin * 0.98, fn.view.xmax * 0.98));
                                tl.reset();
                            }
                        }}
                        draw={(p) => {
                            p.clear();
                            p.grid(fn.id === 2 ? 1 : 0.5, {majorEvery: 2});
                            p.polyline(curve, {color: C.white, width: 2.5});
                            // Past iterates and their hops.
                            for (let i = 0; i < k; i++) {
                                p.line(traj[2 * i], traj[2 * i + 1], traj[2 * i + 2], traj[2 * i + 3], {color: rgba(C.yellow, 0.35), width: 1.2, dash: [3, 4]});
                                p.circle(traj[2 * i], traj[2 * i + 1], 3.5, {fill: rgba(C.yellow, 0.5)});
                            }
                            // Tangent at the current iterate.
                            const xk = traj[2 * k];
                            const span = (fn.view.xmax - fn.view.xmin) * 0.12;
                            p.line(xk - span, fx - dfx * span, xk + span, fx + dfx * span, {color: C.pink, width: 2});
                            // Step arrow along the x axis: −η f'(x).
                            const base = fn.view.ymin + (fn.view.ymax - fn.view.ymin) * 0.06;
                            const step = -lr * dfx;
                            if (Math.abs(step) * p.kx > 4 && k < K)
                                p.arrow(xk, base, clamp(xk + step, fn.view.xmin, fn.view.xmax), base, {color: C.yellow, width: 2.5, head: 9});
                            p.line(xk, base, xk, fx, {color: rgba(C.yellow, 0.3), width: 1, dash: [2, 3]});
                            p.circle(x, yLin + hop, 8, {fill: C.yellow, stroke: C.canvas, width: 2});
                            p.textPx(fn.name, 12, 20, {color: C.muted, size: 12});
                        }}
                    />
                }
                side={
                    <div className="controls">
                        <Segmented value={fi} onChange={choose} options={FUNCS.map((f, i) => ({value: i, label: f.name}))} />
                        <div className="note">
                            <M>{fn.tex}</M>
                        </div>
                        <Slider label="Learning rate η" value={lr} min={0.01} max={1.6} log onChange={(v) => (setLr(v), tl.reset())} color={C.yellow} format={(v) => v.toFixed(3)} />
                        {reg && (
                            <p className="note" style={{color: reg.color}}>
                                {reg.text}
                            </p>
                        )}
                        <Stats>
                            <Stat label="x" value={fmt(traj[2 * k], 3)} color={C.yellow} />
                            <Stat label="f(x)" value={fmt(fx, 3)} />
                            <Stat label="f′(x)" value={fmt(dfx, 3)} color={C.pink} />
                            <Stat label="step −ηf′" value={fmt(-lr * dfx, 3)} color={C.yellow} />
                        </Stats>
                        <Player timeline={tl} showSpeed={false} status={`step ${k} of ${K}`} />
                    </div>
                }
            />
        </Figure>
    );
}

// ===========================================================================
// Figure 2 — optimizers racing on 2D surfaces
// ===========================================================================

interface SurfaceDef {
    id: number;
    name: string;
    view: View;
    start: [number, number];
    lr: [number, number, number, number];
    blurb: string;
}

const SURFACES: SurfaceDef[] = [
    {
        id: 0,
        name: "Narrow bowl",
        view: {xmin: -2.4, xmax: 2.4, ymin: -2.0, ymax: 2.0},
        start: [-2.0, 0.6],
        lr: [0.17, 0.012, 0.05, 0.1],
        blurb: "A stretched, tilted bowl. Plain gradient descent zig-zags across the valley.",
    },
    {
        id: 1,
        name: "Rosenbrock",
        view: {xmin: -2, xmax: 2, ymin: -1, ymax: 3},
        start: [-1.5, 2.4],
        lr: [0.012, 0.0015, 0.02, 0.04],
        blurb: "A curved valley that is easy to fall into and slow to follow. The minimum is at (1, 1).",
    },
    {
        id: 2,
        name: "Four minima",
        view: {xmin: -2.6, xmax: 2.6, ymin: -2.6, ymax: 2.6},
        start: [-0.2, -0.1],
        lr: [0.05, 0.008, 0.03, 0.06],
        blurb: "Himmelblau's function has four equally good minima. Where you end up depends on where you start, and on the optimizer.",
    },
    {
        id: 3,
        name: "Saddle",
        view: {xmin: -2, xmax: 2, ymin: -2, ymax: 2},
        start: [-1.8, 0.01],
        lr: [0.1, 0.02, 0.04, 0.05],
        blurb: "Starting almost exactly on the ridge of a saddle, the gradient barely points toward either valley.",
    },
    {
        id: 4,
        name: "Egg crate",
        view: {xmin: -3, xmax: 3, ymin: -3, ymax: 3},
        start: [2.5, 2.3],
        lr: [0.05, 0.03, 0.08, 0.12],
        blurb: "Many local minima. Momentum can carry an optimizer over small bumps; plain gradient descent stops in the first dip.",
    },
];

const OPTIMIZERS = [
    {id: 0, name: "Gradient descent", color: C.white},
    {id: 1, name: "Momentum", color: C.blue},
    {id: 2, name: "RMSProp", color: C.green},
    {id: 3, name: "Adam", color: C.pink},
];

const G = 150;
const STEPS = 400;

function RaceFigure() {
    const [si, setSi] = useState(0);
    const surf = SURFACES[si];
    const [start, setStart] = useState<[number, number]>(surf.start);
    const [lrs, setLrs] = useState<number[]>(surf.lr);
    const [enabled, setEnabled] = useState([true, true, false, true]);
    const [hover, setHover] = useState<{x: number; y: number} | null>(null);

    const field = useMemo(() => {
        const v = surf.view;
        const g = surface_grid(surf.id, v.xmin, v.xmax, v.ymin, v.ymax, G, G);
        let lo = Infinity;
        let hi = -Infinity;
        for (const x of g) {
            lo = Math.min(lo, x);
            hi = Math.max(hi, x);
        }
        const tf = (x: number) => Math.log1p(((x - lo) / (hi - lo)) * 30) / Math.log1p(30);
        const img = heatImage(g, G, G, (x) => sequential(0.05 + 0.62 * tf(x)).map((c) => c * 0.75) as [number, number, number]);
        const levels = Array.from({length: 16}, (_, i) => lo + (Math.expm1(((i + 0.5) / 16) * Math.log1p(30)) / 30) * (hi - lo));
        const contours = contour_segments(g, G, G, v.xmin, v.xmax, v.ymin, v.ymax, new Float64Array(levels));
        return {img, contours, lo};
    }, [surf]);

    const runs = useMemo(() => OPTIMIZERS.map((o, i) => optimize(surf.id, o.id, start[0], start[1], lrs[i], STEPS)), [surf, start, lrs]);
    const tl = useTimeline(STEPS, 40);

    const offset = useMemo(() => {
        let lo = field.lo;
        runs.forEach((r) => {
            for (let i = 2; i < r.length; i += 3) if (Number.isFinite(r[i])) lo = Math.min(lo, r[i]);
        });
        return lo;
    }, [runs, field]);

    const posAt = (r: Float64Array, t: number) => {
        const K = r.length / 3 - 1;
        const k = Math.min(K, Math.floor(t));
        const k1 = Math.min(K, k + 1);
        const s = t - k;
        return {x: lerp(r[3 * k], r[3 * k1], s), y: lerp(r[3 * k + 1], r[3 * k1 + 1], s), f: r[3 * k + 2], k};
    };

    const choose = (i: number) => {
        setSi(i);
        setStart(SURFACES[i].start);
        setLrs(SURFACES[i].lr);
        tl.reset();
    };

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") {
            setStart([e.x, e.y]);
            tl.reset();
            tl.play();
        }
        if (e.type === "move") setHover({x: e.x, y: e.y});
        if (e.type === "leave") setHover(null);
    };

    const lossSeries = OPTIMIZERS.map((o, i) => {
        const r = runs[i];
        const vals = new Float64Array(r.length / 3);
        for (let k = 0; k < vals.length; k++) vals[k] = r[3 * k + 2] - offset + 1e-4;
        return {values: vals, color: o.color, upto: tl.t};
    }).filter((_, i) => enabled[i]);

    return (
        <Figure n={2} title="Four optimizers, one landscape" hint="Click anywhere to drop the optimizers there">
            <FigureLayout
                sideWidth={300}
                main={
                    <>
                        <Plot
                            label="Optimizer trajectories on a 2D loss surface"
                            view={surf.view}
                            maxHeight={620}
                            onPointer={onPointer}
                            draw={(p) => {
                                p.clear();
                                p.image(field.img, surf.view);
                                p.contours(field.contours, () => ({color: "rgba(255,255,255,0.14)", width: 1}));
                                p.grid(0.5, {majorEvery: 2});
                                // Gradient probe under the cursor: it is perpendicular to the contour.
                                if (hover) {
                                    const [, gx, gy] = surface_eval(surf.id, hover.x, hover.y);
                                    const gn = Math.hypot(gx, gy);
                                    if (gn > 1e-9) {
                                        const L = 0.35;
                                        p.arrow(hover.x, hover.y, hover.x - (gx / gn) * L, hover.y - (gy / gn) * L, {color: C.red, width: 2, head: 8});
                                        p.line(hover.x - (gy / gn) * L * 0.6, hover.y + (gx / gn) * L * 0.6, hover.x + (gy / gn) * L * 0.6, hover.y - (gx / gn) * L * 0.6, {
                                            color: rgba(C.white, 0.6),
                                            width: 1.2,
                                            dash: [3, 3],
                                        });
                                    }
                                }
                                OPTIMIZERS.forEach((o, i) => {
                                    if (!enabled[i]) return;
                                    const r = runs[i];
                                    const cur = posAt(r, tl.t);
                                    const pts: number[] = [];
                                    for (let k = 0; k <= cur.k; k++) pts.push(r[3 * k], r[3 * k + 1]);
                                    pts.push(cur.x, cur.y);
                                    p.polyline(pts, {color: o.color, width: 2, alpha: 0.9});
                                    p.circle(cur.x, cur.y, 6.5, {fill: o.color, stroke: C.canvas, width: 2});
                                });
                                p.circle(start[0], start[1], 6, {stroke: C.white, width: 1.5, dash: [3, 3]});
                            }}
                        />
                        <Player timeline={tl} speeds={[10, 40, 120]} status={surf.blurb} />
                    </>
                }
                side={
                    <div className="controls">
                        <Segmented value={si} onChange={choose} options={SURFACES.map((s, i) => ({value: i, label: s.name}))} />
                        {OPTIMIZERS.map((o, i) => {
                            const r = runs[i];
                            const cur = posAt(r, tl.t);
                            return (
                                <div key={o.id} className="side-block" style={{opacity: enabled[i] ? 1 : 0.5}}>
                                    <label className="toggle">
                                        <input type="checkbox" checked={enabled[i]} onChange={(e) => setEnabled(enabled.map((v, j) => (j === i ? e.target.checked : v)))} />
                                        <span className="toggle-track" style={{background: enabled[i] ? o.color : undefined}} aria-hidden />
                                        <span className="toggle-label" style={{color: o.color, fontWeight: 600}}>
                                            {o.name}
                                        </span>
                                        <span className="mono" style={{marginLeft: "auto", color: "var(--muted)", fontSize: 12}}>
                                            f = {fmt(cur.f, 3)}
                                        </span>
                                    </label>
                                    <Slider
                                        label="step size η"
                                        value={lrs[i]}
                                        min={0.0005}
                                        max={1}
                                        log
                                        color={o.color}
                                        format={(v) => v.toPrecision(2)}
                                        onChange={(v) => {
                                            setLrs(lrs.map((x, j) => (j === i ? v : x)));
                                            tl.reset();
                                        }}
                                        disabled={!enabled[i]}
                                    />
                                </div>
                            );
                        })}
                        <LineChart label="Loss over steps" series={lossSeries} logY height={110} xMax={STEPS} xLabel="step" yLabel="f − f_min (log)" />
                    </div>
                }
            />
        </Figure>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

export default function GradientDescent() {
    return (
        <>
            <p>
                Nearly every model in machine learning is trained the same way. Write down a number, the <strong>loss</strong>, that measures how wrong
                the model is. Then adjust the parameters, a little at a time, in whichever direction makes that number smaller. That is{" "}
                <strong>gradient descent</strong>. It is easy to state, and full of surprises once you actually run it.
            </p>

            <h2 id="one-d">The idea in one dimension</h2>
            <p>
                Picture a ball on a curve <M>{"f(x)"}</M>. The derivative <M>{"\\pnk{f'(x)}"}</M> is the slope of the{" "}
                <span className="c-pink">tangent line</span>: positive means uphill to the right, negative means uphill to the left. To go downhill, move
                against the slope:
            </p>
            <MathBlock label="gradient descent">{"x_{t+1} = x_t - \\yel{\\eta}\\, \\pnk{f'(x_t)}"}</MathBlock>
            <p>
                The <span className="c-yellow">learning rate</span> <M>{"\\eta"}</M> sets how far each step goes. Note that the step is proportional to the
                slope: steep ground means big steps, and flat ground means tiny ones.
            </p>

            <OneDFigure />

            <p>
                The parabola <M>{"f(x) = x^2"}</M> lets us do the whole analysis exactly. Its derivative is <M>{"2x"}</M>, so one step gives
            </p>
            <MathBlock>{"x_{t+1} = x_t - \\eta \\cdot 2x_t = (1 - 2\\eta)\\, x_t \\quad\\Longrightarrow\\quad x_t = (1 - 2\\eta)^t\\, x_0"}</MathBlock>
            <p>Everything depends on the factor <M>{"1 - 2\\eta"}</M>:</p>
            <ul>
                <li>
                    <M>{"0 < \\eta < \\tfrac12"}</M>: the factor is between 0 and 1. The ball slides smoothly to the bottom.
                </li>
                <li>
                    <M>{"\\eta = \\tfrac12"}</M>: the factor is 0. One perfect step lands exactly on the minimum.
                </li>
                <li>
                    <M>{"\\tfrac12 < \\eta < 1"}</M>: the factor is negative but smaller than 1 in size. Each step overshoots to the other side, by less
                    each time.
                </li>
                <li>
                    <M>{"\\eta > 1"}</M>: overshoots grow and the iterates <span className="c-red">diverge</span> to infinity.
                </li>
            </ul>
            <p>
                In general, the critical learning rate depends on the <em>curvature</em> <M>{"f''"}</M>. Gradient descent is stable when{" "}
                <M>{"\\eta < 2 / f''"}</M>, so sharply curved valleys demand small steps.
            </p>

            <Callout kind="try">
                <ul>
                    <li>On the double well, start on the right. Gradient descent rolls into the nearer, shallower dip and stays there. It is a local method and only sees the slope where it stands.</li>
                    <li>
                        On the plateau, start far out. The slope is almost zero there, so steps are tiny and progress crawls. Flat regions, or{" "}
                        <em>vanishing gradients</em>, are a real obstacle when training deep networks.
                    </li>
                </ul>
            </Callout>

            <h2 id="two-d">Two dimensions: the gradient</h2>
            <p>
                With two parameters the loss is a landscape <M>{"f(x, y)"}</M>, and the slope becomes a vector, the <strong>gradient</strong>:
            </p>
            <MathBlock>{"\\nabla f = \\begin{pmatrix} \\partial f / \\partial x \\\\ \\partial f / \\partial y \\end{pmatrix}"}</MathBlock>
            <p>
                It points in the direction of steepest ascent, and it is always <strong>perpendicular to the contour lines</strong>. Hover over the
                landscape below to see the <span className="c-red">downhill direction</span> next to the local contour (dashed). Following{" "}
                <M>{"-\\nabla f"}</M> blindly is plain gradient descent. Its weakness is that steepest isn't the same as shortest: in a narrow valley, the
                steepest direction points mostly <em>across</em> the valley rather than along it.
            </p>
            <h3>Three fixes</h3>
            <p>
                <strong className="c-blue">Momentum</strong> treats the parameters like a heavy ball. It accumulates a velocity, so zig-zags that point in
                alternating directions cancel out, and steps along the valley floor add up:
            </p>
            <MathBlock>{"\\vv \\leftarrow \\beta\\, \\vv + \\nabla f, \\qquad \\theta \\leftarrow \\theta - \\eta\\, \\vv \\qquad (\\beta \\approx 0.9)"}</MathBlock>
            <p>
                <strong className="c-green">RMSProp</strong> rescales each coordinate by a running average of its squared gradients. Directions that are
                consistently steep get smaller steps, and flat directions get larger ones:
            </p>
            <MathBlock>{"\\mathbf{s} \\leftarrow \\rho\\, \\mathbf{s} + (1 - \\rho)\\, (\\nabla f)^2, \\qquad \\theta \\leftarrow \\theta - \\eta\\, \\frac{\\nabla f}{\\sqrt{\\mathbf{s}} + \\epsilon}"}</MathBlock>
            <p>
                <strong className="c-pink">Adam</strong> combines the two: momentum on the gradient and RMSProp-style scaling, with a correction for the
                first few steps. It is the default optimizer for most deep learning.
            </p>

            <RaceFigure />

            <Callout>
                There is no universally best optimizer. Adam's per-coordinate scaling makes it robust to badly scaled problems. The same scaling also makes
                its steps roughly the size of <M>{"\\eta"}</M>, regardless of how small the gradient is, so it can rattle around a minimum instead of
                settling. Momentum can roll right out of a good valley. The landscape decides which one wins.
            </Callout>

            <section className="takeaways">
                <h2>Takeaways</h2>
                <ul>
                    <li>
                        Gradient descent steps against the slope: <M>{"\\theta \\leftarrow \\theta - \\eta \\nabla f"}</M>. It only sees the local slope
                        and can stop in any local minimum.
                    </li>
                    <li>
                        The learning rate must fit the curvature. Too small and progress crawls, too large and the iterates oscillate or diverge. For{" "}
                        <M>{"x^2"}</M> the boundary is exactly <M>{"\\eta = 1"}</M>.
                    </li>
                    <li>The gradient is perpendicular to the contours, which is why narrow valleys cause zig-zagging.</li>
                    <li>Momentum, RMSProp and Adam reshape the steps to cope with valleys, plateaus and saddles.</li>
                </ul>
            </section>
        </>
    );
}
