import {useMemo, useState} from "react";
import {Link} from "react-router";
import {Button, Legend, Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {Callout, Figure, FigureLayout} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {probabilityColor} from "../../lib/colors";
import {appendRow, drawLabelled, hitPoint, removeRow} from "../../lib/data";
import {lerp, useTimeline} from "../../lib/hooks";
import {heatImage, type View} from "../../lib/painter";
import {C, classColor, rgba} from "../../lib/theme";
import {logreg_grid, logreg_loss, logreg_train, make_two_blobs, randomSeed} from "../../lib/wasm";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// ===========================================================================
// Figure 1 — a sigmoid on a line
// ===========================================================================

const VIEW_1D: View = {xmin: -2, xmax: 2, ymin: -0.18, ymax: 1.18};
const LOSS_VIEW: View = {xmin: -2, xmax: 2, ymin: 0, ymax: 4.2};

function make1d(seed: number): Float64Array {
    const blobs = make_two_blobs(26, 1.6, 0.42, seed);
    const out = new Float64Array(26 * 2);
    for (let i = 0; i < 26; i++) {
        out[2 * i] = Math.max(-1.9, Math.min(1.9, blobs[3 * i]));
        out[2 * i + 1] = blobs[3 * i + 2];
    }
    return out;
}

function SigmoidFigure() {
    const [seed, setSeed] = useState(5);
    const data = useMemo(() => make1d(seed), [seed]);
    const [params, setParams] = useState({w: 1.2, b: 0.6});
    const [fit, setFit] = useState<Float64Array | null>(null);
    const K = fit ? fit.length / 4 - 1 : 0;
    const tl = useTimeline(K, 30);

    let w = params.w;
    let b = params.b;
    if (fit) {
        const k = Math.min(K, Math.floor(tl.t));
        const k1 = Math.min(K, k + 1);
        const s = tl.t - k;
        w = lerp(fit[4 * k], fit[4 * k1], s);
        b = lerp(fit[4 * k + 1], fit[4 * k1 + 1], s);
    }
    const [loss, acc] = logreg_loss(data, new Float64Array([w, b]), 0);

    const set = (next: {w: number; b: number}) => {
        setFit(null);
        setParams(next);
    };

    const runFit = () => {
        const traj = logreg_train(data, new Float64Array([w, b]), 1.5, 0, 300, 3);
        setParams({w: traj[traj.length - 4], b: traj[traj.length - 3]});
        setFit(traj);
        tl.reset();
        tl.play();
    };

    const jitter = (i: number) => (((i * 37) % 11) / 11 - 0.5) * 0.07;
    const threshold = Math.abs(w) > 1e-6 ? -b / w : NaN;

    return (
        <Figure n={1} title="Squashing a score into a probability" hint="Move the sliders, then let gradient descent fit them">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Sigmoid curve fit to one-dimensional labelled data"
                            view={VIEW_1D}
                            aspect={0.5}
                            draw={(p) => {
                                p.clear();
                                p.grid(0.25, {axes: false});
                                p.line(VIEW_1D.xmin, 0, VIEW_1D.xmax, 0, {color: C.axis, width: 1});
                                p.line(VIEW_1D.xmin, 1, VIEW_1D.xmax, 1, {color: C.axis, width: 1});
                                p.line(VIEW_1D.xmin, 0.5, VIEW_1D.xmax, 0.5, {color: rgba(C.muted, 0.4), width: 1, dash: [2, 5]});
                                if (Number.isFinite(threshold)) p.line(threshold, VIEW_1D.ymin, threshold, VIEW_1D.ymax, {color: C.white, width: 1.5, dash: [5, 5], alpha: 0.7});
                                const pts: number[] = [];
                                for (let i = 0; i <= 200; i++) {
                                    const x = VIEW_1D.xmin + ((VIEW_1D.xmax - VIEW_1D.xmin) * i) / 200;
                                    pts.push(x, sigmoid(w * x + b));
                                }
                                for (let i = 0; i < data.length; i += 2) {
                                    const x = data[i];
                                    const y = data[i + 1];
                                    p.line(x, y + jitter(i), x, sigmoid(w * x + b), {color: rgba(C.red, 0.45), width: 1.2});
                                }
                                p.polyline(pts, {color: C.yellow, width: 3});
                                for (let i = 0; i < data.length; i += 2) p.point(data[i], data[i + 1] + jitter(i), classColor(data[i + 1]), {r: 5});
                                p.textPx("p = 1", 8, p.sy(1) - 6, {color: C.muted, size: 11, font: "mono"});
                                p.textPx("p = 0", 8, p.sy(0) + 14, {color: C.muted, size: 11, font: "mono"});
                            }}
                        />
                        <div className="panel-label" style={{marginTop: 4}}>
                            <span>Each point's penalty −log p(correct label)</span>
                        </div>
                        <Plot
                            label="Per-point cross-entropy penalties"
                            view={LOSS_VIEW}
                            height={90}
                            draw={(p) => {
                                p.clear();
                                for (let i = 0; i < data.length; i += 2) {
                                    const x = data[i];
                                    const q = sigmoid(w * x + b);
                                    const pc = data[i + 1] > 0.5 ? q : 1 - q;
                                    const l = Math.min(4.1, -Math.log(Math.max(pc, 1e-9)));
                                    const half = 0.025;
                                    p.polyline([x - half, 0, x + half, 0, x + half, l, x - half, l], {close: true, fill: classColor(data[i + 1]), fillAlpha: 0.85, color: "none"});
                                }
                            }}
                        />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label={<>weight <M>{"\\yel{w}"}</M> (steepness)</>} value={w} min={-6} max={14} onChange={(v) => set({w: v, b})} color={C.yellow} />
                        <Slider label={<>bias <M>{"b"}</M> (shift)</>} value={b} min={-8} max={8} onChange={(v) => set({w, b: v})} color={C.text2} />
                        <Stats>
                            <Stat label="mean loss" value={loss.toFixed(3)} color={C.red} />
                            <Stat label="accuracy" value={`${Math.round(acc * 100)}%`} color={C.green} />
                        </Stats>
                        <div className="control-row">
                            <Button variant="primary" onClick={runFit}>
                                Fit by gradient descent
                            </Button>
                            <Button onClick={() => (setSeed(randomSeed()), setFit(null))}>New data</Button>
                        </div>
                        <Legend
                            items={[
                                {color: C.yellow, label: "σ(wx + b)", shape: "line"},
                                {color: C.white, label: "threshold p = ½", shape: "dash"},
                            ]}
                        />
                    </div>
                }
            />
        </Figure>
    );
}

// ===========================================================================
// Figure 2 — logistic regression in 2D
// ===========================================================================

const VIEW: View = {xmin: -1.4, xmax: 1.4, ymin: -1.4, ymax: 1.4};
const G = 120;
const RECORD = 2;
const STEPS = 1600;

function PlaneFigure() {
    const [seed, setSeed] = useState(2);
    const [sep, setSep] = useState(1.1);
    const [l2, setL2] = useState(0);
    const [extra, setExtra] = useState<Float64Array>(new Float64Array());
    const [brush, setBrush] = useState<0 | 1>(0);
    const base = useMemo(() => make_two_blobs(120, sep, 0.24, seed), [sep, seed]);
    const data = useMemo(() => {
        const out = new Float64Array(base.length + extra.length);
        out.set(base);
        out.set(extra, base.length);
        return out;
    }, [base, extra]);

    const traj = useMemo(() => logreg_train(data, new Float64Array([0, 0, 0]), 2, l2, STEPS, RECORD), [data, l2]);
    const K = traj.length / 5 - 1;
    const tl = useTimeline(K, 60);
    const k = Math.min(K, Math.floor(tl.t));
    const k1 = Math.min(K, k + 1);
    const s = tl.t - k;
    const w1 = lerp(traj[5 * k], traj[5 * k1], s);
    const w2 = lerp(traj[5 * k + 1], traj[5 * k1 + 1], s);
    const b = lerp(traj[5 * k + 2], traj[5 * k1 + 2], s);
    const loss = traj[5 * k + 3];
    const acc = traj[5 * k + 4];
    const norm = Math.hypot(w1, w2);

    const img = useMemo(() => heatImage(logreg_grid(w1, w2, b, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G), G, G, (p) => probabilityColor(p, 0.3)), [w1, w2, b]);
    const losses = useMemo(() => {
        const out = new Float64Array(K + 1);
        for (let i = 0; i <= K; i++) out[i] = traj[5 * i + 3];
        return out;
    }, [traj, K]);
    const norms = useMemo(() => {
        const out = new Float64Array(K + 1);
        for (let i = 0; i <= K; i++) out[i] = Math.hypot(traj[5 * i], traj[5 * i + 1]);
        return out;
    }, [traj, K]);

    const onPointer = (e: PlotPointer) => {
        if (e.type !== "down") return;
        const nBase = base.length / 3;
        const hit = hitPoint(data, 3, e.x, e.y, e.kx, e.ky);
        if (hit >= nBase) setExtra(removeRow(extra, 3, hit - nBase));
        else if (hit < 0) setExtra(appendRow(extra, [e.x, e.y, e.shift || e.button === 2 ? 1 - brush : brush]));
    };

    const levels = [0.1, 0.25, 0.5, 0.75, 0.9];

    return (
        <Figure n={2} title="Probabilities over the plane" hint="Click to add points · click an added point to remove it">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Logistic regression probability map"
                            view={VIEW}
                            maxHeight={580}
                            onPointer={onPointer}
                            draw={(p) => {
                                p.clear();
                                p.image(img, VIEW);
                                p.grid(0.25);
                                if (norm > 1e-6) {
                                    // Only label the level lines when they are far enough apart to read.
                                    const labelable = (Math.log(3) / norm) * p.kx > 30;
                                    for (const q of levels) {
                                        const z = Math.log(q / (1 - q));
                                        const isHalf = q === 0.5;
                                        p.infiniteLine(w1, w2, b - z, {color: isHalf ? C.white : rgba(C.white, 0.35), width: isHalf ? 2.5 : 1.2, dash: isHalf ? undefined : [4, 5]});
                                        // Label each level line near the right edge of the view.
                                        const seg = p.clipLine(w1, w2, b - z);
                                        if (seg && !isHalf && labelable) {
                                            const [x1, y1, x2, y2] = seg;
                                            const t = 0.85;
                                            p.text(q.toFixed(2), x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, {color: C.text2, size: 10.5, font: "mono", align: "center", baseline: "middle"});
                                        }
                                    }
                                    // w, drawn from the point of the boundary nearest the origin.
                                    const fx = (-b * w1) / (norm * norm);
                                    const fy = (-b * w2) / (norm * norm);
                                    const L = Math.min(0.9, 0.18 + norm * 0.06);
                                    p.arrow(fx, fy, fx + (w1 / norm) * L, fy + (w2 / norm) * L, {color: C.yellow, width: 3, head: 12});
                                }
                                drawLabelled(p, data, {r: 4.2});
                            }}
                        />
                        <Player timeline={tl} speeds={[20, 60, 200]} status={`gradient descent step ${k * RECORD} of ${STEPS}`} />
                    </>
                }
                side={
                    <div className="controls">
                        <Slider label="Class separation" value={sep} min={0} max={2.4} onChange={(v) => (setSep(v), tl.reset())} />
                        <Slider label={<>L2 penalty <M>{"\\lambda"}</M></>} value={l2} min={0} max={0.05} step={0.001} onChange={(v) => (setL2(v), tl.reset())} color={C.purple} format={(v) => v.toFixed(3)} />
                        <Stats>
                            <Stat label="loss" value={loss.toFixed(3)} color={C.red} />
                            <Stat label="accuracy" value={`${Math.round(acc * 100)}%`} color={C.green} />
                            <Stat label={<M>{"\\lVert\\vw\\rVert"}</M>} value={norm.toFixed(2)} color={C.yellow} />
                        </Stats>
                        <div className="panel-label">
                            <span>Loss</span>
                        </div>
                        <LineChart label="Cross-entropy loss over training" series={[{values: losses, color: C.red, upto: tl.t, dx: RECORD}]} height={80} yMin={0} />
                        <div className="panel-label">
                            <span>Weight norm</span>
                        </div>
                        <LineChart label="Weight norm over training" series={[{values: norms, color: C.yellow, upto: tl.t, dx: RECORD}]} height={80} yMin={0} />
                        <Segmented
                            label="Clicking adds"
                            value={brush}
                            onChange={setBrush}
                            options={[
                                {value: 1, label: <span className="c-blue">● class 1</span>},
                                {value: 0, label: <span className="c-orange">● class 0</span>},
                            ]}
                        />
                        <div className="control-row">
                            <Button onClick={() => (setSeed(randomSeed()), setExtra(new Float64Array()), tl.reset())}>New data</Button>
                        </div>
                    </div>
                }
            />
        </Figure>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

export default function LogisticRegression() {
    return (
        <>
            <p>
                The <Link to="/learn/perceptron">perceptron</Link> answers every question with a flat "yes" or "no". A point a hair's breadth from the boundary
                gets the same verdict as one far away, and when no perfect line exists, the algorithm never settles. <strong>Logistic regression</strong>{" "}
                keeps the same linear score but turns it into a <em>probability</em>, and that one change fixes both problems.
            </p>

            <h2 id="sigmoid">From scores to probabilities</h2>
            <p>
                Start with the same score as before, <M>{"z = \\vw\\cdot\\vx + b"}</M>. It can be any real number, and a probability has to live in{" "}
                <M>{"[0, 1]"}</M>. The <strong>sigmoid</strong> (or logistic) function squashes one into the other:
            </p>
            <MathBlock>{"\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\qquad P(y = 1 \\mid \\vx) = \\sigma(\\vw\\cdot\\vx + b)"}</MathBlock>
            <p>
                Large positive scores map to probabilities near 1, large negative scores to near 0, and a score of exactly 0 to ½. The boundary{" "}
                <M>{"z = 0"}</M> is the same line as the perceptron's. What's new is the gentle ramp around it. In 1D the weight{" "}
                <M>{"\\yel{w}"}</M> controls how steep the ramp is, which is to say how <em>confident</em> the model is.
            </p>
            <p>
                Reading the formula backwards gives a useful interpretation. The score is the <strong>log-odds</strong>,{" "}
                <M>{"z = \\log\\frac{p}{1-p}"}</M>, so each unit of <M>{"z"}</M> multiplies the odds of class 1 by <M>{"e \\approx 2.72"}</M>.
            </p>

            <h2 id="loss">Scoring a probability: cross-entropy</h2>
            <p>
                How should a probabilistic prediction be graded? We look at the probability the model gave to the label that actually happened, and charge
                its negative logarithm:
            </p>
            <MathBlock label="binary cross-entropy">{"\\mathcal{L} = -\\frac{1}{n}\\sum_{i=1}^n \\Big[ y_i \\log p_i + (1 - y_i) \\log (1 - p_i) \\Big]"}</MathBlock>
            <p>
                A point predicted correctly with confidence costs almost nothing. A point predicted at 50/50 costs <M>{"\\log 2 \\approx 0.69"}</M>. A point
                predicted <em>confidently wrong</em> costs a lot, and the cost grows without bound as the model's confidence in the wrong answer approaches
                certainty. Watch the bars under the curve as you drag the sliders.
            </p>

            <SigmoidFigure />

            <Callout kind="try">
                Make the curve very steep (large <M>{"w"}</M>) with the threshold in a slightly wrong place. The few points on the wrong side light up
                with huge penalties. Cross-entropy is unforgiving of overconfidence, which is exactly what makes the fitted probabilities honest.
            </Callout>

            <h2 id="gradient">The gradient is beautifully simple</h2>
            <p>
                Cross-entropy and the sigmoid fit together neatly. Because <M>{"\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))"}</M>, almost everything cancels
                when you differentiate, and the gradient is
            </p>
            <MathBlock>{"\\nabla_{\\vw} \\mathcal{L} = \\frac{1}{n}\\sum_{i=1}^n \\big(\\,p_i - y_i\\,\\big)\\,\\vx_i, \\qquad \\frac{\\partial \\mathcal{L}}{\\partial b} = \\frac{1}{n}\\sum_{i=1}^n \\big(p_i - y_i\\big)"}</MathBlock>
            <p>
                Compare it to the perceptron. Each point pushes the weights along <M>{"\\vx_i"}</M>, weighted by its error <M>{"p_i - y_i"}</M>. The
                perceptron pushed only for mistakes and always pushed by a full unit. Logistic regression pushes for every point, softly, in proportion to
                how wrong it was. Points that are confidently right contribute almost nothing. The loss is also <em>convex</em>: a single bowl with no
                local minima, so gradient descent will find the best answer.
            </p>

            <h2 id="plane">In two dimensions</h2>
            <p>
                In 2D, the probability is a smooth ramp across the plane. Its level sets <M>{"P = 0.1, 0.25, 0.5, \\ldots"}</M> are parallel lines, all
                perpendicular to <M>{"\\vw"}</M>. The spacing between them is inversely proportional to <M>{"\\lVert\\vw\\rVert"}</M>: a longer weight
                vector means a sharper, more confident transition.
            </p>

            <PlaneFigure />

            <Callout>
                Push the classes apart until they are perfectly separable, and watch <M>{"\\lVert\\vw\\rVert"}</M>. It never stops growing. On separable
                data the loss can always be lowered a little further by becoming more confident, so the probability lines squeeze together forever,
                toward a hard perceptron-style step. The fix is <strong>regularization</strong>: adding <M>{"\\tfrac{\\lambda}{2}\\lVert\\vw\\rVert^2"}</M>{" "}
                to the loss charges for confidence, and the weights settle at a finite size.
            </Callout>

            <section className="takeaways">
                <h2>Takeaways</h2>
                <ul>
                    <li>
                        Logistic regression keeps the linear score <M>{"\\vw\\cdot\\vx + b"}</M> and maps it through a sigmoid to get a probability. The
                        score is the log-odds.
                    </li>
                    <li>Cross-entropy charges <M>{"-\\log"}</M> of the probability given to the true label. Confident mistakes are very expensive.</li>
                    <li>
                        The gradient is <M>{"(p - y)\\,\\vx"}</M>, a soft, always-on version of the perceptron update, on a convex loss with a single
                        minimum.
                    </li>
                    <li>
                        <M>{"\\lVert\\vw\\rVert"}</M> sets the confidence. On separable data it grows without bound unless you regularize.
                    </li>
                </ul>
            </section>
        </>
    );
}
