import {useMemo, useState} from "react";
import {Link} from "react-router";
import {Button, Legend, Player, Slider, Stat, Stats} from "../../components/Controls";
import {Callout, Figure, FigureLayout, Panel} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Plot, type PlotPointer} from "../../components/Plot";
import {ease, useTimeline} from "../../lib/hooks";
import type {Painter, View} from "../../lib/painter";
import {C, rgba} from "../../lib/theme";
import {useTween} from "../../lib/tween";
import {make_gaussian, pca, randomSeed} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const N = 70;
const SAMPLES = 180;

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

function DirectionFigure() {
    const [seed, setSeed] = useState(8);
    const [stretch, setStretch] = useState(0.3);
    const data = useMemo(() => make_gaussian(N, 0.05, -0.05, 0.5, 0.5 * stretch, 0.6, seed), [seed, stretch]);
    const stats = useMemo(() => pca2(data), [data]);
    const [angle, animateAngle, setAngle] = useTween([2.2], 0.8);
    const theta = angle[0];
    const [dragging, setDragging] = useState(false);

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

    const snap = () => {
        // Rotate the short way: the line at θ and θ + π is the same line.
        let target = pcAngle;
        while (target - theta > Math.PI / 2) target -= Math.PI;
        while (target - theta < -Math.PI / 2) target += Math.PI;
        animateAngle([target]);
    };

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") setDragging(true);
        if (e.type === "up") setDragging(false);
        if ((e.type === "down" || (e.type === "move" && dragging)) && Math.hypot(e.x - stats.mean[0], e.y - stats.mean[1]) > 0.05) {
            let a = Math.atan2(e.y - stats.mean[1], e.x - stats.mean[0]);
            while (a - theta > Math.PI / 2) a -= Math.PI;
            while (a - theta < -Math.PI / 2) a += Math.PI;
            setAngle([a]);
        }
    };

    const ux = Math.cos(theta);
    const uy = Math.sin(theta);
    const [mx, my] = stats.mean;

    return (
        <Figure n={1} title="Squash the cloud onto a line" hint="Drag anywhere to rotate the line">
            <FigureLayout
                main={
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
                            onPick={(x) => setAngle([(Math.max(0, Math.min(180, x)) * Math.PI) / 180])}
                        />
                        <Stats>
                            <Stat label="θ" value={`${deg.toFixed(0)}°`} color={C.yellow} />
                            <Stat label="best θ" value={`${(((((pcAngle * 180) / Math.PI) % 180) + 180) % 180).toFixed(0)}°`} color={C.green} />
                        </Stats>
                        <Slider label="How elongated the cloud is" value={stretch} min={0.08} max={1} onChange={setStretch} color={C.teal} format={(v) => `${v.toFixed(2)}`} />
                        <div className="control-row">
                            <Button variant="primary" onClick={snap}>
                                Find the best line
                            </Button>
                            <Button onClick={() => setSeed(randomSeed())}>New sample</Button>
                        </div>
                    </div>
                }
            />
        </Figure>
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
    const tl = useTimeline(2, 0.5);
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
        <Figure n={2} title="PCA is a rotation, then a projection" hint="Press play">
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
        </Figure>
    );
}

// ===========================================================================
// Lesson
// ===========================================================================

export default function Pca() {
    return (
        <>
            <p>
                Real datasets often have hundreds of features, but much of that is redundant: height in centimeters and height in inches, or a dozen survey
                questions that all measure the same mood. <strong>Principal component analysis</strong> finds the few directions along which the data
                actually varies, so you can keep those and drop the rest. It is used to visualize, to compress, to denoise, and to speed up other
                algorithms.
            </p>
            <p>
                Everything about PCA can be seen in two dimensions, so that is where we will look. The goal there is to summarize each 2D point with a
                single number without losing much.
            </p>

            <h2 id="projection">The best line to project onto</h2>
            <p>
                To summarize 2D points with one number, choose a line through the center of the data, with unit direction <M>{"\\yel{\\vu}"}</M>, and
                record where each point lands when you drop it straight onto the line: <M>{"\\yel{\\vu}\\cdot(\\vx_i - \\bar\\vx)"}</M>. Some lines keep
                the points well spread out along them. Others crush the points together and lose the differences between them. PCA picks the line along
                which the projected points have the <strong>largest variance</strong>.
            </p>

            <DirectionFigure />

            <p>
                Notice the bar: the two colored pieces always add up to the same total. That is Pythagoras. For each point, the squared distance to the
                center splits into a part along the line and a part perpendicular to it:
            </p>
            <MathBlock>{"\\lVert \\vx_i - \\bar\\vx \\rVert^2 = \\underbrace{\\big(\\yel{\\vu}\\cdot(\\vx_i - \\bar\\vx)\\big)^2}_{\\text{kept}} + \\underbrace{\\pnk{\\text{dist}(\\vx_i, \\text{line})^2}}_{\\text{lost}}"}</MathBlock>
            <p>
                The left side is fixed by the data. So <em>maximizing the variance you keep</em> and <em>minimizing the squared error you throw away</em>{" "}
                are the same goal. The direction that best preserves the spread of the data is also the line that best fits it, in the perpendicular
                least-squares sense. That is different from <Link to="/learn/linear-regression">linear regression</Link>, which only measures vertical
                errors.
            </p>

            <h2 id="eigen">Where the answer comes from</h2>
            <p>
                Collect the data's spread into the <strong>covariance matrix</strong>{" "}
                <M>{"\\Sigma = \\frac1n \\sum_i (\\vx_i - \\bar\\vx)(\\vx_i - \\bar\\vx)^\\top"}</M>. The variance along any unit direction is then a
                quadratic form:
            </p>
            <MathBlock>{"\\operatorname{Var}\\big(\\yel{\\vu}\\cdot\\vx\\big) = \\yel{\\vu}^\\top \\Sigma\\, \\yel{\\vu}"}</MathBlock>
            <p>
                That is the yellow curve in Figure 1. Maximize it subject to <M>{"\\lVert\\vu\\rVert = 1"}</M> with a Lagrange multiplier and the
                condition that pops out is
            </p>
            <MathBlock>{"\\Sigma\\, \\yel{\\vu} = \\lambda\\, \\yel{\\vu}"}</MathBlock>
            <p>
                The best direction is an <strong>eigenvector</strong> of the covariance matrix, and the variance it captures is its eigenvalue{" "}
                <M>{"\\lambda"}</M>. The top eigenvector is the first <em>principal component</em>. The next one, orthogonal to it, captures the most of
                what's left, and so on. In <M>{"d"}</M> dimensions there are <M>{"d"}</M> of them, ordered by eigenvalue.
            </p>

            <h2 id="rotation">PCA as a change of basis</h2>
            <p>
                Put all the principal components together and they form a new set of axes. Re-expressing the data in those axes is just a rotation, after
                centering. In that frame the covariance matrix is <em>diagonal</em>: the new coordinates are uncorrelated, and their variances are the
                eigenvalues. Dimensionality reduction is then the easy part: drop the axes with small eigenvalues.
            </p>

            <RotateFigure />

            <p>
                The fraction <M>{"\\lambda_1 / (\\lambda_1 + \\lambda_2)"}</M> is the <strong>explained variance</strong> of the first component. In high
                dimensions you keep as many components as you need to explain, say, 95% of the variance. Image datasets with thousands of pixels are often
                summarized well by a few dozen components.
            </p>

            <Callout kind="note" title="Caveats">
                <ul>
                    <li>
                        <strong>Scale matters.</strong> PCA chases variance, so a feature measured in larger units dominates. Standardize features first
                        unless their units are comparable.
                    </li>
                    <li>
                        <strong>It is linear.</strong> PCA finds flat subspaces. Data on a curved surface, such as a spiral or a ring, needs nonlinear
                        methods like kernel PCA, t-SNE, UMAP or autoencoders.
                    </li>
                    <li>
                        <strong>Variance isn't importance.</strong> The direction that matters for your task, such as the one separating two classes, can
                        have small variance and be the first thing PCA throws away.
                    </li>
                </ul>
            </Callout>

            <section className="takeaways">
                <h2>Takeaways</h2>
                <ul>
                    <li>PCA projects data onto the directions of greatest variance. By Pythagoras, these are also the directions of least reconstruction error.</li>
                    <li>Those directions are the eigenvectors of the covariance matrix, and the eigenvalues are the variances they capture.</li>
                    <li>In the eigenvector basis the covariance is diagonal. Dropping low-variance axes is dimensionality reduction.</li>
                    <li>Standardize first, and remember that PCA is linear and unsupervised.</li>
                </ul>
            </section>
        </>
    );
}
