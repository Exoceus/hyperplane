import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {FigureLayout} from "../../components/Figure";
import {M, MathBlock} from "../../components/Math";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {Plot, type PlotPointer} from "../../components/Plot";
import {appendRow, drawLabelled, hitPoint, removeRow, setRow} from "../../lib/data";
import {ease, useTimeline} from "../../lib/hooks";
import {heatImage, tint, type Painter, type View} from "../../lib/painter";
import {C, CLASS_RGB, rgba} from "../../lib/theme";
import {contour_segments, make_shape, make_two_blobs, randomSeed, svm_decisions, svm_grid_from, svm_trace, svm_train} from "../../lib/wasm";

const VIEW: View = {xmin: -1.4, xmax: 1.4, ymin: -1.4, ymax: 1.4};
const G = 110;
const FRAMES = 100;

type DataKind = "separable" | "overlap" | "outlier" | "circles" | "xor" | "moons";

function generate(kind: DataKind, seed: number): Float64Array {
    switch (kind) {
        case "separable":
            return make_two_blobs(60, 1.7, 0.2, seed);
        case "overlap":
            return make_two_blobs(90, 0.95, 0.28, seed);
        case "outlier":
            return appendRow(make_two_blobs(60, 1.7, 0.2, seed), [-0.35, -0.15, 1]);
        default:
            return make_shape(kind, 150, kind === "moons" ? 0.18 : 0.12, seed);
    }
}

const KERNELS = [
    {value: 0, label: "Linear"},
    {value: 1, label: "RBF"},
    {value: 2, label: "Polynomial"},
];

interface Fit {
    alphas: Float64Array;
    w: Float64Array;
    regions: HTMLCanvasElement;
    contours: Float64Array;
    acc: number;
    iterations: number;
}

// ===========================================================================
// 2D SVM instrument
// ===========================================================================

function SvmInstrument(props: {kind?: DataKind; kernel?: number; c?: number; gamma?: number; degree?: number}) {
    const [kind] = useState<DataKind>(props.kind ?? "separable");
    const [data, setData] = useState<Float64Array>(() => generate(props.kind ?? "separable", 5));
    const [kernel, setKernel] = useState(props.kernel ?? 0);
    const [c, setC] = useState(props.c ?? 1000);
    const [gamma, setGamma] = useState(props.gamma ?? 2);
    const [degree, setDegree] = useState(props.degree ?? 2);
    const [brush, setBrush] = useState(1);
    const [drag, setDrag] = useState<number | null>(null);

    // The whole SMO run, recorded; the animation replays it whenever anything changes.
    const trace = useMemo(() => svm_trace(data, 2, c, kernel, gamma, degree, 1, 50_000, FRAMES), [data, c, kernel, gamma, degree]);
    const n = data.length / 3;
    const stride = n + 3;
    const frameCount = trace.length / stride;
    const tl = useTimeline(frameCount - 1, 1, {autoplay: true, runKey: trace, duration: 3});
    const f = drag !== null ? frameCount - 1 : Math.min(frameCount - 1, Math.round(tl.t));
    const pairI = trace[f * stride];
    const pairJ = trace[f * stride + 1];

    const fit = useMemo<Fit>(() => {
        const alphas = trace.slice(f * stride + 3, (f + 1) * stride);
        const rho = trace[f * stride + 2];
        const grid = svm_grid_from(data, alphas, rho, kernel, gamma, degree, 1, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G);
        const regions = heatImage(grid, G, G, (v) => tint(CLASS_RGB[v >= 0 ? 1 : 0], 0.1 + 0.22 * Math.min(1, Math.abs(v))));
        const contours = contour_segments(grid, G, G, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, new Float64Array([-1, 0, 1]));
        // Weight vector (meaningful for the linear kernel) and training accuracy of this snapshot.
        const w = new Float64Array(2);
        const queries = new Float64Array(n * 2);
        for (let i = 0; i < n; i++) {
            const yi = data[3 * i + 2] > 0.5 ? 1 : -1;
            w[0] += alphas[i] * yi * data[3 * i];
            w[1] += alphas[i] * yi * data[3 * i + 1];
            queries[2 * i] = data[3 * i];
            queries[2 * i + 1] = data[3 * i + 1];
        }
        const dec = svm_decisions(data, alphas, rho, kernel, gamma, degree, 1, queries);
        let correct = 0;
        for (let i = 0; i < n; i++) if (dec[i] >= 0 === data[3 * i + 2] > 0.5) correct++;
        return {alphas, w, regions, contours, acc: correct / n, iterations: f};
    }, [trace, f, stride, data, kernel, gamma, degree, n]);

    let svCount = 0;
    let bounded = 0;
    for (const a of fit.alphas) {
        if (a > 1e-6) svCount++;
        if (a > c * (1 - 1e-6)) bounded++;
    }
    const wNorm = Math.hypot(fit.w[0], fit.w[1]);

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") {
            const hit = hitPoint(data, 3, e.x, e.y, e.kx, e.ky, 9);
            if (hit >= 0 && (e.shift || e.button === 2)) {
                if (n > 4) setData(removeRow(data, 3, hit));
            } else if (hit >= 0) setDrag(hit);
            else setData(appendRow(data, [e.x, e.y, brush]));
        }
        if (e.type === "move" && drag !== null) setData(setRow(data, 3, drag, [Math.max(-1.35, Math.min(1.35, e.x)), Math.max(-1.35, Math.min(1.35, e.y)), data[3 * drag + 2]]));
        if (e.type === "up") setDrag(null);
    };

    return (
        <Instrument hint="Drag points · click empty space to add · shift-click a point to delete it">
            <FigureLayout
                main={
                    <>
                    <Plot
                        label="Support vector machine decision regions and margin"
                        view={VIEW}
                        maxHeight={600}
                        cursor={drag !== null ? "grabbing" : "crosshair"}
                        onPointer={onPointer}
                        draw={(p) => {
                            p.clear();
                            p.image(fit.regions, VIEW);
                            p.grid(0.25, {axes: false});
                            p.contours(fit.contours, (lvl) => (lvl === 1 ? {color: C.white, width: 2.5} : {color: rgba(C.white, 0.7), width: 1.4, dash: [6, 5]}));
                            if (pairI >= 0 && f < frameCount - 1)
                                p.line(data[3 * pairI], data[3 * pairI + 1], data[3 * pairJ], data[3 * pairJ + 1], {color: C.yellow, width: 2, dash: [4, 4]});
                            drawLabelled(p, data, {
                                r: 4.5,
                                ring: (k) =>
                                    f < frameCount - 1 && (k === pairI || k === pairJ)
                                        ? C.yellow
                                        : fit.alphas[k] > c * (1 - 1e-6)
                                          ? C.red
                                          : fit.alphas[k] > 1e-6
                                            ? C.white
                                            : undefined,
                            });
                            if (kernel === 0 && wNorm > 1e-9) {
                                // Margin width annotation along w.
                                p.textPx(`street width 2/‖w‖ = ${(2 / wNorm).toFixed(3)}`, 12, 22, {color: C.text2, size: 12.5});
                            }
                        }}
                    />
                    <Player
                        timeline={tl}
                        speeds={[0.5, 1, 3]}
                        status={
                            f >= frameCount - 1 ? (
                                <span className="c-green">Converged: every point satisfies the margin conditions (KKT) up to tolerance.</span>
                            ) : f === 0 ? (
                                <>All α = 0: no support vectors yet.</>
                            ) : (
                                <>
                                    SMO snapshot {f} of {frameCount - 1}: re-optimizing the <span className="c-yellow">yellow pair</span> of multipliers while keeping
                                    every other α fixed.
                                </>
                            )
                        }
                    />
                    </>
                }
                side={
                    <div className="controls">
                        <Segmented label="kernel" value={kernel} onChange={setKernel} options={KERNELS} />
                        <Slider label={<>C (penalty for violations)</>} value={c} min={0.01} max={1000} log onChange={setC} color={C.yellow} format={(v) => (v >= 10 ? v.toFixed(0) : v.toPrecision(2))} />
                        {kernel !== 0 && (
                            <Slider label={<>γ (kernel width⁻¹)</>} value={gamma} min={0.1} max={100} log onChange={setGamma} color={C.purple} format={(v) => v.toPrecision(2)} />
                        )}
                        {kernel === 2 && <Slider label="degree" value={degree} min={2} max={5} step={1} onChange={setDegree} format={(v) => v.toFixed(0)} />}
                        <Stats>
                            <Stat label="support vectors" value={`${svCount} / ${n}`} />
                            <Stat label="inside the street" value={bounded} color={bounded ? C.red : C.green} />
                            <Stat label="train accuracy" value={`${Math.round(fit.acc * 100)}%`} color={C.green} />
                            <Stat label="snapshot" value={`${f} / ${frameCount - 1}`} />
                        </Stats>
                        <p className="note">
                            <span style={{color: C.white}}>○</span> support vectors define the boundary. <span className="c-red">○</span> are at the C limit:
                            inside the street or misclassified. Dashed lines are <M>{"f(\\vx) = \\pm 1"}</M>.
                        </p>
                        <Segmented
                            label="Clicking adds"
                            value={brush}
                            onChange={setBrush}
                            options={[
                                {value: 1, label: <span className="c-blue">● class +1</span>},
                                {value: 0, label: <span className="c-orange">● class −1</span>},
                            ]}
                        />
                        <div className="control-row">
                            <Button onClick={() => setData(generate(kind, randomSeed()))}>New data</Button>
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// The kernel trick, in 3D
// ===========================================================================

const VIEW3D: View = {xmin: -1.7, xmax: 1.7, ymin: -1.25, ymax: 1.45};

interface Camera {
    yaw: number;
    pitch: number;
}

function project(cam: Camera, x: number, y: number, z: number): [number, number, number] {
    const cy = Math.cos(cam.yaw);
    const sy = Math.sin(cam.yaw);
    const x1 = cy * x - sy * y;
    const y1 = sy * x + cy * y;
    const ce = Math.cos(cam.pitch);
    const se = Math.sin(cam.pitch);
    return [x1, z * ce + y1 * se - 0.25, y1 * ce - z * se];
}

function KernelTrick() {
    const data = useMemo(() => make_shape("circles", 140, 0.08, 9), []);
    const [cam, setCam] = useState<Camera>({yaw: 0.6, pitch: 0.42});
    const [last, setLast] = useState<{x: number; y: number} | null>(null);
    const tl = useTimeline(1, 0.5, {autoplay: true, runKey: data, delay: 0.8});
    const lift = ease(tl.t);

    const plane = useMemo(() => {
        const lifted = new Float64Array((data.length / 3) * 4);
        for (let i = 0, k = 0; i < data.length; i += 3) {
            lifted[k++] = data[i];
            lifted[k++] = data[i + 1];
            lifted[k++] = data[i] ** 2 + data[i + 1] ** 2;
            lifted[k++] = data[i + 2];
        }
        const m = svm_train(lifted, 3, 100, 0, 1, 1, 0, 50_000);
        const w = m.weights();
        const rho = m.rho;
        m.free();
        return {w1: w[0], w2: w[1], w3: w[2], rho};
    }, [data]);

    // The plane meets the bowl z = x² + y² in a circle; project it to the floor.
    const circle = useMemo(() => {
        const {w1, w2, w3, rho} = plane;
        const cx = -w1 / (2 * w3);
        const cy = -w2 / (2 * w3);
        const r2 = rho / w3 + cx * cx + cy * cy;
        return {cx, cy, r: Math.sqrt(Math.max(0, r2))};
    }, [plane]);

    const onPointer = (e: PlotPointer) => {
        if (e.type === "down") setLast({x: e.x, y: e.y});
        if (e.type === "move" && last) {
            setCam({yaw: cam.yaw + (e.x - last.x) * 1.6, pitch: Math.max(0.05, Math.min(1.5, cam.pitch - (e.y - last.y) * 1.6))});
            setLast({x: e.x, y: e.y});
        }
        if (e.type === "up") setLast(null);
    };

    const line3 = (p: Painter, a: [number, number, number], b: [number, number, number], color: string, width = 1, dash?: number[]) => {
        const pa = project(cam, ...a);
        const pb = project(cam, ...b);
        p.line(pa[0], pa[1], pb[0], pb[1], {color, width, dash});
    };

    const planeAlpha = ease((tl.t - 0.75) / 0.25);

    return (
        <Instrument hint="Watch the points lift into 3D · drag to rotate the view">
            <FigureLayout
                main={
                    <>
                        <Plot
                            label="Ring data lifted onto a paraboloid, separated by a plane"
                            view={VIEW3D}
                            aspect={0.78}
                            cursor={last ? "grabbing" : "grab"}
                            onPointer={onPointer}
                            draw={(p) => {
                                p.clear();
                                // Floor grid.
                                for (let i = -6; i <= 6; i++) {
                                    const t = i * 0.2;
                                    line3(p, [t, -1.2, 0], [t, 1.2, 0], i === 0 ? rgba(C.muted, 0.5) : rgba(C.muted, 0.18));
                                    line3(p, [-1.2, t, 0], [1.2, t, 0], i === 0 ? rgba(C.muted, 0.5) : rgba(C.muted, 0.18));
                                }
                                // The bowl z = lift·(x² + y²) as rings and spokes.
                                for (let r = 0.2; r <= 1.0001; r += 0.2) {
                                    const pts: number[] = [];
                                    for (let a = 0; a <= 64; a++) {
                                        const th = (a / 64) * Math.PI * 2;
                                        const [u, v] = project(cam, r * Math.cos(th), r * Math.sin(th), lift * r * r);
                                        pts.push(u, v);
                                    }
                                    p.polyline(pts, {color: rgba(C.purple, 0.25 * lift + 0.05), width: 1});
                                }
                                // The decision circle on the floor.
                                const ring: number[] = [];
                                for (let a = 0; a <= 90; a++) {
                                    const th = (a / 90) * Math.PI * 2;
                                    const [u, v] = project(cam, circle.cx + circle.r * Math.cos(th), circle.cy + circle.r * Math.sin(th), 0);
                                    ring.push(u, v);
                                }
                                p.polyline(ring, {color: C.white, width: 2, dash: [6, 5], alpha: planeAlpha});
                                // Points, far to near.
                                const order: {u: number; v: number; d: number; c: number; below: boolean}[] = [];
                                for (let i = 0; i < data.length; i += 3) {
                                    const x = data[i];
                                    const y = data[i + 1];
                                    const z = lift * (x * x + y * y);
                                    const [u, v, d] = project(cam, x, y, z);
                                    const below = plane.w1 * x + plane.w2 * y + plane.w3 * z - plane.rho < 0;
                                    order.push({u, v, d, c: data[i + 2], below: below === plane.w3 > 0});
                                }
                                order.sort((a, b) => b.d - a.d);
                                // Points under the plane are drawn first, so the translucent plane covers them.
                                if (planeAlpha > 0) for (const o of order) if (o.below) p.point(o.u, o.v, o.c > 0.5 ? C.blue : C.orange, {r: 4});
                                // Separating plane (drawn mid-way so near points sit on top of it).
                                if (planeAlpha > 0) {
                                    const {w1, w2, w3, rho} = plane;
                                    const z = (x: number, y: number) => (rho - w1 * x - w2 * y) / w3;
                                    const corners = [
                                        [-1.2, -1.2],
                                        [1.2, -1.2],
                                        [1.2, 1.2],
                                        [-1.2, 1.2],
                                    ].map(([x, y]) => project(cam, x, y, z(x, y)));
                                    p.polyline(
                                        corners.flatMap((c) => [c[0], c[1]]),
                                        {close: true, fill: C.yellow, fillAlpha: 0.13 * planeAlpha, color: rgba(C.yellow, 0.7 * planeAlpha), width: 1.5}
                                    );
                                    for (let i = -2; i <= 2; i++) {
                                        const t = i * 0.6;
                                        line3(p, [t, -1.2, z(t, -1.2)], [t, 1.2, z(t, 1.2)], rgba(C.yellow, 0.25 * planeAlpha));
                                        line3(p, [-1.2, t, z(-1.2, t)], [1.2, t, z(1.2, t)], rgba(C.yellow, 0.25 * planeAlpha));
                                    }
                                }
                                for (const o of order) if (!(planeAlpha > 0 && o.below)) p.point(o.u, o.v, o.c > 0.5 ? C.blue : C.orange, {r: 4});
                            }}
                        />
                        <Player timeline={tl} showSpeed={false} status={tl.t < 0.02 ? "Flat: no line separates the rings." : tl.t > 0.98 ? "Lifted to z = x² + y²: a flat plane separates them." : "Lifting…"} />
                    </>
                }
                side={
                    <div className="controls">
                        <p className="note">
                            Each point <M>{"(x, y)"}</M> is mapped to <M>{"\\varphi(x, y) = (x,\\, y,\\, x^2 + y^2)"}</M>. The inner ring stays low and the
                            outer ring rises, so a linear SVM in 3D can slide a flat plane between them.
                        </p>
                        <p className="note">
                            Where that plane cuts the bowl is a <strong>circle</strong>. Projected back to the floor (dashed), it is the curved boundary an
                            RBF or polynomial SVM draws in 2D.
                        </p>
                        <p className="note note-muted">
                            Kernels make this lift implicit: the SVM only ever needs dot products <M>{"\\varphi(\\vx)\\cdot\\varphi(\\vx')"}</M>, and{" "}
                            <M>{"K(\\vx, \\vx')"}</M> computes them directly, even for infinite-dimensional <M>{"\\varphi"}</M> like the RBF kernel's.
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
        id: "widest-street",
        title: "Widest street",
        prompt: <>Only the circled support vectors hold the street in place. Drag an uncircled point and nothing moves. Push one into the street and it narrows and tilts.</>,
        render: () => <SvmInstrument />,
    },
    {
        id: "soft-margin",
        title: "Soft margin",
        prompt: <>The classes overlap, so some points must sit inside the street (red rings). Slide C: small C buys a wide, tolerant street, and large C fights for every point.</>,
        render: () => <SvmInstrument kind="overlap" c={1} />,
    },
    {
        id: "outlier",
        title: "One outlier",
        prompt: <>One blue point sits deep in orange territory. With a huge C, the SVM contorts to respect it. Lower C until it gives up on the outlier and the street snaps back.</>,
        render: () => <SvmInstrument kind="outlier" c={1000} />,
    },
    {
        id: "rbf-rings",
        title: "RBF on rings",
        prompt: <>No line separates rings, but an RBF kernel curves the boundary around the inner cluster. Try switching back to Linear.</>,
        render: () => <SvmInstrument kind="circles" kernel={1} c={10} gamma={2} />,
    },
    {
        id: "gamma",
        title: "γ too high",
        prompt: <>With a huge γ each point's influence is a tiny bubble, so the boundary wraps individual points. That is overfitting. Lower γ until the moons reappear.</>,
        render: () => <SvmInstrument kind="moons" kernel={1} c={10} gamma={60} />,
    },
    {
        id: "poly-xor",
        title: "Polynomial XOR",
        prompt: <>A degree-2 polynomial kernel gives the model the feature x·y, which is exactly what XOR needs. Try degree 3, or switch to linear.</>,
        render: () => <SvmInstrument kind="xor" kernel={2} c={10} gamma={1} degree={2} />,
    },
    {
        id: "kernel-trick",
        title: "Kernel trick in 3D",
        prompt: <>Every point lifts to height x² + y². Drag to rotate: in 3D a flat plane separates the rings, and it meets the bowl in a circle.</>,
        render: () => <KernelTrick />,
    },
];

export default function Svm() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            Among all separating lines, an SVM picks the one with the <strong>widest street</strong>. The street's edges are{" "}
                            <M>{"\\vw\\cdot\\vx + b = \\pm 1"}</M>, so its width is <M>{"2/\\lVert\\vw\\rVert"}</M>. Points may enter the street at a cost{" "}
                            <M>{"C"}</M>:
                        </li>
                    </ul>
                    <MathBlock>{"\\min_{\\vw, b}\\; \\tfrac12\\lVert\\vw\\rVert^2 + C\\sum_i \\max\\big(0,\\; 1 - y_i(\\vw\\cdot\\vx_i + b)\\big)"}</MathBlock>
                    <ul>
                        <li>
                            The solution depends only on the <strong>support vectors</strong>, the points on or inside the street. All the others could be
                            deleted without changing anything.
                        </li>
                        <li>
                            In the dual problem the data appears only through dot products, so replacing <M>{"\\vx\\cdot\\vx'"}</M> with a kernel{" "}
                            <M>{"K(\\vx,\\vx')"}</M> (RBF <M>{"e^{-\\gamma\\lVert\\vx-\\vx'\\rVert^2}"}</M>, polynomial <M>{"(\\gamma\\vx\\cdot\\vx' + 1)^d"}</M>)
                            fits a linear separator in a higher-dimensional feature space. This is the <em>kernel trick</em>.
                        </li>
                        <li>
                            Here the dual is solved in WebAssembly with SMO, using LIBSVM's second-order working-set selection, every time you drag a point.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
