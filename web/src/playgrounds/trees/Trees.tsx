import {useMemo, useState} from "react";
import {Button, Player, Segmented, Slider, Stat, Stats} from "../../components/Controls";
import {FigureLayout, Panel} from "../../components/Figure";
import {LineChart} from "../../components/LineChart";
import {M, MathBlock} from "../../components/Math";
import {Instrument, Playground, type Scenario} from "../../components/Playground";
import {Plot, type PlotPointer} from "../../components/Plot";
import {classRegionColor} from "../../lib/colors";
import {drawLabelled} from "../../lib/data";
import {ease, useTimeline} from "../../lib/hooks";
import {heatImage, type View} from "../../lib/painter";
import {C, classColor, rgba} from "../../lib/theme";
import {forest_grid, make_blobs, make_linear, make_shape, randomSeed, tree_build, tree_depth_curve} from "../../lib/wasm";

const VIEW: View = {xmin: -1.3, xmax: 1.3, ymin: -1.3, ymax: 1.3};
const STRIDE = 9;
const DEPTH_MAX = 14;

type Kind = "moons" | "xor" | "diagonal" | "blobs" | "spiral" | "circles";
const KINDS: {value: Kind; label: string}[] = [
    {value: "moons", label: "Moons"},
    {value: "xor", label: "XOR"},
    {value: "diagonal", label: "Diagonal"},
    {value: "circles", label: "Rings"},
    {value: "spiral", label: "Spiral"},
    {value: "blobs", label: "3 classes"},
];

function generate(kind: Kind, noise: number, seed: number): Float64Array {
    if (kind === "diagonal") return make_linear(160, 1, 1, 0, 0, noise * 0.3, seed);
    if (kind === "blobs") return make_blobs(180, 3, 0.15 + noise * 0.3, seed);
    return make_shape(kind, 180, noise, seed);
}

interface TreeNode {
    feature: number;
    threshold: number;
    left: number;
    right: number;
    majority: number;
    impurity: number;
    n: number;
    depth: number;
    parent: number;
    box: View;
    /** Order in which this node was split (−1 for leaves). */
    rank: number;
    /** x position in the diagram (leaf order). */
    x: number;
}

function parseTree(raw: Float64Array): TreeNode[] {
    const nodes: TreeNode[] = [];
    let rank = 0;
    for (let i = 0; i < raw.length; i += STRIDE) {
        const feature = raw[i];
        nodes.push({
            feature,
            threshold: raw[i + 1],
            left: raw[i + 2],
            right: raw[i + 3],
            majority: raw[i + 4],
            impurity: raw[i + 5],
            n: raw[i + 6],
            depth: raw[i + 7],
            parent: raw[i + 8],
            box: {...VIEW},
            rank: feature >= 0 ? rank++ : -1,
            x: 0,
        });
    }
    // Boxes, top-down (parents always precede children).
    for (const node of nodes) {
        if (node.feature < 0) continue;
        const b = node.box;
        const l = nodes[node.left];
        const r = nodes[node.right];
        if (node.feature === 0) {
            l.box = {...b, xmax: node.threshold};
            r.box = {...b, xmin: node.threshold};
        } else {
            l.box = {...b, ymax: node.threshold};
            r.box = {...b, ymin: node.threshold};
        }
    }
    // Diagram x positions: leaves in order, parents centered over children.
    let leaf = 0;
    const place = (id: number): number => {
        const n = nodes[id];
        if (n.feature < 0) return (n.x = leaf++);
        return (n.x = (place(n.left) + place(n.right)) / 2);
    };
    if (nodes.length) place(0);
    return nodes;
}

/** Which nodes act as leaves when only the first `splits` splits are done. */
function frontier(nodes: TreeNode[], splits: number): number[] {
    const out: number[] = [];
    const walk = (id: number) => {
        const n = nodes[id];
        if (n.feature >= 0 && n.rank < splits) {
            walk(n.left);
            walk(n.right);
        } else out.push(id);
    };
    if (nodes.length) walk(0);
    return out;
}

function pathTo(nodes: TreeNode[], splits: number, x: number, y: number): number[] {
    const path: number[] = [];
    let id = 0;
    while (nodes[id]) {
        path.push(id);
        const n = nodes[id];
        if (n.feature < 0 || n.rank >= splits) break;
        id = (n.feature === 0 ? x : y) < n.threshold ? n.left : n.right;
    }
    return path;
}

// ===========================================================================
// Single tree, grown split by split
// ===========================================================================

function TreeInstrument(props: {kind?: Kind; depth?: number; minLeaf?: number; noise?: number}) {
    const [kind, setKind] = useState<Kind>(props.kind ?? "moons");
    const [noise, setNoise] = useState(props.noise ?? 0.2);
    const [seed, setSeed] = useState(3);
    const [depth, setDepth] = useState(props.depth ?? 5);
    const [minLeaf, setMinLeaf] = useState(props.minLeaf ?? 1);
    const [criterion, setCriterion] = useState(0);
    const [hover, setHover] = useState<{x: number; y: number} | null>(null);

    const data = useMemo(() => generate(kind, noise, seed), [kind, noise, seed]);
    const test = useMemo(() => generate(kind, noise, seed + 7919), [kind, noise, seed]);
    const nodes = useMemo(() => parseTree(tree_build(data, depth, minLeaf, criterion)), [data, depth, minLeaf, criterion]);
    const splits = nodes.filter((n) => n.feature >= 0).length;
    const tl = useTimeline(splits, 3);
    const done = Math.floor(tl.t + 1e-9);
    const growing = tl.t - done;
    const leaves = frontier(nodes, done);
    const classes = kind === "blobs" ? 3 : 2;
    const maxImp = criterion === 1 ? Math.log2(classes) : 1 - 1 / classes;

    const curve = useMemo(() => {
        const c = tree_depth_curve(data, test, DEPTH_MAX, minLeaf, criterion);
        const train = new Float64Array(DEPTH_MAX);
        const held = new Float64Array(DEPTH_MAX);
        for (let i = 0; i < DEPTH_MAX; i++) {
            train[i] = c[2 * i];
            held[i] = c[2 * i + 1];
        }
        return {train, held};
    }, [data, test, minLeaf, criterion]);

    const path = hover ? pathTo(nodes, done, hover.x, hover.y) : [];
    const leafCount = Math.max(1, nodes.filter((n) => n.feature < 0).length);
    const maxDepth = nodes.reduce((m, n) => Math.max(m, n.depth), 0);
    const reset = () => tl.reset();

    const onPointer = (e: PlotPointer) => setHover(e.type === "leave" ? null : {x: e.x, y: e.y});
    const next = nodes.find((n) => n.rank === done);

    return (
        <Instrument hint="Press play to grow the tree one split at a time · hover the plane to trace a point down the tree">
            <FigureLayout
                sideWidth={300}
                main={
                    <>
                        <div className="duo">
                            <Panel label="Feature space" right={`${leaves.length} regions`}>
                                <Plot
                                    label="Decision tree regions"
                                    view={VIEW}
                                    onPointer={onPointer}
                                    draw={(p) => {
                                        p.clear();
                                        for (const id of leaves) {
                                            const n = nodes[id];
                                            const purity = 1 - Math.min(1, n.impurity / maxImp);
                                            const [r, g, b] = classRegionColor(n.majority, 0.25 + 0.75 * purity, 0.36);
                                            const bx = n.box;
                                            p.polyline([bx.xmin, bx.ymin, bx.xmax, bx.ymin, bx.xmax, bx.ymax, bx.xmin, bx.ymax], {
                                                close: true,
                                                fill: `rgb(${r},${g},${b})`,
                                                color: "none",
                                            });
                                        }
                                        if (path.length) {
                                            const bx = nodes[path[path.length - 1]].box;
                                            p.polyline([bx.xmin, bx.ymin, bx.xmax, bx.ymin, bx.xmax, bx.ymax, bx.xmin, bx.ymax], {close: true, color: C.yellow, width: 2});
                                        }
                                        p.grid(0.25, {axes: false});
                                        for (const n of nodes) {
                                            if (n.feature < 0 || n.rank > done) continue;
                                            const f = n.rank === done ? ease(growing) : 1;
                                            if (f <= 0) continue;
                                            const b = n.box;
                                            if (n.feature === 0) {
                                                const mid = (b.ymin + b.ymax) / 2;
                                                const h = ((b.ymax - b.ymin) / 2) * f;
                                                p.line(n.threshold, mid - h, n.threshold, mid + h, {color: C.white, width: n.rank === done ? 2.5 : 1.5});
                                            } else {
                                                const mid = (b.xmin + b.xmax) / 2;
                                                const w = ((b.xmax - b.xmin) / 2) * f;
                                                p.line(mid - w, n.threshold, mid + w, n.threshold, {color: C.white, width: n.rank === done ? 2.5 : 1.5});
                                            }
                                        }
                                        drawLabelled(p, data, {r: 3.8});
                                        if (hover) p.circle(hover.x, hover.y, 5, {fill: C.yellow, stroke: C.canvas, width: 2});
                                    }}
                                />
                            </Panel>
                            <Panel label="The tree" right={`depth ${Math.min(maxDepth, nodes.filter((n) => n.rank >= 0 && n.rank < done).reduce((m, n) => Math.max(m, n.depth + 1), 0))}`}>
                                <Plot
                                    label="Tree diagram"
                                    view={{xmin: -0.8, xmax: leafCount - 0.2, ymin: -(maxDepth + 0.6), ymax: 0.6}}
                                    aspect={1}
                                    draw={(p) => {
                                        p.clear();
                                        const visible = (n: TreeNode) => n.parent < 0 || nodes[n.parent].rank < done || (nodes[n.parent].rank === done && growing > 0);
                                        for (const n of nodes) {
                                            if (n.parent < 0 || !visible(n)) continue;
                                            const par = nodes[n.parent];
                                            const onPath = path.includes(n.parent) && path.includes(nodes.indexOf(n));
                                            const a = par.rank === done ? ease(growing) : 1;
                                            p.line(par.x, -par.depth, par.x + (n.x - par.x) * a, -par.depth - (n.depth - par.depth) * a, {
                                                color: onPath ? C.yellow : rgba(C.muted, 0.6),
                                                width: onPath ? 2.5 : 1.2,
                                            });
                                        }
                                        const r = Math.max(3, Math.min(10, (p.width / leafCount) * 0.28));
                                        nodes.forEach((n, id) => {
                                            if (!visible(n)) return;
                                            const isLeafNow = n.feature < 0 || n.rank >= done;
                                            const onPath = path.includes(id);
                                            p.circle(n.x, -n.depth, r, {
                                                fill: isLeafNow ? classColor(n.majority) : C.canvas,
                                                stroke: onPath ? C.yellow : isLeafNow ? C.canvas : classColor(n.majority),
                                                width: onPath ? 3 : 2,
                                            });
                                            if (!isLeafNow && leafCount <= 18)
                                                p.text(`${n.feature === 0 ? "x" : "y"} < ${n.threshold.toFixed(2)}`, n.x, -n.depth, {
                                                    color: C.text2,
                                                    size: 10.5,
                                                    font: "mono",
                                                    align: "center",
                                                    dy: -r - 5,
                                                });
                                        });
                                    }}
                                />
                            </Panel>
                        </div>
                        <Player
                            timeline={tl}
                            speeds={[1.5, 3, 10]}
                            status={
                                next && tl.t < splits ? (
                                    <>
                                        Split {done + 1} of {splits}: the region with {next.n} points is cut at{" "}
                                        <b className="mono">
                                            {next.feature === 0 ? "x" : "y"} = {next.threshold.toFixed(2)}
                                        </b>
                                        , the cut that makes the two halves purest.
                                    </>
                                ) : (
                                    <>Fully grown: {splits} splits, {leafCount} leaves.</>
                                )
                            }
                        />
                    </>
                }
                side={
                    <div className="controls">
                        <Segmented value={kind} onChange={(v) => (setKind(v), reset())} options={KINDS} />
                        <Slider label="max depth" value={depth} min={1} max={DEPTH_MAX} step={1} onChange={(v) => (setDepth(v), reset())} format={(v) => v.toFixed(0)} color={C.yellow} />
                        <Slider label="min points per leaf" value={minLeaf} min={1} max={30} step={1} onChange={(v) => (setMinLeaf(v), reset())} format={(v) => v.toFixed(0)} />
                        <Slider label="noise" value={noise} min={0} max={0.6} onChange={(v) => (setNoise(v), reset())} />
                        <Segmented
                            label="impurity"
                            value={criterion}
                            onChange={(v) => (setCriterion(v), reset())}
                            options={[
                                {value: 0, label: "Gini"},
                                {value: 1, label: "Entropy"},
                            ]}
                        />
                        <div className="panel-label">
                            <span>Accuracy vs. depth</span>
                        </div>
                        <LineChart
                            label="Training and test accuracy for each depth"
                            series={[
                                {values: curve.train, color: C.muted, x0: 1, dash: [4, 3]},
                                {values: curve.held, color: C.green, x0: 1},
                            ]}
                            height={100}
                            xMin={1}
                            yMax={1}
                            marker={depth}
                            yFormat={(v) => `${Math.round(v * 100)}%`}
                            onPick={(x) => (setDepth(Math.max(1, Math.min(DEPTH_MAX, Math.round(x)))), reset())}
                        />
                        <Stats>
                            <Stat label="test accuracy" value={`${Math.round(curve.held[depth - 1] * 100)}%`} color={C.green} />
                            <Stat label="leaves" value={leafCount} />
                        </Stats>
                        <div className="control-row">
                            <Button onClick={() => (setSeed(randomSeed()), reset())}>New data</Button>
                        </div>
                    </div>
                }
            />
        </Instrument>
    );
}

// ===========================================================================
// One tree vs. a forest
// ===========================================================================

const G = 150;

function ForestInstrument(props: {kind?: Kind; trees?: number; depth?: number}) {
    const [kind, setKind] = useState<Kind>(props.kind ?? "moons");
    const [seed, setSeed] = useState(8);
    const [trees, setTrees] = useState(props.trees ?? 50);
    const [depth, setDepth] = useState(props.depth ?? 10);
    const data = useMemo(() => generate(kind, 0.35, seed), [kind, seed]);

    const tree = useMemo(() => parseTree(tree_build(data, depth, 1, 0)), [data, depth]);
    const forest = useMemo(() => {
        const g = forest_grid(data, trees, depth, 1, seed, VIEW.xmin, VIEW.xmax, VIEW.ymin, VIEW.ymax, G, G);
        return heatImage(g, G, G, (c, i) => classRegionColor(c, Math.max(0, (g[2 * i + 1] - 0.5) * 2) * 0.85 + 0.15, 0.36), 2);
    }, [data, trees, depth, seed]);
    const leaves = tree.filter((n) => n.feature < 0);

    return (
        <Instrument hint="Same data, same depth: one tree versus the vote of many randomized trees">
            <FigureLayout
                sideWidth={280}
                main={
                    <div className="duo">
                        <Panel label="One deep tree">
                            <Plot
                                label="Single decision tree regions"
                                view={VIEW}
                                draw={(p) => {
                                    p.clear();
                                    for (const n of leaves) {
                                        const [r, g, b] = classRegionColor(n.majority, 1, 0.36);
                                        const bx = n.box;
                                        p.polyline([bx.xmin, bx.ymin, bx.xmax, bx.ymin, bx.xmax, bx.ymax, bx.xmin, bx.ymax], {close: true, fill: `rgb(${r},${g},${b})`, color: "none"});
                                    }
                                    p.grid(0.25, {axes: false});
                                    drawLabelled(p, data, {r: 3.6});
                                }}
                            />
                        </Panel>
                        <Panel label={`Random forest · ${trees} trees`}>
                            <Plot
                                label="Random forest vote"
                                view={VIEW}
                                draw={(p) => {
                                    p.clear();
                                    p.image(forest, VIEW);
                                    p.grid(0.25, {axes: false});
                                    drawLabelled(p, data, {r: 3.6});
                                }}
                            />
                        </Panel>
                    </div>
                }
                side={
                    <div className="controls">
                        <Segmented value={kind} onChange={setKind} options={KINDS} />
                        <Slider label="trees in the forest" value={trees} min={1} max={150} step={1} onChange={setTrees} format={(v) => v.toFixed(0)} color={C.green} />
                        <Slider label="max depth" value={depth} min={1} max={DEPTH_MAX} step={1} onChange={setDepth} format={(v) => v.toFixed(0)} color={C.yellow} />
                        <p className="note">
                            Each forest tree trains on a <strong>bootstrap</strong> resample of the data, and every split considers only a random feature. The
                            trees make different mistakes, so their vote averages the noise away. The shading shows how unanimous the vote is.
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
// Playground
// ===========================================================================

const SCENARIOS: Scenario[] = [
    {
        id: "grow",
        title: "Grow a tree",
        prompt: <>Press play. Each split asks one yes/no question about one feature, chosen greedily to make the two halves as pure as possible. Hover to trace a point down the tree.</>,
        render: () => <TreeInstrument />,
    },
    {
        id: "too-deep",
        title: "Too deep",
        prompt: <>Depth 14 on noisy data: the tree carves a tiny box around almost every point and hits 100% on training data. Check the green test curve, then find the depth where it peaks.</>,
        render: () => <TreeInstrument depth={14} noise={0.4} />,
    },
    {
        id: "diagonal",
        title: "Diagonal boundary",
        prompt: <>The true boundary is a diagonal line, but trees can only cut parallel to the axes, so they build a staircase. Raise the depth and watch the steps shrink.</>,
        render: () => <TreeInstrument kind="diagonal" depth={4} noise={0} />,
    },
    {
        id: "xor",
        title: "Greedy blindness",
        prompt: <>On XOR no single cut helps by itself, so the first split is nearly arbitrary. The second level fixes it: greedy splitting can't see two moves ahead.</>,
        render: () => <TreeInstrument kind="xor" depth={2} noise={0.05} />,
    },
    {
        id: "forest",
        title: "Forest vs. tree",
        prompt: <>One deep tree is jagged and memorizes the noise. Many randomized trees voting together give a smoother, more trustworthy boundary. Slide the number of trees from 1 upward.</>,
        render: () => <ForestInstrument />,
    },
    {
        id: "forest-spiral",
        title: "Forest on a spiral",
        prompt: <>Axis-aligned boxes can still follow a spiral if you stack enough of them. Lower the depth to see where the forest gives up.</>,
        render: () => <ForestInstrument kind="spiral" trees={80} depth={12} />,
    },
];

export default function Trees() {
    return (
        <Playground
            scenarios={SCENARIOS}
            explainer={
                <>
                    <ul>
                        <li>
                            A decision tree asks a series of yes/no questions such as <M>{"x < 0.31"}</M>. Each leaf is an axis-aligned box that predicts its
                            majority class.
                        </li>
                        <li>
                            Trees are grown greedily (CART). At each node, try every feature and threshold, and keep the split that most reduces the weighted
                            impurity of the children:
                        </li>
                    </ul>
                    <MathBlock>{"\\text{Gini}(S) = 1 - \\sum_c p_c^2, \\qquad \\text{Entropy}(S) = -\\sum_c p_c \\log_2 p_c"}</MathBlock>
                    <ul>
                        <li>
                            Deep trees fit any training set (low bias) but chase noise (high variance). Limit the depth or the leaf size, or average many
                            trees.
                        </li>
                        <li>
                            A <strong>random forest</strong> trains each tree on a bootstrap sample with random feature choices, then takes a vote. Averaging
                            many decorrelated, overfit trees cuts the variance without adding much bias.
                        </li>
                    </ul>
                </>
            }
        />
    );
}
