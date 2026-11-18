/**
 * Shared color palette. The same hues are used in the figures, the prose and
 * the equations, so that "the yellow arrow" in a picture is visibly the
 * yellow 𝐰 in the formula next to it.
 */
export const C = {
    bg: "#0e0d0b",
    canvas: "#12110f",
    panel: "#171613",
    grid: "rgba(186, 178, 160, 0.07)",
    gridMajor: "rgba(186, 178, 160, 0.14)",
    axis: "rgba(186, 178, 160, 0.38)",
    text: "#f1eee6",
    text2: "#c4bfb3",
    muted: "#8b867a",
    white: "#f7f5ef",

    blue: "#58c4dd",
    orange: "#f0ac5f",
    green: "#83c167",
    red: "#fc6255",
    yellow: "#ffd866",
    purple: "#a78bfa",
    teal: "#5cd0b3",
    pink: "#f38ba8",
} as const;

/** Class colors: class 0 = orange (negative), class 1 = blue (positive), … */
export const CLASS_COLORS = [C.orange, C.blue, C.green, C.purple, C.pink, C.teal, C.yellow, C.red];

export function classColor(c: number): string {
    return CLASS_COLORS[((c % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];
}

export type RGB = [number, number, number];

export function hexToRgb(hex: string): RGB {
    const h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

export function rgba(hex: string, alpha: number): string {
    const [r, g, b] = hexToRgb(hex);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function mix(a: RGB, b: RGB, t: number): RGB {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export const BG_RGB = hexToRgb(C.canvas);
export const CLASS_RGB = CLASS_COLORS.map(hexToRgb);

/** Sequential colormap for loss surfaces: ink → violet → magenta → amber → cream. */
const SEQ_STOPS: RGB[] = [
    [18, 16, 20],
    [36, 30, 78],
    [86, 44, 120],
    [150, 56, 122],
    [212, 92, 96],
    [245, 156, 82],
    [252, 226, 160],
];

export function sequential(t: number): RGB {
    const x = Math.min(1, Math.max(0, t)) * (SEQ_STOPS.length - 1);
    const i = Math.min(SEQ_STOPS.length - 2, Math.floor(x));
    return mix(SEQ_STOPS[i], SEQ_STOPS[i + 1], x - i);
}

/** Hex without "#": inside a TeX macro body "#" would start a parameter. */
const tex = (hex: string) => hex.replace("#", "");

/** TeX macros that color symbols to match the figures. */
export const TEX_MACROS: Record<string, string> = {
    "\\blu": `\\textcolor{${tex(C.blue)}}{#1}`,
    "\\org": `\\textcolor{${tex(C.orange)}}{#1}`,
    "\\grn": `\\textcolor{${tex(C.green)}}{#1}`,
    "\\red": `\\textcolor{${tex(C.red)}}{#1}`,
    "\\yel": `\\textcolor{${tex(C.yellow)}}{#1}`,
    "\\pur": `\\textcolor{${tex(C.purple)}}{#1}`,
    "\\tea": `\\textcolor{${tex(C.teal)}}{#1}`,
    "\\pnk": `\\textcolor{${tex(C.pink)}}{#1}`,
    "\\R": "\\mathbb{R}",
    "\\vw": `\\textcolor{${tex(C.yellow)}}{\\mathbf{w}}`,
    "\\vx": "\\mathbf{x}",
    "\\vu": "\\mathbf{u}",
    "\\vv": "\\mathbf{v}",
};
