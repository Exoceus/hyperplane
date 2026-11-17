import {tint} from "./painter";
import {CLASS_RGB, type RGB} from "./theme";

/**
 * Background color for a predicted probability of class 1: orange-ish for
 * p → 0, blue-ish for p → 1, fading into the canvas near p = 0.5.
 */
export function probabilityColor(p: number, strength = 0.5): RGB {
    const t = Math.min(1, Math.abs(p - 0.5) * 2);
    return tint(p >= 0.5 ? CLASS_RGB[1] : CLASS_RGB[0], 0.06 + strength * Math.pow(t, 0.85));
}

/** Region color for a class index with a confidence in [0, 1]. */
export function classRegionColor(c: number, confidence = 1, strength = 0.32): RGB {
    return tint(CLASS_RGB[c % CLASS_RGB.length], strength * confidence);
}
