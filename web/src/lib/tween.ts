import {useRef, useState} from "react";
import {ease, useAnimationFrame} from "./hooks";

/**
 * Animate a numeric vector toward a target over `duration` seconds.
 * Returns `[current, animateTo, setImmediately]`.
 */
export function useTween(initial: number[], duration = 0.6): [number[], (to: number[]) => void, (v: number[]) => void] {
    const [value, setValue] = useState(initial);
    const anim = useRef<{from: number[]; to: number[]; t: number} | null>(null);
    const [active, setActive] = useState(false);

    useAnimationFrame((dt) => {
        const a = anim.current;
        if (!a) return;
        a.t = Math.min(1, a.t + dt / duration);
        const e = ease(a.t);
        setValue(a.from.map((f, i) => f + (a.to[i] - f) * e));
        if (a.t >= 1) {
            anim.current = null;
            setActive(false);
        }
    }, active);

    const animateTo = (to: number[]) => {
        anim.current = {from: value, to, t: 0};
        setActive(true);
    };
    const set = (v: number[]) => {
        anim.current = null;
        setActive(false);
        setValue(v);
    };
    return [value, animateTo, set];
}
