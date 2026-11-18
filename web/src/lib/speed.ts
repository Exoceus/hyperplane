import {useSyncExternalStore} from "react";

/**
 * One playback speed for the whole site, chosen by the viewer and remembered
 * between visits. Every animation multiplies its own natural pace by it, so
 * "½×" means "half as fast as this playground would normally run", whether
 * that is a scrubbable timeline or a live simulation.
 */
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 10] as const;

const KEY = "hyperplane:speed";
const subscribers = new Set<() => void>();

function load(): number {
    try {
        const v = Number(window.localStorage.getItem(KEY));
        return (SPEEDS as readonly number[]).includes(v) ? v : 1;
    } catch {
        return 1;
    }
}

let current = typeof window === "undefined" ? 1 : load();

export function getSpeed(): number {
    return current;
}

export function setSpeed(v: number) {
    if (v === current) return;
    current = v;
    try {
        window.localStorage.setItem(KEY, String(v));
    } catch {
        // Private mode or blocked storage: the choice still lasts for this visit.
    }
    subscribers.forEach((fn) => fn());
}

function subscribe(fn: () => void) {
    subscribers.add(fn);
    return () => subscribers.delete(fn);
}

/** The current speed multiplier, re-rendering when the viewer changes it. */
export function useSpeed(): number {
    return useSyncExternalStore(subscribe, getSpeed, getSpeed);
}

export function speedLabel(v: number): string {
    if (v === 0.25) return "¼×";
    if (v === 0.5) return "½×";
    return `${v}×`;
}

/**
 * Turns a fractional amount of work per frame into whole steps, carrying the
 * remainder, so a simulation running at "¼×" takes one step every 4 frames.
 */
export function takeSteps(acc: {current: number}, amount: number, max = 10_000): number {
    acc.current += amount;
    const n = Math.floor(acc.current);
    acc.current -= n;
    return Math.min(max, n);
}
