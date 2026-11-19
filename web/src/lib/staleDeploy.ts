/**
 * Recovering from a deploy that happened while the page was open.
 *
 * Every playground is a lazily loaded chunk with a content hash in its name.
 * A deploy replaces those files, so a tab opened before it asks for chunks
 * that no longer exist, and the dynamic import fails. The fix is to reload
 * once onto the new deploy. A timestamp in sessionStorage stops this from
 * looping if a chunk is genuinely broken.
 */

const KEY = "hyperplane:stale-reload";
const WINDOW_MS = 20_000;

/** Whether an error means "this code chunk couldn't be loaded". */
export function isChunkLoadError(error: unknown): boolean {
    const msg = error instanceof Error ? `${error.name} ${error.message}` : String(error);
    return /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Unable to preload CSS|ChunkLoadError|Expected a JavaScript.*module script/i.test(msg);
}

/** Reload the page unless we already did so moments ago. Returns whether a reload started. */
export function reloadForNewDeploy(): boolean {
    try {
        const last = Number(window.sessionStorage.getItem(KEY) || 0);
        if (Date.now() - last < WINDOW_MS) return false;
        window.sessionStorage.setItem(KEY, String(Date.now()));
    } catch {
        // No sessionStorage (private mode): still reload, the browser stops obvious loops.
    }
    window.location.reload();
    return true;
}

/**
 * Vite reports failed dynamic-import preloads with this event. Reloading here
 * recovers before React even renders the error.
 */
export function watchForStaleDeploys() {
    window.addEventListener("vite:preloadError", (event) => {
        // Vite fires this for any failed dynamic import, including a module
        // that throws while evaluating; only a missing file means a new deploy.
        const error = (event as Event & {payload?: unknown}).payload;
        if (isChunkLoadError(error) && reloadForNewDeploy()) event.preventDefault();
    });
}
