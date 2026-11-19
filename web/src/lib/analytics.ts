/**
 * Privacy-friendly, cookieless visit counts with Umami (https://umami.is).
 *
 * Nothing loads unless `VITE_UMAMI_WEBSITE_ID` is set at build time. It is set
 * only in Netlify's production deploy context, so local dev, deploy previews
 * and forks send nothing. The ID is not a secret (it ships in the page, like
 * any analytics ID), but keeping it out of the repo means forks don't report
 * into this site's dashboard.
 *
 * Optional: `VITE_UMAMI_DOMAINS` (comma-separated hostnames to track, as an
 * extra guard) and `VITE_UMAMI_SRC` (a self-hosted tracker script URL).
 */

declare global {
    interface Window {
        umami?: {track: (event: string, data?: Record<string, string | number | boolean>) => void};
    }
}

const env = import.meta.env;
const WEBSITE_ID: string | undefined = env.VITE_UMAMI_WEBSITE_ID;
const DOMAINS: string | undefined = env.VITE_UMAMI_DOMAINS;
const SRC: string = env.VITE_UMAMI_SRC || "https://cloud.umami.is/script.js";

/** Add the tracker script. Pageviews, including client-side navigation, are tracked automatically. */
export function initAnalytics() {
    if (!WEBSITE_ID || typeof document === "undefined") return;
    const s = document.createElement("script");
    s.defer = true;
    s.src = SRC;
    s.dataset.websiteId = WEBSITE_ID;
    s.dataset.doNotTrack = "true"; // honor the browser's Do Not Track setting
    if (DOMAINS) s.dataset.domains = DOMAINS;
    document.head.appendChild(s);
}

/** Record a custom event (a no-op when analytics is off or blocked). */
export function track(event: string, data?: Record<string, string | number | boolean>) {
    try {
        window.umami?.track(event, data);
    } catch {
        // Analytics must never break the page.
    }
}
